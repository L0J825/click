import os
import tempfile

import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np
import time
import sys
from collections import deque

try:
    from thop import profile
except ImportError:
    print("Warning: thop is not installed. Please install it using 'pip install thop' for FLOPs calculation.")

sys.path.append('/root/7.3T/RJ/mcdc')
from src.utils import calculate_total_bitstream_size, calculate_psnr
from src.Modules.DCVC_DC.models.video_model import DMC
from src.Modules.DCVC_DC.models.image_model import IntraNoAR
from src.Modules.DCVC_DC.utils.stream_helper import get_padding_size, get_state_dict
from src.Modules.LCEN.LCEN import LCEN_v6 as LCEN
from src.Modules.Restoration.Enhancer import RFDMNet


class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.i_frame_model = IntraNoAR(anchor_num=4, ec_thread=args.ec_thread,
                                       stream_part=args.stream_part_i, inplace=True)
        self.p_frame_model = DMC(ec_thread=args.ec_thread, stream_part=args.stream_part_p,
                                 inplace=True)
        self.enhance_model = RFDMNet(in_dim=self.args.in_dim, hidden_dim1=self.args.hidden_dim1,
                                         hidden_dim2=self.args.hidden_dim2, num_layers=self.args.num_layers,
                                         use_checkpoint=False)
        self.lossless_model = LCEN(bit_depth=args.bit_depth, blocks=args.blocks)

        if args.write_stream:
            self.p_frame_model.update(force=True)
            self.i_frame_model.update(force=True)

    def encode_decode(self, npy_filepath, skip_dec=False):
        assert os.path.splitext(npy_filepath)[-1] == '.npy', "必须传入npy文件"
        tmp_dir = self.args.temp_dir
        lossy_bin_folder = tmp_dir
        bin_path = os.path.join(lossy_bin_folder, f"lossy.bin")
        psnrs = []
        psnrs_enh = []
        lossy_bits = []
        lossless_bitstreams = []

        lossy_compress_time = 0
        lossless_compress_time = 0
        lossy_decompress_time = 0
        lossless_decompress_time = 0
        enhance_compress_time = 0
        enhance_decompress_time = 0

        ori_slices = torch.from_numpy(np.load(npy_filepath).astype(np.float32)).to(self.args.device)
        if self.args.bit_depth == 16:
            max_value = ori_slices.max()
        else:
            max_value = 255.0
        num_slices, height, width = ori_slices.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

        for frame_idx in range(num_slices):
            torch.cuda.synchronize()
            lossy_start_time = time.time()
            x = ori_slices[frame_idx]
            x = x / max_value
            x = torch.stack([x, x, x])
            x_padded = torch.nn.functional.pad(
                x.unsqueeze(0), (padding_l, padding_r, padding_t, padding_b), mode="replicate",
            ).to(self.args.device)
            if frame_idx == 0:
                lossy_com_result, lossy_string = self.i_frame_model.encode_decode(
                    x_padded, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.i_frame_q_indexes,
                    pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['x_hat'][0][0] * max_value, min=0,
                                              max=max_value).type(torch.int32)
            else:
                lossy_com_result, lossy_string = self.p_frame_model.encode_decode(
                    x_padded, dpb, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.p_frame_q_indexes,
                    frame_idx=self.args.frame_idx, pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['dpb']["ref_frame"][0][0] * max_value, min=0,
                                              max=max_value).type(torch.int32)
            torch.cuda.synchronize()
            lossy_compress_time += (time.time() - lossy_start_time)
            lossy_bits.append(lossy_com_result['bit'])

            psnr = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=max_value)
            psnrs.append(psnr)
            torch.cuda.synchronize()
            enhance_start_time = time.time()
            if frame_idx == 0:
                ref_frames = deque([ori_slices[0], ori_slices[0], ori_slices[0]])
            else:
                frames = [ref_frames[i] for i in range(3)] + [lossy_rec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                frames = torch.stack(frames, dim=0).unsqueeze(0) / max_value  # 形状为 (4, h, w)
                out = self.enhance_model(frames)
                lossy_rec_frame = torch.clamp(out[0][0] * max_value, min=0, max=max_value).type(torch.int32)

            torch.cuda.synchronize()
            enhance_compress_time += (time.time() - enhance_start_time)
            psnr_enh = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=max_value)
            psnrs_enh.append(psnr_enh)

            torch.cuda.synchronize()
            lossless_start_time = time.time()
            lossy_rec_frame = lossy_rec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            ori_frame = ori_slices[frame_idx].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            residues = ori_frame.type(torch.float32) - lossy_rec_frame.type(torch.float32)
            if self.args.bit_depth == 16:
                res_min = int(residues.min())
                res_max = int(residues.max())
            else:
                res_min = -255
                res_max = 255

            if frame_idx == 0:
                ref_frame = lossy_rec_frame * 1.
            else:
                ref_frame = ori_slices[frame_idx - 1].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            lossless_stream = self.lossless_model.compress(residues=residues, x_min=res_min, x_max=res_max,
                                                           cur=lossy_rec_frame, ref=ref_frame)
            torch.cuda.synchronize()
            lossless_compress_time += (time.time() - lossless_start_time)
            lossless_bitstreams.append(lossless_stream)

            if not skip_dec: 
                torch.cuda.synchronize()
                lossy_start_time = time.time()
                if frame_idx == 0:
                    lossy_dec_result = self.i_frame_model.decompress(lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.i_frame_q_indexes)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["x_hat"][0][0] * max_value, min=0,
                                                  max=max_value).type(torch.int32)
                else:
                    lossy_dec_result = self.p_frame_model.decompress(dpb, lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.p_frame_q_indexes,
                                                                     0)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["dpb"]["ref_frame"][0][0] * max_value, min=0,
                                                  max=max_value).type(torch.int32)
                torch.cuda.synchronize()
                lossy_decompress_time += (time.time() - lossy_start_time)

                torch.cuda.synchronize()
                enhance_start_time = time.time()
                if frame_idx == 0:
                    pass
                else:
                    frames = [ref_frames[i] for i in range(3)] + [lossy_dec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                    frames = torch.stack(frames, dim=0).unsqueeze(0) / max_value  # 形状为 (4, h, w)
                    out = self.enhance_model(frames)
                    lossy_dec_frame = torch.clamp(out[0][0] * max_value, min=0, max=max_value).type(torch.int32)
                torch.cuda.synchronize()
                enhance_decompress_time += (time.time() - enhance_start_time)
                lossy_dec_frame = lossy_dec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.
                torch.cuda.synchronize()
                lossless_start_time = time.time()
                residues_dec = self.lossless_model.decompress(strings=lossless_stream, x_min=res_min, x_max=res_max,
                                                              cur=lossy_dec_frame, ref=ref_frame)
                torch.cuda.synchronize()
                lossless_decompress_time += (time.time() - lossless_start_time)
                if not torch.equal(lossy_dec_frame, lossy_rec_frame):
                    print("lossy_dec_frame != lossy_rec_frame at {}".format(frame_idx))
                if not torch.equal(residues_dec, residues):
                    print("residues_dec != residues at {}".format(frame_idx))
            ref_frames.popleft()
            ref_frames.append(ori_slices[frame_idx])
            dpb = {
                "ref_frame": x_padded,
                "ref_feature": None, "ref_mv_feature": None,
                "ref_y": None, "ref_mv_y": None,
            }
        lossless_bit_list = calculate_total_bitstream_size(lossless_bitstreams)
        # single channel for grey-scale image
        lossy_bpp_list = [x / 3 / height / width for x in lossy_bits]
        lossless_bpp_list = [x / height / width for x in lossless_bit_list]

        lossy_bpp = sum(lossy_bpp_list) / num_slices
        lossless_bpp = sum(lossless_bpp_list) / num_slices
        bpp = lossless_bpp + lossy_bpp

        return {"bpp": bpp, "lossy_bpp": lossy_bpp, "lossless_bpp": lossless_bpp,
                "lossy_bpp_list": lossy_bpp_list, "lossless_bpp_list": lossless_bpp_list,
                "psnrs": psnrs, "psnrs_enh": psnrs_enh,
                "lossy_compress_time": lossy_compress_time / num_slices,
                "lossy_decompress_time": lossy_decompress_time / num_slices,
                "lossless_compress_time": lossless_compress_time / num_slices,
                "lossless_decompress_time": lossless_decompress_time / num_slices,
                "enhance_compress_time": enhance_compress_time / num_slices,
                "enhance_decompress_time": enhance_decompress_time / num_slices,
                }

    def resume(self):
        load = self.args.i_frame_model_path
        load_dict = get_state_dict(load)
        self.i_frame_model.load_state_dict(load_dict)
        print(f"i_frame_model loaded from {load}")

        load = self.args.p_frame_model_path
        load_dict = get_state_dict(load)
        self.p_frame_model.load_state_dict(load_dict)
        print(f"p_frame_model loaded from {load}")

        load = self.args.enhancer_model_path
        if load != None:
            load_dict = torch.load(load, map_location=lambda storage, loc: storage)
            self.enhance_model.load_state_dict(load_dict['state_dict'] if 'state_dict' in load_dict else load_dict)
            print(f"enhance_model loaded from {load}")

        load = self.args.lossless_model_path
        if load != None:
            load_dict = torch.load(load, map_location=lambda storage, loc: storage)
            self.lossless_model.load_state_dict(load_dict['state_dict'] if 'state_dict' in load_dict else load_dict)
            print(f"lossless_model loaded from {load}")
        else:
            print("没有找到 lossless_model")

        # Move the model components to the correct device
        device = self.args.device
        self.i_frame_model = self.i_frame_model.to(device)
        self.p_frame_model = self.p_frame_model.to(device)
        self.enhance_model = self.enhance_model.to(device)
        self.lossless_model = self.lossless_model.to(device)
        print(f"All model components moved to device: {device}")


class Net_8b(Net):

    def __init__(self, args):
        super().__init__(args)

    def encode_decode(self, npy_filepath, skip_dec=False):
        assert os.path.splitext(npy_filepath)[-1] == '.npy', "必须传入npy文件"
        tmp_dir = self.args.temp_dir
        lossy_bin_folder = tmp_dir
        bin_path = os.path.join(lossy_bin_folder, f"lossy.bin")


        psnrs = []
        psnrs_enh = []
        lossy_bits = []
        lossless_bitstreams = []

        lossy_compress_time = 0
        lossless_compress_time = 0
        lossy_decompress_time = 0
        lossless_decompress_time = 0
        enhance_compress_time = 0
        enhance_decompress_time = 0

        ori_slices = torch.from_numpy(np.load(npy_filepath)).to(self.args.device)
        num_slices, height, width = ori_slices.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

        for frame_idx in range(num_slices):
            torch.cuda.synchronize()
            lossy_start_time = time.time()
            x = ori_slices[frame_idx]
            x = x / 255.0
            x = torch.stack([x, x, x])
            x_padded = torch.nn.functional.pad(
                x.unsqueeze(0), (padding_l, padding_r, padding_t, padding_b), mode="replicate",
            ).to(self.args.device)
            if frame_idx == 0:
                lossy_com_result, lossy_string = self.i_frame_model.encode_decode(
                    x_padded, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.i_frame_q_indexes,
                    pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['x_hat'][0][0] * 255, min=0,
                                              max=255).type(torch.uint8)


            else:
                lossy_com_result, lossy_string = self.p_frame_model.encode_decode(
                    x_padded, dpb, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.p_frame_q_indexes,
                    frame_idx=self.args.frame_idx, pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['dpb']["ref_frame"][0][0] * 255, min=0,
                                              max=255).type(torch.uint8)

            torch.cuda.synchronize()
            lossy_compress_time += (time.time() - lossy_start_time)
            lossy_bits.append(lossy_com_result['bit'])

            psnr = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=255.0)
            psnrs.append(psnr)

            torch.cuda.synchronize()
            enhance_start_time = time.time()
            if frame_idx == 0:
                ref_frames = deque([ori_slices[0], ori_slices[0], ori_slices[0]])
            else:
                frames = [ref_frames[i] for i in range(3)] + [lossy_rec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                frames = torch.stack(frames, dim=0).unsqueeze(0) / 255.0  # 形状为 (4, h, w)
                out = self.enhance_model(frames)
                lossy_rec_frame = torch.clamp(out[0][0] * 255, min=0, max=255).type(torch.uint8)

            torch.cuda.synchronize()
            enhance_compress_time += (time.time() - enhance_start_time)
            psnr_enh = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=255.0)
            psnrs_enh.append(psnr_enh)

            torch.cuda.synchronize()
            lossless_start_time = time.time()
            lossy_rec_frame = lossy_rec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            ori_frame = ori_slices[frame_idx].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            residues = ori_frame.type(torch.float32) - lossy_rec_frame.type(torch.float32)

            if frame_idx == 0:
                ref_frame = lossy_rec_frame * 1.
            else:
                ref_frame = ori_slices[frame_idx - 1].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            lossless_stream = self.lossless_model.compress(residues=residues, x_min=-255, x_max=255,
                                                           cur=lossy_rec_frame, ref=ref_frame)
            torch.cuda.synchronize()
            lossless_compress_time += (time.time() - lossless_start_time)
            lossless_bitstreams.append(lossless_stream)

            if not skip_dec:  # 快速测试，跳过解码过程
                torch.cuda.synchronize()
                lossy_start_time = time.time()
                if frame_idx == 0:
                    lossy_dec_result = self.i_frame_model.decompress(lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.i_frame_q_indexes)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["x_hat"][0][0] * 255, min=0,
                                                  max=255).type(torch.uint8)
                else:
                    lossy_dec_result = self.p_frame_model.decompress(dpb, lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.p_frame_q_indexes,
                                                                     0)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["dpb"]["ref_frame"][0][0] * 255, min=0,
                                                  max=255).type(torch.uint8)
                torch.cuda.synchronize()
                lossy_decompress_time += (time.time() - lossy_start_time)

                torch.cuda.synchronize()
                enhance_start_time = time.time()
                if frame_idx == 0:
                    pass
                else:
                    frames_dec = [ref_frames[i] for i in range(3)] + [lossy_dec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                    frames_dec = torch.stack(frames_dec, dim=0).unsqueeze(0) / 255.0  # 形状为 (4, h, w)
                    out_dec = self.enhance_model(frames_dec)
                    lossy_dec_frame = torch.clamp(out_dec[0][0] * 255, min=0, max=255).type(torch.uint8)
                torch.cuda.synchronize()
                enhance_decompress_time += (time.time() - enhance_start_time)
                lossy_dec_frame = lossy_dec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.0
                
                torch.cuda.synchronize()
                lossless_start_time = time.time()
                residues_dec = self.lossless_model.decompress(strings=lossless_stream, x_min=-255, x_max=255,
                                                              cur=lossy_dec_frame, ref=ref_frame)
                torch.cuda.synchronize()
                lossless_decompress_time += (time.time() - lossless_start_time)
                if not torch.equal(lossy_dec_frame, lossy_rec_frame):
                    print("lossy_dec_frame != lossy_rec_frame at {}".format(frame_idx))
                if not torch.equal(residues_dec, residues):
                    print("residues_dec != residues at {}".format(frame_idx))
            ref_frames.popleft()
            ref_frames.append(ori_slices[frame_idx])
            dpb = {
                "ref_frame": x_padded,
                "ref_feature": None, "ref_mv_feature": None,
                "ref_y": None, "ref_mv_y": None,
            }
        lossless_bit_list = calculate_total_bitstream_size(lossless_bitstreams)
        lossy_bpp_list = [x / 3 / height / width for x in lossy_bits]
        lossless_bpp_list = [x / height / width for x in lossless_bit_list]

        lossy_bpp = sum(lossy_bpp_list) / num_slices
        lossless_bpp = sum(lossless_bpp_list) / num_slices
        bpp = lossless_bpp + lossy_bpp

        return {"bpp": bpp, "lossy_bpp": lossy_bpp, "lossless_bpp": lossless_bpp,
                "lossy_bpp_list": lossy_bpp_list, "lossless_bpp_list": lossless_bpp_list,
                "psnrs": psnrs, "psnrs_enh": psnrs_enh,
                "lossy_compress_time": lossy_compress_time / num_slices,
                "lossy_decompress_time": lossy_decompress_time / num_slices,
                "lossless_compress_time": lossless_compress_time / num_slices,
                "lossless_decompress_time": lossless_decompress_time / num_slices,
                "enhance_compress_time": enhance_compress_time / num_slices,
                "enhance_decompress_time": enhance_decompress_time / num_slices,
                }


class Net_16b(Net):

    def __init__(self, args):
        super().__init__(args)

    def encode_decode(self, npy_filepath, skip_dec=False):
        assert os.path.splitext(npy_filepath)[-1] == '.npy', "必须传入npy文件"
        tmp_dir = self.args.temp_dir
        lossy_bin_folder = tmp_dir
        bin_path = os.path.join(lossy_bin_folder, f"lossy.bin")

        psnrs = []
        psnrs_enh = []
        lossy_bits = []
        lossless_bitstreams = []

        lossy_compress_time = 0
        lossless_compress_time = 0
        lossy_decompress_time = 0
        lossless_decompress_time = 0
        enhance_compress_time = 0
        enhance_decompress_time = 0

        ori_slices = torch.from_numpy(np.load(npy_filepath).astype(np.float32)).to(self.args.device)
        max_value = ori_slices.max()
        num_slices, height, width = ori_slices.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

        for frame_idx in range(num_slices):
            torch.cuda.synchronize()
            lossy_start_time = time.time()
            x = ori_slices[frame_idx]
            x = x / max_value
            x = torch.stack([x, x, x])
            x_padded = torch.nn.functional.pad(
                x.unsqueeze(0), (padding_l, padding_r, padding_t, padding_b), mode="replicate",
            ).to(self.args.device)
            if frame_idx == 0:
                lossy_com_result, lossy_string = self.i_frame_model.encode_decode(
                    x_padded, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.i_frame_q_indexes,
                    pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['x_hat'][0][0] * max_value, min=0,
                                              max=max_value).type(torch.int32)
            else:
                lossy_com_result, lossy_string = self.p_frame_model.encode_decode(
                    x_padded, dpb, output_path=bin_path,
                    q_in_ckpt=self.args.q_in_ckpt, q_index=self.args.p_frame_q_indexes,
                    frame_idx=self.args.frame_idx, pic_height=height, pic_width=width,
                )
                lossy_rec_frame = torch.clamp(lossy_com_result['dpb']["ref_frame"][0][0] * max_value, min=0,
                                              max=max_value).type(torch.int32)
            torch.cuda.synchronize()
            lossy_compress_time += (time.time() - lossy_start_time)
            lossy_bits.append(lossy_com_result['bit'])

            psnr = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=max_value)
            psnrs.append(psnr)

            torch.cuda.synchronize()
            enhance_start_time = time.time()
            if frame_idx == 0:

                ref_frames = deque([ori_slices[0], ori_slices[0], ori_slices[0]])
            else:
                frames = [ref_frames[i] for i in range(3)] + [lossy_rec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                frames = torch.stack(frames, dim=0).unsqueeze(0) / max_value  # 形状为 (4, h, w)
                out = self.enhance_model(frames)
                lossy_rec_frame = torch.clamp(out[0][0] * max_value, min=0, max=max_value).type(torch.int32)

            torch.cuda.synchronize()
            enhance_compress_time += (time.time() - enhance_start_time)
            psnr_enh = calculate_psnr(lossy_rec_frame, ori_slices[frame_idx], max_value=max_value)
            psnrs_enh.append(psnr_enh)

            torch.cuda.synchronize()
            lossless_start_time = time.time()
            lossy_rec_frame = lossy_rec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            ori_frame = ori_slices[frame_idx].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            residues = ori_frame.type(torch.float32) - lossy_rec_frame.type(torch.float32)
            res_min = int(residues.min())
            res_max = int(residues.max())
            if frame_idx == 0:
                ref_frame = lossy_rec_frame * 1.
            else:
                ref_frame = ori_slices[frame_idx - 1].unsqueeze(dim=0).unsqueeze(dim=0) * 1.
            lossless_stream = self.lossless_model.compress(residues=residues, x_min=res_min, x_max=res_max,
                                                           cur=lossy_rec_frame, ref=ref_frame)
            torch.cuda.synchronize()
            lossless_compress_time += (time.time() - lossless_start_time)
            lossless_bitstreams.append(lossless_stream)

            if not skip_dec:  # 快速测试，跳过解码过程
                torch.cuda.synchronize()
                lossy_start_time = time.time()
                if frame_idx == 0:
                    lossy_dec_result = self.i_frame_model.decompress(lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.i_frame_q_indexes)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["x_hat"][0][0] * max_value, min=0,
                                                  max=max_value).type(torch.int32)
                else:
                    lossy_dec_result = self.p_frame_model.decompress(dpb, lossy_string, height, width,
                                                                     self.args.q_in_ckpt, self.args.p_frame_q_indexes,
                                                                     0)
                    lossy_dec_frame = torch.clamp(lossy_dec_result["dpb"]["ref_frame"][0][0] * max_value, min=0,
                                                  max=max_value).type(torch.int32)
                torch.cuda.synchronize()
                lossy_decompress_time += (time.time() - lossy_start_time)

                torch.cuda.synchronize()
                enhance_start_time = time.time()
                if frame_idx == 0:
                    pass
                else:
                    frames = [ref_frames[i] for i in range(3)] + [lossy_dec_frame]  # 列表长度为 4，每个元素的形状为 (h, w)
                    frames = torch.stack(frames, dim=0).unsqueeze(0) / max_value  # 形状为 (4, h, w)
                    out = self.enhance_model(frames)
                    lossy_dec_frame = torch.clamp(out[0][0] * max_value, min=0, max=max_value).type(torch.int32)
                torch.cuda.synchronize()
                enhance_decompress_time += (time.time() - enhance_start_time)
                lossy_dec_frame = lossy_dec_frame.unsqueeze(dim=0).unsqueeze(dim=0) * 1.
                torch.cuda.synchronize()
                lossless_start_time = time.time()
                residues_dec = self.lossless_model.decompress(strings=lossless_stream, x_min=res_min, x_max=res_max,
                                                              cur=lossy_dec_frame, ref=ref_frame)
                torch.cuda.synchronize()
                lossless_decompress_time += (time.time() - lossless_start_time)
                if not torch.equal(lossy_dec_frame, lossy_rec_frame):
                    print("lossy_dec_frame != lossy_rec_frame at {}".format(frame_idx))
                if not torch.equal(residues_dec, residues):
                    print("residues_dec != residues at {}".format(frame_idx))
            ref_frames.popleft()
            ref_frames.append(ori_slices[frame_idx])
            dpb = {
                "ref_frame": x_padded,
                "ref_feature": None, "ref_mv_feature": None,
                "ref_y": None, "ref_mv_y": None,
            }
        lossless_bit_list = calculate_total_bitstream_size(lossless_bitstreams)
        lossy_bpp_list = [x / 3 / height / width for x in lossy_bits]
        lossless_bpp_list = [x / height / width for x in lossless_bit_list]

        lossy_bpp = sum(lossy_bpp_list) / num_slices
        lossless_bpp = sum(lossless_bpp_list) / num_slices
        bpp = lossless_bpp + lossy_bpp

        return {"bpp": bpp, "lossy_bpp": lossy_bpp, "lossless_bpp": lossless_bpp,
                "lossy_bpp_list": lossy_bpp_list, "lossless_bpp_list": lossless_bpp_list,
                "psnrs": psnrs, "psnrs_enh": psnrs_enh,
                "lossy_compress_time": lossy_compress_time / num_slices,
                "lossy_decompress_time": lossy_decompress_time / num_slices,
                "lossless_compress_time": lossless_compress_time / num_slices,
                "lossless_decompress_time": lossless_decompress_time / num_slices,
                "enhance_compress_time": enhance_compress_time / num_slices,
                "enhance_decompress_time": enhance_decompress_time / num_slices,
                }
