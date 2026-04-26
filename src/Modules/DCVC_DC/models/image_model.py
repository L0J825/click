# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np


from .common_model import CompressionModel
from .layers import conv3x3, DepthConvBlock2, ResidualBlockUpsample, ResidualBlockWithStride
from .video_net import UNet2
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize, \
    get_state_dict


class IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(3, 128, stride=2, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride(128, 192, stride=2, inplace=inplace),
            DepthConvBlock2(192, 192, inplace=inplace),
            ResidualBlockWithStride(192, N, stride=2, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
            ResidualBlockUpsample(N, 192, 2, inplace=inplace),
            DepthConvBlock2(192, 192, inplace=inplace),
            ResidualBlockUpsample(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock2(128, 128, inplace=inplace),
            ResidualBlockUpsample(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)


class IntraNoAR(CompressionModel):
    def __init__(self, N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=N,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock2(N, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock2(N, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N * 3, N * 3, inplace=inplace),
            DepthConvBlock2(N * 3, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )

        self.dec = IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet2(16, 16, inplace=inplace),
            conv3x3(16, 3),
        )

        self.q_basic_enc = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_enc_fine = None
        self.q_basic_dec = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_dec_fine = None

    def get_q_for_inference(self, q_in_ckpt, q_index):
        q_scale_enc = self.q_scale_enc[:, 0, 0, 0] if q_in_ckpt else self.q_scale_enc_fine
        curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
        q_scale_dec = self.q_scale_dec[:, 0, 0, 0] if q_in_ckpt else self.q_scale_dec_fine
        curr_q_dec = self.get_curr_q(q_scale_dec, self.q_basic_dec, q_index=q_index)
        return curr_q_enc, curr_q_dec

    def forward(self, x, q_in_ckpt=False, q_index=None):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scale_enc = ckpt["q_scale_enc"].reshape(-1)
        q_scale_dec = ckpt["q_scale_dec"].reshape(-1)
        return q_scale_enc, q_scale_dec

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            q_scale_enc_fine = np.linspace(np.log(self.q_scale_enc[0, 0, 0, 0]),
                                           np.log(self.q_scale_enc[3, 0, 0, 0]), 64)
            self.q_scale_enc_fine = np.exp(q_scale_enc_fine)
            q_scale_dec_fine = np.linspace(np.log(self.q_scale_dec[0, 0, 0, 0]),
                                           np.log(self.q_scale_dec[3, 0, 0, 0]), 64)
            self.q_scale_dec_fine = np.exp(q_scale_dec_fine)

    def encode_decode(self, x, q_in_ckpt, q_index,
                      output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            encoded = self.forward(x, q_in_ckpt, q_index)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
            }
            return result

        assert pic_height is not None
        assert pic_width is not None
        compressed = self.compress(x, q_in_ckpt, q_index)
        bit_stream = compressed['bit_stream']
        bit_stream_bk = bit_stream
        encode_i(pic_height, pic_width, q_in_ckpt, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_in_ckpt, q_index, bit_stream = decode_i(output_path)

        decompressed = self.decompress(bit_stream, height, width, q_in_ckpt, q_index)
        x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result, bit_stream_bk

    def compress(self, x, q_in_ckpt, q_index):
        """
        压缩输入的图像数据x。

        :param x: 输入的图像数据。
        :param q_in_ckpt: 量化器的检查点。
        :param q_index: 量化器的索引。
        :return: 包含压缩后位流和重构图像的结果字典。
        """
        # 获取推理所需的量化参数
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        # 使用编码器对输入图像进行编码
        y = self.enc(x, curr_q_enc)
        # 对编码后的数据进行padding，以适应后续的超编码
        y_pad, slice_shape = self.pad_for_y(y)
        # 超编码处理
        z = self.hyper_enc(y_pad)
        # 对超编码结果进行量化
        z_hat = torch.round(z)

        # 解码超编码参数
        params = self.hyper_dec(z_hat)
        # 将超编码参数与先验信息融合
        params = self.y_prior_fusion(params)
        # 根据切片形状重构参数
        params = self.slice_to_y(params, slice_shape)
        # 使用四部分先验对编码后的数据进行压缩
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        # 初始化熵编码器并编码量化后的数据
        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        # 刷新熵编码器，完成编码
        self.entropy_coder.flush()

        # 通过解码器重构图像，并确保结果在0到1之间
        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        # 获取编码后的位流
        bit_stream = self.entropy_coder.get_encoded_stream()

        # 返回压缩后的位流和重构图像
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, height, width, q_in_ckpt, q_index):
        """
        解压缩函数，根据给定的位流、图像高度、宽度以及量化参数，恢复原始图像。

        参数:
        bit_stream: 输入的位流，包含编码后的信息。
        height: 原始图像的高度。
        width: 原始图像的宽度。
        q_in_ckpt: 量化参数在检查点中的索引。
        q_index: 当前量化参数的索引。

        返回:
        一个字典，包含解压缩后的图像x_hat。
        """
        # 获取模型参数的数据类型和设备信息
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        # 根据输入的量化参数索引，获取用于解码的量化参数
        _, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        # 设置熵编码器的输入位流
        self.entropy_coder.set_stream(bit_stream)
        # 计算下采样后的z尺寸
        z_size = get_downsampled_shape(height, width, 64)
        # 计算下采样后的y尺寸
        y_height, y_width = get_downsampled_shape(height, width, 16)
        # 计算切片形状，用于后续的切片操作
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        # 从位流中解码z的估计值
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)

        # 使用解码后的z_hat估计参数
        params = self.hyper_dec(z_hat)
        # 将参数与先验融合
        params = self.y_prior_fusion(params)
        # 将参数转换为y的切片形状
        params = self.slice_to_y(params, slice_shape)
        # 使用四个部分的先验恢复y的估计值
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)

        # 细化y_hat的估计，得到最终的x_hat，并确保其值在0到1之间
        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        # 返回解压缩后的图像
        return {"x_hat": x_hat}
