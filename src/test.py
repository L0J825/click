import datetime
import os
import json
import logging
from tqdm import tqdm
import torch

import sys
sys.path.append('/root/7.3T/RJ/CLICK')

from src.utils import sort_files, fix_random_seed, setup_logger, get_args
from src.utils import ROOT as root
from src.models import Net_8b, Net_16b

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class NetTester:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        fix_random_seed(self.args.seed)

        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(self.args.log_root, f"{date}")
        self.args.save_directory = self.log_dir + '/pics'
        os.makedirs(self.args.save_directory, exist_ok=True)
        setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {self.args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        self.model = None
        self.resume()

        self.cuda = self.args.cuda

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(self.args)[k] for k in vars(self.args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    @torch.no_grad()
    def test(self, skip_dec=False):
        self.model.eval()
        self.compress_files = sort_files(folder_path=self.args.testset_compress, ext='.npy')
        test_bar = tqdm(self.compress_files, ncols=120, desc='Compress')
        record = {}
        total_bpp = 0
        for index, file in enumerate(test_bar):
            basename = os.path.splitext(os.path.split(file)[-1])[0]

            compress_result = self.model.encode_decode(npy_filepath=file, skip_dec=skip_dec)

            record[basename] = {"file_name": basename,
                                'bpp': compress_result["bpp"], 'lossy_bpp': compress_result["lossy_bpp"],
                                'lossless_bpp': compress_result["lossless_bpp"],
                                'lossy_bpp_list': compress_result["lossy_bpp_list"],
                                'lossless_bpp_list': compress_result["lossless_bpp_list"],
                                'psnr': compress_result["psnrs"], 'psnr_enh': compress_result["psnrs_enh"],
                                "lossy_compress_time": compress_result["lossy_compress_time"],
                                "lossy_decompress_time": compress_result["lossy_decompress_time"],
                                "lossless_compress_time": compress_result["lossless_compress_time"],
                                "lossless_decompress_time": compress_result["lossless_decompress_time"],
                                "enhance_compress_time": compress_result["enhance_compress_time"],
                                "enhance_decompress_time": compress_result["enhance_decompress_time"],
                                }
            total_bpp += compress_result["bpp"]
            avg_bpp = total_bpp / (index + 1)
            test_bar.set_postfix(name=basename, bpp=compress_result["bpp"], avg_bpp=avg_bpp)
            for k in record.keys():
                if isinstance(record[k]['psnr'], list):
                    record[k]['psnr-avg'] = sum(record[k]['psnr']) / len(record[k]['psnr'])
                if isinstance(record[k]['psnr_enh'], list):
                    record[k]['psnr_enh-avg'] = sum(record[k]['psnr_enh']) / len(record[k]['psnr_enh'])
            with open(os.path.join(self.log_dir, 'results.json'), mode='w') as f:
                json.dump(record, f, indent=4, sort_keys=True)

        total_num = len(self.compress_files)
        bpp_average = sum([record[k]['bpp'] for k in record.keys()]) / total_num
        lossy_bpp_average = sum([record[k]['lossy_bpp'] for k in record.keys()]) / total_num
        lossless_bpp_average = sum([record[k]['lossless_bpp'] for k in record.keys()]) / total_num
        psnr_average = sum([record[k]['psnr-avg'] for k in record.keys()]) / total_num
        psnr_enh_average = sum([record[k]['psnr_enh-avg'] for k in record.keys()]) / total_num
        lossy_compress_time_avg = sum([record[k]['lossy_compress_time'] for k in record.keys()]) / total_num
        lossy_decompress_time_avg = sum([record[k]['lossy_decompress_time'] for k in record.keys()]) / total_num
        lossless_compress_time_avg = sum([record[k]['lossless_compress_time'] for k in record.keys()]) / total_num
        lossless_decompress_time_avg = sum([record[k]['lossless_decompress_time'] for k in record.keys()]) / total_num
        enhance_compress_time_avg = sum([record[k]['enhance_compress_time'] for k in record.keys()]) / total_num
        enhance_decompress_time_avg = sum([record[k]['enhance_decompress_time'] for k in record.keys()]) / total_num

        record['average'] = {
            'bpp': bpp_average, 'lossy_bpp': lossy_bpp_average, 'lossless_bpp': lossless_bpp_average,
            'psnr': psnr_average, 'psnr_enh': psnr_enh_average,
            'lossy_compress_time': lossy_compress_time_avg,
            'lossy_decompress_time': lossy_decompress_time_avg,
            'lossless_compress_time': lossless_compress_time_avg,
            'lossless_decompress_time': lossless_decompress_time_avg,
            'enhance_compress_time': enhance_compress_time_avg,
            'enhance_decompress_time': enhance_decompress_time_avg,
        }
        with open(os.path.join(self.log_dir, 'results.json'), mode='w') as f:
            json.dump(record, f, indent=4, sort_keys=True)

        self.logger.info(f'[*] Results saved to {os.path.join(self.log_dir, "results.json")}')
        self.logger.info('=' * 80)
        self.logger.info('[*] Performance Summary:')
        self.logger.info(f'    Total BPP: {bpp_average:.4f}')
        self.logger.info(f'    Lossy BPP: {lossy_bpp_average:.4f}')
        self.logger.info(f'    Lossless BPP: {lossless_bpp_average:.4f}')
        self.logger.info(f'    PSNR Average: {psnr_average:.4f}')
        self.logger.info(f'    PSNR Enhanced Average: {psnr_enh_average:.4f}')
        self.logger.info(f'    Lossy Compress Time: {lossy_compress_time_avg:.4f}s')
        self.logger.info(f'    Lossy Decompress Time: {lossy_decompress_time_avg:.4f}s')
        self.logger.info(f'    Lossless Compress Time: {lossless_compress_time_avg:.4f}s')
        self.logger.info(f'    Lossless Decompress Time: {lossless_decompress_time_avg:.4f}s')
        self.logger.info(f'    Enhance Compress Time: {enhance_compress_time_avg:.4f}s')
        self.logger.info(f'    Enhance Decompress Time: {enhance_decompress_time_avg:.4f}s')
        self.logger.info(f'    Total Files: {total_num}')
        self.logger.info('=' * 80)

    def resume(self):
        if self.args.bit_depth == 16:
            self.model = Net_16b(self.args)
        else:
            self.model = Net_8b(self.args)
        self.model.resume()
        self.logger.info(f'[*] Model: {type(self.model)}')
        self.logger.info(f'[*] Total Parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if self.args.cuda:
            self.model = self.model.cuda()


def lossless_test(args, skip_dec=True):
    args.desc = "Lossless enhanced compression model test"
    args.blocks = [1, 3, 1]

    args.enhancer_model_path = '/root/7.3T/RJ/CLICK/weights/axial_deb.pth'
    args.lossless_model_path = '/root/7.3T/RJ/CLICK/weights/axial_ll.pth'
    args.log_root = os.path.join(args.log_root, 'test', 'lossless', args.dataset)
    args.testset_compress = os.path.join('/root/7.3T/RJ/mcdc', 'data', args.dataset, 'valid')

    if args.dataset == 'axial' or args.dataset == 'coronal' or args.dataset == 'sagittal':
        args.bit_depth = 8
        args.p_frame_q_indexes = 3
        args.i_frame_q_indexes = 3
    else:
        args.bit_depth = 16
        args.p_frame_q_indexes = 3
        args.i_frame_q_indexes = 3

    T = NetTester(args)
    T.test(skip_dec=skip_dec)


if __name__ == '__main__':
    args = get_args().parse_args()
    lossless_test(args, skip_dec=not args.decode)
