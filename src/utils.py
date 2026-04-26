# encoding: utf-8
import os
import re
import numpy as np
import random
import logging
import datetime
import argparse
import torch

ROOT = '/home/CLICK'


def get_args():
    parser = argparse.ArgumentParser(description='CLICK Argument Parser')
    parser.add_argument('--desc', type=str, default='no special description')

    parser.add_argument('--dataset', '-d', type=str, default='axial',
                        choices=["axial", "coronal", "sagittal", "mosmed", "chaosct"],
                        help='Dataset type')
    parser.add_argument("--log_root", type=str, default="{}/logs".format(ROOT),
                        help="Log file root directory")
    parser.add_argument("--temp_dir", type=str, default="{}/logs/Temp".format(ROOT),
                        help="Temporary directory")
    parser.add_argument("--seed", type=int, default=1234, )
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size, default is 8")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help="Whether to use GPU")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of data loading threads")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device type, default is 'cuda' or 'cpu'")
    parser.add_argument('--height', type=int, default=256,
                        help="Input image height, default is 256")
    parser.add_argument('--width', type=int, default=256,
                        help="Input image width, default is 256")
    parser.add_argument('--state', type=str, default='test', choices=['train', 'test'], )
    parser.add_argument("--use_checkpoint", type=bool, default=False,
                        help="Whether to use torch checkpoint function, trading time for space")

    parser.add_argument("--trainset_compress", type=str,
                        help="Training set compressed image path")
    parser.add_argument("--testset_compress", type=str,
                        help="Test set compressed image path")
    parser.add_argument("--trainset_enhance", type=str,
                        help="Enhancement network training dataset path")
    parser.add_argument("--testset_enhance", type=str,
                        help="Enhancement network test dataset path")

    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Node rank for distributed training')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs, default is 200")
    parser.add_argument("--update_freq", type=int, default=20,
                        help="Learning rate update frequency")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate, default is 1e-4")
    parser.add_argument('--warmup_iter', type=int, default=-1,
                        help="Warmup iterations, default is -1")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum, default is 0.9")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999),
                        help="Adam optimizer betas, default is (0.9, 0.9999)")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Adam optimizer epsilon, default is 1e-8")
    parser.add_argument("--valid_freq", type=int, default=5,
                        help="Validation frequency in epochs")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Model save frequency in epochs")

    parser.add_argument("--ec_thread", type=bool, nargs='?', const=True, default=False,
                        help="Whether to enable enhanced encoding thread, default is False")
    parser.add_argument("--stream_part_i", type=int, default=1,
                        help="Stream part I, default is 1")
    parser.add_argument("--stream_part_p", type=int, default=1,
                        help="Stream part P, default is 1")
    parser.add_argument('--i_frame_model_path', type=str,
                        default='/root/7.3T/RJ/mcdc/weights/cvpr2023_image_psnr.pth.tar',
                        help="I-frame model path")
    parser.add_argument('--p_frame_model_path', type=str,
                        default='/root/7.3T/RJ/mcdc/weights/cvpr2023_video_psnr.pth.tar',
                        help="P-frame model path")
    parser.add_argument('--q_in_ckpt', type=bool, default=True, help="Whether to use q in checkpoint")
    parser.add_argument('--i_frame_q_indexes', type=int, default=3, help="I-frame quantization index")
    parser.add_argument('--p_frame_q_indexes', type=int, default=3, help="P-frame quantization index")
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--write_stream', type=str, default=True, help="Whether to generate bitstream")

    parser.add_argument('--lossless_model_path', '-lm', type=str,
                        help='Lossless model checkpoint path')
    parser.add_argument('--bit_depth', type=int, default=8,
                        help='Input data bit depth')
    parser.add_argument('--blocks', type=list, default=[1, 3, 1])

    parser.add_argument('--enhancer_model_path', type=str,
                        help='Enhancer model checkpoint path')
    parser.add_argument('--in_dim', type=int, default=4,
                        help='Number of previous reference frames + 1 (current frame)')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='Output channels')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Enhancement network layers')
    parser.add_argument('--hidden_dim1', type=int, default=64,
                        help='Alignment network channels')
    parser.add_argument('--hidden_dim2', type=int, default=96,
                        help='Quality enhancement network channels')
    parser.add_argument('--nf', type=int, default=32,
                        help='UNet channels')
    parser.add_argument('--base_ks', type=int, default=3,
                        help='Base kernel size')
    parser.add_argument('--deform_ks', type=int, default=3,
                        help='Deformable kernel size')

    parser.add_argument('--decode', action='store_true',
                        help='Enable decoding step (skipped by default, only test compression)')
    return parser


def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, ROOT, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(ROOT, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def fix_random_seed(seed_value=2021):
    os.environ['PYTHONPATHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return 0


def calculate_total_bitstream_size(A, nlb=4):
    total_sizes = []
    for i in range(0, len(A)):
        total_size = 0
        if len(A[i]) != 0:
            for k in range(nlb):
                total_size += len(A[i][k]) * 8
        total_sizes.append(total_size)
    return total_sizes


def calculate_psnr(ori_frame, recon_frame, max_value=1.0):
    """
    Calculate the Mean Squared Error (MSE) between two image frames represented as PyTorch tensors.

    Args:
    ori_frame (torch.Tensor): Original image frame.
    recon_frame (torch.Tensor): Reconstructed image frame.

    Returns:
    float: MSE value.
    """
    assert ori_frame.shape == recon_frame.shape, "Frames must have the same dimensions, {} and {}".format(
        ori_frame.shape, recon_frame.shape)

    ori_frame = ori_frame.to(torch.float32)
    recon_frame = recon_frame.to(torch.float32)
    squared_diff = torch.pow(ori_frame - recon_frame, 2)

    mse = torch.mean(squared_diff)
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()


def sort_files(folder_path, ext='.npy'):
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith(ext)]
    jpg_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    sorted_files = []
    for i, file in enumerate(jpg_files):
        if i % 10 == 9:
            sorted_files.append(file)
        else:
            sorted_files.insert(i // 10 * 10 + 9, file)
    files = [os.path.join(folder_path, file) for file in sorted_files]
    return files
