# encoding: utf-8
from utils import get_args
import sys
from trainer import LossLess_Trainer
from src.utils import ROOT as root
sys.path.append(root)


def lossless(args):
    args.log_root = "{}/train/lossless/{}".format(args.log_root, args.dataset)
    args.blocks = [1, 3, 1]
    args.trainset_compress = "" # 需要设置
    args.testset_compress = ""

    args.lr = 1e-4
    args.epochs = 100
    args.save_freq = 10
    args.update_freq = 50

    if args.dataset == 'axial' or args.dataset == 'coronal' or args.dataset == 'sagittal':
        args.batch_size = 8
        args.bit_depth = 8
        args.num_workers = 4
    elif args.dataset == 'mosmed' or args.dataset == 'chaosct':
        args.batch_size = 2
        args.bit_depth = 16
        args.num_workers = 4
        args.update_freq = 20
        if args.dataset == 'chaosct':
            args.epochs = 30
            args.save_freq = 1
    else:
        print('Invalid dataset')
    trainer = LossLess_Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_args().parse_args()
    lossless(args)
