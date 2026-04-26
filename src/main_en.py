from utils import get_args
from trainer import Restoration8BTrainer, Restoration16BTrainer


def enhance_8bit(args):
    args.batch_size = 16
    args.num_workers = 4
    args.bit_depth = 8
    args.epochs = 100
    args.update_freq = 50
    trainer = Restoration8BTrainer(args)
    trainer.train()


def enhance_16bit(args):
    args.batch_size = 4
    args.num_workers = 4
    args.bit_depth = 16
    args.epochs = 150
    args.update_freq = 100
    args.use_checkpoint = True

    trainer = Restoration16BTrainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_args().parse_args()

    args.log_root = "{}/train/enhance/{}".format(args.log_root, args.dataset)
    args.trainset_enhance = ""
    args.testset_enhance = ""


    if args.dataset == 'axial' or args.dataset == 'coronal' or args.dataset == 'sagittal':
        enhance_8bit(args)
    elif args.dataset == 'mosmed' or args.dataset == 'chaosct':
        enhance_16bit(args)
    else:
        print('Invalid dataset')
