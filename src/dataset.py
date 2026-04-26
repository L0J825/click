import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LossySet(Dataset):
    def __init__(self, npy_dir, bit_depth=8, state='train'):
        assert bit_depth in [8, 16], f"Invalid bit depth. Supported values are 8 and 16, given {bit_depth}."
        self.file_folder = npy_dir
        self.bit_depth = bit_depth
        self.state = state
        self.npy_files = glob(os.path.join(npy_dir, "*.npy"))
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((128, 128)),  
        ])

    def __getitem__(self, item):
        filename = self.npy_files[item]
        npy = np.load(filename)
        tensor = torch.from_numpy(npy.astype(np.float32))
        if self.state == 'train':
            tensor = self.transform(tensor.unsqueeze(0)).squeeze(0)

        if self.bit_depth == 8:
            return tensor.to(torch.uint8), filename
        elif self.bit_depth == 16:
            return tensor, filename
        else:
            raise ValueError("Invalid bit depth. Supported values are 8 and 16.")

    def __len__(self):
        return len(self.npy_files)
F

class EnhanceSet(Dataset):
    def __init__(self, npy_dir, bit_depth=8, state='train'):
        assert bit_depth in [8, 16], f"Invalid bit depth. Supported values are 8 and 16, given {bit_depth}."
        self.file_folder = npy_dir
        self.bit_depth = bit_depth
        self.state = state
        self.npy_files = glob(os.path.join(npy_dir, "*.npy"))
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def __getitem__(self, item):
        filename = self.npy_files[item]
        npy = np.load(filename)
        tensor = torch.from_numpy(npy.astype(np.float32))
        if self.state == 'train':
            tensor = self.transform(tensor.unsqueeze(0)).squeeze(0)

        return tensor, filename

    def __len__(self):
        return len(self.npy_files)


def get_lossy_dataset(args):
    train_set = LossySet(args.trainset_compress, bit_depth=args.bit_depth, state='train')
    valid_set = LossySet(args.testset_compress, bit_depth=args.bit_depth, state='test')
    return train_set, valid_set


def get_enhance_dataset(args):
    train_set = EnhanceSet(args.trainset_enhance, bit_depth=args.bit_depth, state='train')
    valid_set = EnhanceSet(args.testset_enhance, bit_depth=args.bit_depth, state='test')
    return train_set, valid_set
