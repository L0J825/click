import os
import datetime
import json
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import sys

root = '/root/7.3T/RJ/mcdc'
sys.path.append(root)

from src.utils import setup_logger, fix_random_seed, calculate_psnr
from src.dataset import get_lossy_dataset, get_enhance_dataset
from src.loss import CharbonnierLoss
from src.Modules.Restoration.Enhancer import RFDMNet
from src.Modules.LCEN.LCEN import LCEN_v6 as LCEN

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Restoration_Trainer(object):
    def __init__(self, args):
        args.cuda = torch.cuda.is_available()
        self.args = args
        fix_random_seed(args.seed)
        self.grad_clip = 1.0

        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(self.args.log_root, f"{date}")
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')
        setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        self.batch_size = args.batch_size

        training_set, valid_set = get_enhance_dataset(self.args)
        if args.bit_depth == 16:
            valid_set.npy_files = valid_set.npy_files[:100]
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )

        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')
        del training_set, valid_set

        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        self.graph = None
        self.train_loss = None

        self.model_init()

        self.configure_optimizers()

        self.cuda = self.args.cuda

        self.device = next(self.graph.parameters()).device

        self.lowest_val_loss = 0

        if args.resume:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.enhancer_model_path}...")
        checkpoint = torch.load(self.args.enhancer_model_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        self.logger.info(f"[*] global_step {self.global_step}...")
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lowest_val_loss = checkpoint["lowest_val_loss"]
        del checkpoint

    def save_checkpoint(self, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "state_dict": self.graph.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "scheduler": self.scheduler.state_dict(),
            "lowest_val_loss": self.lowest_val_loss,
        }
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))
            print("******** SAVE BEST MODEL ********")
        else:
            torch.save(state, os.path.join(self.checkpoints_dir, name + '.pth'))

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.args.lr, betas=self.args.betas)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.update_freq, gamma=0.75)

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def model_init(self):
        self.graph = RFDMNet(in_dim=self.args.in_dim, hidden_dim1=self.args.hidden_dim1,
                                 hidden_dim2=self.args.hidden_dim2, num_layers=self.args.num_layers,
                                 use_checkpoint=self.args.use_checkpoint)
        self.train_loss = CharbonnierLoss()
        self.logger.info(f'[*] Model = {self.graph.__class__.__name__}')
        self.logger.info(f'[*] Train Loss = {self.train_loss}')
        if self.args.cuda:
            self.graph = self.graph.cuda()
        self.logger.info(f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')


class Restoration8BTrainer(Restoration_Trainer):
    def __init__(self, args):
        super(Restoration8BTrainer, self).__init__(args)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.graph.train()
            train_bar = tqdm(self.training_set_loader, desc=f'Train{epoch + 1}|{self.num_epochs}', ncols=120)
            running_loss = 0.0
            for _, (img, name) in enumerate(train_bar):
                self.graph.train()
                img = img.to(self.device).requires_grad_(True)
                img = img / 255.0
                out = self.graph(torch.cat([img[:, 2:, :, :], img[:, 0, :, :].unsqueeze(1)], dim=1))
                loss = self.train_loss(out, img[:, 1, :, :].unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_gradient(self.optimizer, self.grad_clip)
                self.optimizer.step()

                running_loss += loss.item()
                self.global_step += 1
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                train_bar.set_postfix(loss=loss.item())
            self.scheduler.step()

            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            avg_loss = running_loss / len(self.training_set_loader) / self.batch_size
            self.writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
            self.logger.info(f'[*] Epoch {epoch + 1}/{self.num_epochs} Training Loss {avg_loss}')
            self.save_checkpoint('latest_checkpoint', False)
            if (epoch + 1) % self.args.valid_freq == 0 or epoch < 5:
                self.validate()
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(str(self.global_epoch), False)

    def validate(self):
        with torch.no_grad():
            self.graph.eval()
            running_val_psnr = 0.0
            valid_bar = tqdm(self.valid_set_loader, desc=f'Valid{self.global_epoch + 1}|{self.num_epochs}', ncols=140)
            for _, (img, name) in enumerate(valid_bar):
                img = img.to(self.device)
                x = torch.cat([img[:, 2:, :, :], img[:, 0, :, :].unsqueeze(1)], dim=1) / 255.0
                out = self.graph(x)
                out = torch.clamp(out * 255, 0, 255).type(torch.uint8)

                psnr = calculate_psnr(out, img[:, 1, :, :].unsqueeze(1), max_value=255)
                running_val_psnr += psnr

            avg_val_psnr = running_val_psnr / len(self.valid_set_loader)
            valid_bar.set_postfix(psnr=avg_val_psnr)
            self.writer.add_scalar('Validation/PSNR', avg_val_psnr, self.global_epoch)
            self.logger.info("[*] Validation PSNR: {:.4f}".format(avg_val_psnr))

        if avg_val_psnr > self.lowest_val_loss:
            self.lowest_val_loss = avg_val_psnr
            self.logger.info("[*] New Highest Validation PSNR: {:.4f}".format(avg_val_psnr))
            self.save_checkpoint(str(self.global_epoch), True)
            self.logger.info("VALID [{}|{}] PSNR[{:.2f}]".format(self.global_epoch + 1, self.num_epochs, avg_val_psnr))
        self.graph.train()


class Restoration16BTrainer(Restoration_Trainer):
    def __init__(self, args):
        super(Restoration16BTrainer, self).__init__(args)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.graph.train()
            train_bar = tqdm(self.training_set_loader, desc=f'Train{epoch + 1}|{self.num_epochs}', ncols=120)
            running_loss = 0.0
            for _, (img, name) in enumerate(train_bar):
                self.graph.train()
                img = img.to(self.device).requires_grad_(True)
                max_value = torch.max(img).item()
                img = img / max_value
                out = self.graph(torch.cat([img[:, 2:, :, :], img[:, 0, :, :].unsqueeze(1)], dim=1))
                loss = self.train_loss(out, img[:, 1, :, :].unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_gradient(self.optimizer, self.grad_clip)
                self.optimizer.step()

                running_loss += loss.item()
                self.global_step += 1
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                train_bar.set_postfix(loss=loss.item())
            self.scheduler.step()

            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            avg_loss = running_loss / len(self.training_set_loader) / self.batch_size
            self.writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
            self.logger.info(f'[*] Epoch {epoch + 1}/{self.num_epochs} Training Loss {avg_loss}')
            self.save_checkpoint('latest_checkpoint', False)
            if (epoch + 1) % self.args.valid_freq == 0 or epoch < 5:
                self.validate()
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(str(self.global_epoch), False)

    def validate(self):
        with torch.no_grad():
            self.graph.eval()
            running_val_psnr = 0.0
            valid_bar = tqdm(self.valid_set_loader, desc=f'Valid{self.global_epoch + 1}|{self.num_epochs}', ncols=140)
            for _, (img, name) in enumerate(valid_bar):
                img = img.to(self.device)
                max_value = torch.max(img).item()
                x = torch.cat([img[:, 2:, :, :], img[:, 0, :, :].unsqueeze(1)], dim=1) / max_value
                out = self.graph(x)
                out = torch.clamp(out * max_value, 0, max_value).type(torch.int32)

                psnr = calculate_psnr(out, img[:, 1, :, :].unsqueeze(1), max_value=max_value)
                running_val_psnr += psnr

            avg_val_psnr = running_val_psnr / len(self.valid_set_loader)
            valid_bar.set_postfix(psnr=avg_val_psnr)
            self.writer.add_scalar('Validation/PSNR', avg_val_psnr, self.global_epoch)
            self.logger.info("[*] Validation PSNR: {:.4f}".format(avg_val_psnr))

        if avg_val_psnr > self.lowest_val_loss:
            self.lowest_val_loss = avg_val_psnr
            self.logger.info("[*] New Highest Validation PSNR: {:.4f}".format(avg_val_psnr))
            self.save_checkpoint(str(self.global_epoch), True)
            self.logger.info("VALID [{}|{}] PSNR[{:.2f}]".format(self.global_epoch + 1, self.num_epochs, avg_val_psnr))
        self.graph.train()


class LossLess_Trainer(object):
    def __init__(self, args):
        args.cuda = torch.cuda.is_available()
        self.args = args
        fix_random_seed(args.seed)

        self.grad_clip = 1.0

        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(self.args.log_root, f"{date}")
        self.args.save_directory = self.log_dir + '/pics'
        os.makedirs(self.args.save_directory, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        self.batch_size = args.batch_size

        training_set, valid_set = get_lossy_dataset(self.args)
        if args.bit_depth == 16:
            valid_set.npy_files = valid_set.npy_files[:100]
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=False,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )

        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')
        del training_set, valid_set

        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        self.graph = None
        self.model_init()

        self.configure_optimizers()

        self.cuda = self.args.cuda

        self.device = next(self.graph.parameters()).device

        self.lowest_val_loss = float("inf")

        if args.resume:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.graph.train()
            train_bar = tqdm(self.training_set_loader, desc=f'Train{epoch + 1}|{self.num_epochs}', ncols=120)
            running_loss = 0.0
            for _, (img, name) in enumerate(train_bar):
                self.graph.train()
                img = img.to(self.device)
                lossless_slice = img[:, 0, :, :].unsqueeze(1)
                lossy_slice = img[:, 1, :, :].unsqueeze(1)
                ref_slice = img[:, 2, :, :].unsqueeze(1)
                residues = lossless_slice.type(torch.float32) - lossy_slice.type(torch.float32)
                residues_min = int(residues.min())
                residues_max = int(residues.max())
                loss = self.graph(residues, lossy_slice, ref_slice, x_min=residues_min, x_max=residues_max)
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_gradient(self.optimizer, self.grad_clip)
                self.optimizer.step()
                running_loss += loss.item()
                self.global_step += 1
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                train_bar.set_postfix(loss=loss.item())

                del img, lossless_slice, lossy_slice, ref_slice, residues, loss

            self.scheduler.step()

            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            avg_loss = running_loss / len(self.training_set_loader)
            self.writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
            self.logger.info(f'[*] Epoch {epoch + 1}/{self.num_epochs} Training Loss {avg_loss}')
            self.save_checkpoint(avg_loss, 'latest_checkpoint', False)
            if (epoch + 1) % self.args.valid_freq == 0 or epoch < 1:
                self.validate()
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(avg_loss, str(self.global_epoch), False)

    def validate(self):
        self.graph.eval()
        running_val_bpp = 0.0
        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader, desc=f'Valid{self.global_epoch + 1}|{self.num_epochs}', ncols=140)
            for _, (img, name) in enumerate(valid_bar):
                img = img.to(self.device)
                lossless_slice = img[:, 0, :, :].unsqueeze(1)
                lossy_slice = img[:, 1, :, :].unsqueeze(1)
                ref_slice = img[:, 2, :, :].unsqueeze(1)
                residues = lossless_slice.type(torch.float32) - lossy_slice.type(torch.float32)
                residues_min = int(residues.min())
                residues_max = int(residues.max())
                strings = self.graph.compress(residues, lossy_slice, ref_slice, x_min=residues_min, x_max=residues_max)
                bpp = sum(len(s) for s in strings) * 8 / (
                        lossless_slice.shape[0] * lossless_slice.shape[1] * lossless_slice.shape[2] *
                        lossless_slice.shape[3])
                running_val_bpp += bpp
                valid_bar.set_postfix(bpp=bpp)

                del img, lossless_slice, lossy_slice, ref_slice, residues, strings

            avg_val_bpp = running_val_bpp / len(self.valid_set_loader)

            self.writer.add_scalar('Validation/BPP', avg_val_bpp, self.global_epoch)
            self.logger.info("[*] Validation BPP: {:.4f}".format(avg_val_bpp))

        if avg_val_bpp < self.lowest_val_loss:
            self.lowest_val_loss = avg_val_bpp
            self.logger.info("[*] New Lowest Validation Loss: {:.4f}".format(avg_val_bpp))
            self.save_checkpoint(avg_val_bpp, str(self.global_epoch), True)
            self.logger.info("VALID [{}|{}] LOSS[{:.4f}]".format(self.global_epoch + 1, self.num_epochs, avg_val_bpp))
        self.graph.train()

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.lossless_model_path}...")
        checkpoint = torch.load(self.args.lossless_model_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        self.logger.info(f"[*] global_step {self.global_step}...")
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lowest_val_loss = checkpoint["lowest_val_loss"]
        del checkpoint

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "scheduler": self.scheduler.state_dict(),
            "lowest_val_loss": self.lowest_val_loss,
        }
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))
            print("******** SAVE BEST MODEL ********")
        else:
            torch.save(state, os.path.join(self.checkpoints_dir, name + '.pth'))
            print("save checkpoint model...")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.args.lr, betas=self.args.betas)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.update_freq, gamma=0.75)

        return None

    def clip_gradient(self, optimizer, grad_clip):

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def model_init(self):
        self.graph = LCEN(bit_depth=self.args.bit_depth, blocks=self.args.blocks)
        self.logger.info(f'[*] Model = {self.graph.__class__.__name__}')
        if self.args.cuda:
            self.graph = self.graph.cuda()
        self.logger.info(f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')


        