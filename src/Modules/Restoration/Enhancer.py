import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.Modules.Restoration.QualityEnhance import RFDM
from src.Modules.Restoration.DeformConv import ModulatedDeformConv


class STDF(nn.Module):
    def __init__(self, in_nc=4, out_nc=1, nf=32, nb=3, base_ks=3, deform_ks=3):
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        out_lst = [self.in_conv(inputs)]
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        out = self.tr_conv(out_lst[-1])
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )

        return fused_feat


class RFDMNet(nn.Module):
    def __init__(self, in_dim=4, hidden_dim1=64, hidden_dim2=96, out_dim=1, nf=32, base_ks=3, deform_ks=3,
                 num_layers=3, use_checkpoint=False):
        super(RFDMNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.nb = 3
        self.in_dim = in_dim
        self.size_dk = deform_ks * deform_ks
        self.deform_ks = deform_ks

        self.ffnet = STDF(in_nc=in_dim, out_nc=hidden_dim1, nf=nf, nb=num_layers, deform_ks=base_ks)
        self.rfdm = RFDM(num_layers=num_layers, in_dim=hidden_dim1, out_dim=out_dim,
                                              hidden_dim=hidden_dim2)

    def forward(self, x):
        if self.use_checkpoint:
            fused_feat = checkpoint(self.ffnet, x)
            res = checkpoint(self.rfdm, fused_feat)
        else:
            fused_feat = self.ffnet(x)
            res = self.rfdm(fused_feat)
        enhanced_img = res + x[:, -1:, :, :]
        return enhanced_img
