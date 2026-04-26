import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.Modules.LCEN.ETFM import EnhancedTemporalFeatureModule_v6
from src.Modules.LCEN.EntropyModel import EntropyModel


class LCEN_v6(nn.Module):
    def __init__(self, bit_depth: int, channels_X: int = 1, channels_F: int = 64, K: int = 10, channel_M: int = 64,
                 blocks=[1, 1, 1]):
        super().__init__()

        self.bit_depth = bit_depth
        self.entropy_models = EntropyModel(K=K, channels_ctx=channel_M, channels_data=channels_X)
        self.etfm = EnhancedTemporalFeatureModule_v6(in_dim=channels_X, hidden_dim=channels_F, out_dim=channel_M,
                                                     blocks=blocks)

    def forward(self, residues: torch.Tensor, cur: torch.Tensor, ref: torch.Tensor = None,
                x_min: int = -255, x_max: int = 255, use_checkpoint: bool = False):
        batch, _, height, width = residues.shape
        residues = residues * 1.
        cur = cur / (2 ** self.bit_depth - 1) * 1.

        if ref is not None:
            ref = ref / (2 ** self.bit_depth - 1) * 1.
        else:
            ref = cur.clone().detach()

        if use_checkpoint:
            prior_feats = checkpoint(self.etfm, cur, ref)
            loss = checkpoint(self.entropy_models.forward, residues, ctx=prior_feats, x_min=x_min, x_max=x_max)
        else:
            prior_feats = self.etfm(cur, ref)
            loss = self.entropy_models.forward(residues, ctx=prior_feats, x_min=x_min, x_max=x_max)
        return loss

    @torch.no_grad()
    def compress(self, residues: torch.Tensor, cur: torch.Tensor, ref: torch.Tensor = None,
                 x_min: int = -255, x_max: int = 255):
        batch, _, height, width = residues.shape
        residues = residues * 1.
        cur = cur / (2 ** self.bit_depth - 1) * 1.

        if ref is not None:
            ref = ref / (2 ** self.bit_depth - 1) * 1.
        else:
            ref = cur.clone().detach()

        prior_feats = self.etfm(cur, ref)
        string = self.entropy_models.compress(residues, ctx=prior_feats, x_min=x_min, x_max=x_max)
        return string

    @torch.no_grad()
    def decompress(self, strings: torch.Tensor, cur: torch.Tensor, ref: torch.Tensor = None,
                   x_min: int = -255, x_max: int = 255):
        batch, _, height, width = cur.shape
        cur = cur / (2 ** self.bit_depth - 1) * 1.

        if ref is not None:
            ref = ref / (2 ** self.bit_depth - 1) * 1.
        else:
            ref = cur.clone().detach()

        prior_feats = self.etfm(cur, ref)
        residues = self.entropy_models.decompress(strings, ctx=prior_feats, x_min=x_min, x_max=x_max)
        return residues

if __name__ == '__main__':
    import torch

    torch.cuda.set_device(1)

    model = LCEN_Lite(bit_depth=8, blocks=[1, 1, 1]).cuda()

    from thop import profile

    model.eval()
    cur = torch.randn(1, 1, 256, 256).cuda()
    ref = torch.randn(1, 1, 256, 256).cuda()

    flops, params = profile(model, inputs=(cur, ref))

    flops_g = flops / 10 ** 9
    params_m = params / 10 ** 6

    print(f"Model FLOPs: {flops_g:.2f} G")
    print(f"Model Parameters: {params_m:.2f} M")
    print(f"Raw FLOPs: {flops}")
    print(f"Raw Parameters: {params}")
