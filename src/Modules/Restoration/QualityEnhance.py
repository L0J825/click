import torch
import torch.nn as nn
from einops import rearrange
import math


class MLP(nn.Module):
    def __init__(self, channel, bias=True):
        super().__init__()
        self.w_1 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)

    def forward(self, x):
        return self.w_2(torch.tanh(self.w_1(x)))


class MHRB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mlp_z = MLP(channels)
        self.mlp_f = MLP(channels)

    def forward(self, inputs):
        Z = self.mlp_z(inputs).tanh()
        F = self.mlp_f(inputs).sigmoid()

        Z1, Z2 = Z.split(self.channels // 2, 1)
        Z2 = torch.flip(Z2, [2])
        Z = torch.cat([Z1, Z2], dim=1)

        F1, F2 = F.split(self.channels // 2, 1)
        F2 = torch.flip(F2, [2])
        F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
            h = (1 - f) * z if h is None else f * h + (1 - f) * z
            h_time.append(h)

        y = torch.cat(h_time, dim=2)
        y1, y2 = y.split(self.channels // 2, 1)
        y2 = torch.flip(y2, [2])
        y = torch.cat([y1, y2], dim=1)

        return y


class FRFN(nn.Module):
    def __init__(self, dim=64, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_init = x
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)

        return x + x_init


class RFDM(nn.Module):
    def __init__(self, num_layers=3, in_dim=64, out_dim=1, hidden_dim=128, patch_size=16):
        super(RFDM, self).__init__()
        self.body = nn.ModuleList()
        for i in range(num_layers):
            self.body.append(nn.Sequential(
                MHRB(channels=in_dim),
                FRFN(dim=in_dim, hidden_dim=hidden_dim)
            ))
        self.conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_init = x
        B, C, H, W = x.shape

        for body in self.body:
            x = x.unsqueeze(2)
            x = body[0](x)
            x = x.squeeze(2)

            x = rearrange(x, "b c h w -> b (h w) c")
            x = body[1](x)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return self.conv_out(x + x_init)
