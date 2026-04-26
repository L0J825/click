import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Modules.LCEN.CrossWarping import CrossWarpingModule

class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class MultiScaleFeatureExtractor_res(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, blocks=[1, 1, 1]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

        self.body = nn.ModuleList()
        for layer_count in blocks:
            layer_blocks = nn.Sequential(*[ResBlock(out_channel) for _ in range(layer_count)])
            self.body.append(layer_blocks)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.body[0](layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.body[1](layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.body[2](layer3)

        return layer1, layer2, layer3


class EnhancedTemporalFeatureModule_v6(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=64, blocks=[1, 3, 1]):
        super().__init__()
        self.fea = MultiScaleFeatureExtractor_res(in_channel=in_dim, out_channel=hidden_dim, blocks=blocks)
        self.ctx_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.warp = CrossWarpingModule(channels=hidden_dim)

    def forward(self, cur, ref):
        cur_fea1, cur_fea2, cur_fea3 = self.fea(cur)
        ref_fea1, ref_fea2, ref_fea3 = self.fea(ref)

        cur_fea = F.interpolate(cur_fea3, size=(cur_fea2.size()[2:]), mode='bilinear', align_corners=True)
        cur_fea = torch.mul(cur_fea2, cur_fea)
        cur_fea = F.interpolate(cur_fea, size=(cur_fea1.size()[2:]), mode='bilinear', align_corners=True)
        cur_fea = torch.mul(cur_fea1, cur_fea)
        del cur_fea3, cur_fea2, cur_fea1

        ref_fea = F.interpolate(ref_fea3, size=(ref_fea2.size()[2:]), mode='bilinear', align_corners=True)
        ref_fea = torch.mul(ref_fea2, ref_fea)
        ref_fea = F.interpolate(ref_fea, size=(ref_fea1.size()[2:]), mode='bilinear', align_corners=True)
        ref_fea = torch.mul(ref_fea1, ref_fea)
        del ref_fea3, ref_fea2, ref_fea1

        ctx = self.warp(cur_fea, ref_fea)

        return self.ctx_fusion(ctx)


if __name__ == '__main__':
    import torch

    model = EnhancedTemporalFeatureModule_v6(in_dim=1, hidden_dim=64, out_dim=64, blocks=[1, 1, 1], conv_ratio=0.3)

    import torch.optim as optim

    cur = torch.randn(1, 1, 256, 256)
    ref = torch.randn(1, 1, 256, 256)
    target = torch.randn(1, 64, 128, 128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    weights_before = {name: param.data.clone() for name, param in model.named_parameters()}
    model.train()
    output = model(cur, ref)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    weights_after = {name: param.data.clone() for name, param in model.named_parameters()}

    Flag = True
    for name in weights_before.keys():
        if torch.equal(weights_before[name], weights_after[name]):
            print(f"Weight {name} not updated.")
            Flag = False
    if Flag:
        print("Weights updated")
