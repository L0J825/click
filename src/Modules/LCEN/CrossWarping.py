import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def create_identity_grid_from_tensor(reference_tensor, H, W):
    B = reference_tensor.size(0)
    dtype = reference_tensor.dtype

    y_coords = reference_tensor.new_zeros(H)
    x_coords = reference_tensor.new_zeros(W)

    for i in range(H):
        y_coords[i] = float(i)
    for i in range(W):
        x_coords[i] = float(i)

    h_minus_1 = max(float(H - 1), 1.0)
    w_minus_1 = max(float(W - 1), 1.0)
    y_normalized = -1.0 + 2.0 * y_coords / h_minus_1
    x_normalized = -1.0 + 2.0 * x_coords / w_minus_1

    y_grid = y_normalized.view(H, 1).expand(H, W)
    x_grid = x_normalized.view(1, W).expand(H, W)

    grid = torch.stack([x_grid, y_grid], dim=2)

    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    return grid


class CrossWarpingModule(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 8, num_heads: int = 8, num_sub_img=4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_sub_img = num_sub_img

        self.query_map = nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.key_map = nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.value_map = nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.ver_map = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, bias=False)
        self.hor_map = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, bias=False)

    def forward(self, cur_fea: torch.Tensor, ref_fea: torch.Tensor) -> torch.Tensor:
        cur_fea_sub = self.spatial_split(cur_fea)
        ref_fea_sub = self.spatial_split(ref_fea)

        warped_sub_list = []
        for cur_sub, ref_sub in zip(cur_fea_sub, ref_fea_sub):
            warped_feature_sub = self.process_sub_image(cur_sub, ref_sub)
            warped_sub_list.append(warped_feature_sub)

        warped_feature = self.spatial_merge(warped_sub_list)

        return warped_feature

    def process_sub_image(self, cur_fea: torch.Tensor, ref_fea: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cur_fea.shape

        query = self.query_map(cur_fea)
        key = self.key_map(ref_fea)
        value = self.value_map(ref_fea)

        query_ver = rearrange(query, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        key_ver = rearrange(key, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        attn_ver = torch.softmax(torch.einsum("bkhc, bklc -> bkhl", query_ver, key_ver), dim=-1)
        del query_ver, key_ver
        value_ver = rearrange(value, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        out_ver = rearrange(torch.einsum("bkhl, bklc -> bkhc", attn_ver, value_ver), '(b w) k h c -> b (k c) h w', w=W)
        out_ver = self.ver_map(out_ver)
        del attn_ver, value_ver

        query_hor = rearrange(query, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        key_hor = rearrange(key, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        attn_hor = torch.softmax(torch.einsum("bkwc, bklc -> bkwl", query_hor, key_hor), dim=-1)
        del query_hor, key_hor
        value_hor = rearrange(value, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        out_hor = rearrange(torch.einsum("bkwl, bklc -> bkwc", attn_hor, value_hor), '(b h) k w c -> b (k c) h w', h=H)
        out_hor = self.hor_map(out_hor)
        del attn_hor, value_hor

        grid = create_identity_grid_from_tensor(cur_fea, H, W)
        flow = torch.cat([
            out_hor / ((W - 1.0) / 2.0),
            out_ver / ((H - 1.0) / 2.0)
        ], 1)
        grid = grid + flow.permute(0, 2, 3, 1)
        warped_feature = F.grid_sample(ref_fea, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped_feature

    def spatial_split(self, x: torch.Tensor) -> tuple:
        upper_left = x[:, :, ::2, ::2]
        upper_right = x[:, :, ::2, 1::2]
        bottom_left = x[:, :, 1::2, ::2]
        bottom_right = x[:, :, 1::2, 1::2]
        return upper_left, bottom_right, upper_right, bottom_left

    def spatial_merge(self, sub_list: list) -> torch.Tensor:
        assert len(sub_list) == self.num_sub_img
        upper_left, bottom_right, upper_right, bottom_left = sub_list
        B, C, H_half, W_half = upper_left.shape
        x = torch.zeros(B, C, H_half * 2, W_half * 2, device=upper_left.device)
        x[:, :, ::2, ::2] = upper_left
        x[:, :, ::2, 1::2] = upper_right
        x[:, :, 1::2, ::2] = bottom_left
        x[:, :, 1::2, 1::2] = bottom_right
        return x


if __name__ == '__main__':
    model = CrossWarpingModule(channels=64, num_heads=8)
    cur_fea = torch.randn(1, 64, 256, 256)
    ref_fea = torch.randn(1, 64, 256, 256)
    warped_feature = model(cur_fea, ref_fea)
    print(warped_feature.shape)
