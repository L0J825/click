import torch
from einops import rearrange
from torch import nn as nn
from torchac import torchac
import torch.nn.functional as F
from collections import namedtuple
import time


_NUM_PARAMS = 3
_NUM_SUB_IMG = 4
_LOG_SCALES_MIN = -7.
_BOUND_EPS = 0.001
_CDF_LOWER_BOUND = 1e-12
_BIN_WIDTH = 1

_SEQUENTIAL_CDF_CAL_LEVEL_THRESHOLD = 1024
_MAX_PATCH_SIZE_CDF = 64
_MAX_K_FOR_VIS = 10

CDFOut = namedtuple('CDFOut', ['logit_probs_c_sm',
                               'means_c',
                               'log_scales_c',
                               'K',
                               'targets'])


def log_softmax(logit_probs, dim):
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))


def log_sum_exp(log_probs, dim):
    m, _ = torch.max(log_probs, dim=dim)
    m_keep, _ = torch.max(log_probs, dim=dim, keepdim=True)
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


def to_data(symbols: torch.Tensor, x_min: int, dtype=torch.float32) -> torch.Tensor:
    data = symbols.to(dtype) * _BIN_WIDTH + x_min
    return data


def to_symbol(x: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
    symbols = torch.clamp(x, min=x_min, max=x_max)
    symbols = (symbols - x_min) / _BIN_WIDTH
    symbols = torch.round(symbols).long().to(torch.int16)
    return symbols


class DiscreteLogisticMixtureModel(nn.Module):
    def __init__(self, K: int) -> None:
        super().__init__()
        self.K = K

    def forward(self, x, params: torch.Tensor, x_min: int, x_max: int):
        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)

        logit_pis = params[:, 0, ...]
        weights_softmax = torch.softmax(logit_pis, dim=2)

        means = params[:, 1, ...]
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN)
        inv_stdv = torch.exp(-log_scales)

        x = x.unsqueeze(dim=1)
        centered_x = x - means

        plus_in = inv_stdv * (centered_x + _BIN_WIDTH / 2)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - _BIN_WIDTH / 2)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        x_lower_bound = x_min + 0.001
        x_upper_bound = x_max - 0.001
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))

        log_probs = out_A

        log_probs_weighted = log_probs.add(
            log_softmax(logit_pis, dim=2))
        loss = -log_sum_exp(log_probs_weighted, dim=2).sum()
        return loss

    def cdf(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        L = x_max - x_min + 1
        if L > _SEQUENTIAL_CDF_CAL_LEVEL_THRESHOLD:
            return self.sequential_cdf_calculate(params, x_min=x_min, x_max=x_max)
        else:
            return self.parallel_cdf_calculate(params, x_min=x_min, x_max=x_max)

    def sequential_cdf_calculate(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        L = x_max - x_min + 1

        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)
        B, _, C, _, H, W = params.shape

        weights_softmax = torch.softmax(params[:, 0, ...], dim=2).unsqueeze(dim=-1)
        means = params[:, 1, ...].unsqueeze(dim=-1)
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN).unsqueeze(dim=-1)
        inv_sigma = torch.exp(-log_scales)

        targets = torch.linspace(start=x_min - _BIN_WIDTH / 2, end=x_max + _BIN_WIDTH / 2, steps=L + 1,
                                 dtype=torch.float32, device=params.device)

        cdf = torch.zeros(B, C, H, W, L + 1, device=params.device)

        for i in range(H // _MAX_PATCH_SIZE_CDF):
            for j in range(W // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :,
                                             i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                                             j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets *= inv_sigma[:, :, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                                    j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets.sigmoid_()
                centered_targets *= weights_softmax[:, :, :,
                                    i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                                    j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                cdf[:, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF), :] = centered_targets.sum(dim=2)
        if H % _MAX_PATCH_SIZE_CDF != 0:
            start_idx = H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF
            for j in range(W // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :, start_idx:,
                                             j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets = centered_targets * inv_sigma[:, :, :, start_idx:,
                                                      j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                centered_targets.sigmoid_()
                centered_targets = centered_targets * weights_softmax[:, :, :, start_idx:,
                                                      j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)]
                cdf[:, :, start_idx:,
                j * _MAX_PATCH_SIZE_CDF: min(W, (j + 1) * _MAX_PATCH_SIZE_CDF)] = centered_targets.sum(dim=2)
        if W % _MAX_PATCH_SIZE_CDF != 0:
            start_idx = W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF
            for i in range(H // _MAX_PATCH_SIZE_CDF):
                centered_targets = targets - means[:, :, :,
                                             i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF), start_idx:]
                centered_targets = centered_targets * inv_sigma[:, :, :,
                                                      i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                                                      start_idx:]
                centered_targets.sigmoid_()
                centered_targets = centered_targets * weights_softmax[:, :, :,
                                                      i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                                                      start_idx:]
                cdf[:, :, i * _MAX_PATCH_SIZE_CDF: min(H, (i + 1) * _MAX_PATCH_SIZE_CDF),
                start_idx:] = centered_targets.sum(dim=2)
        if H % _MAX_PATCH_SIZE_CDF != 0 and W % _MAX_PATCH_SIZE_CDF != 0:
            centered_targets = targets - means[:, :, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:,
                                         W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            centered_targets = centered_targets * inv_sigma[:, :, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:,
                                                  W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            centered_targets.sigmoid_()
            centered_targets = centered_targets * weights_softmax[:, :, :,
                                                  H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:,
                                                  W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:]
            cdf[:, :, H // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:,
            W // _MAX_PATCH_SIZE_CDF * _MAX_PATCH_SIZE_CDF:] = centered_targets.sum(dim=2)
        return cdf

    def parallel_cdf_calculate(self, params: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        L = x_max - x_min + 1

        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)

        weights_softmax = torch.softmax(params[:, 0, ...], dim=2)
        means = params[:, 1, ...]
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN)
        inv_sigma = torch.exp(-log_scales).unsqueeze(dim=-1)

        targets = torch.linspace(start=x_min - _BIN_WIDTH / 2, end=x_max + _BIN_WIDTH / 2, steps=L + 1,
                                 dtype=torch.float32, device=params.device)

        centered_targets = targets - means.unsqueeze(dim=-1)
        centered_targets = centered_targets * inv_sigma
        centered_targets.sigmoid_()
        centered_targets = centered_targets * weights_softmax.unsqueeze(dim=-1)
        cdf = centered_targets.sum(dim=2)
        return cdf


class ParametersEstimatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AutoregressiveContextExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EntropyModel(nn.Module):
    def __init__(self, K: int, channels_ctx: int, channels_data: int, num_layers: int = 10) -> None:
        super().__init__()
        self.auto_ctx_extraction = nn.ModuleList([
            AutoregressiveContextExtraction(in_channels=i * channels_data, out_channels=channels_ctx) for i in
            range(1, _NUM_SUB_IMG)
        ])

        self.params_estimator = nn.ModuleList([
            ParametersEstimatorBlock(in_channels=(1 + int(i > 0)) * channels_ctx,
                                     out_channels=channels_data * num_layers * _NUM_PARAMS) for
            i in range(_NUM_SUB_IMG)
        ])

        self.discrete_logistic_mixture_model = DiscreteLogisticMixtureModel(K)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, x_min: int, x_max: int):
        sub_x = self.spatial_split(x)
        total_bits = 0
        for i in range(_NUM_SUB_IMG):
            if i == 0:
                params = self.params_estimator[i](ctx)
            else:
                auto_ctx = self.auto_ctx_extraction[i - 1](
                    torch.cat(sub_x[:i], dim=1))
                params = self.params_estimator[i](torch.cat([ctx, auto_ctx], dim=1))
            loss = self.discrete_logistic_mixture_model(sub_x[i], params, x_min, x_max).sum()
            total_bits += loss
        bpp = total_bits / (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
        return bpp

    @torch.no_grad()
    def compress(self, x: torch.Tensor, ctx: torch.Tensor, x_min: int, x_max: int):
        sub_x = self.spatial_split(x)
        strings = []
        for i in range(_NUM_SUB_IMG):
            if i == 0:
                params = self.params_estimator[i](ctx)
            else:
                auto_ctx = self.auto_ctx_extraction[i - 1](
                    torch.cat(sub_x[:i], dim=1))
                params = self.params_estimator[i](torch.cat([ctx, auto_ctx], dim=1))
            cdf = self.discrete_logistic_mixture_model.cdf(params, x_min=x_min, x_max=x_max).cpu()
            symbols = to_symbol(sub_x[i], x_min=x_min, x_max=x_max).cpu()
            strings.append(torchac.encode_float_cdf(cdf_float=cdf, sym=symbols))
        return strings

    @torch.no_grad()
    def decompress(self, strings: list, ctx: torch.Tensor, x_min: int, x_max: int) -> torch.Tensor:
        assert len(strings) == _NUM_SUB_IMG, f'Number of bitstreams {len(strings)} is not equal to {_NUM_SUB_IMG}.'
        sub_x = []
        for i in range(_NUM_SUB_IMG):
            if i == 0:
                params = self.params_estimator[i](ctx)
            else:
                auto_ctx = self.auto_ctx_extraction[i - 1](torch.cat(sub_x[:i], dim=1))
                params = self.params_estimator[i](torch.cat([ctx, auto_ctx], dim=1))
            cdf = self.discrete_logistic_mixture_model.cdf(params, x_min=x_min, x_max=x_max).cpu()
            symbols = torchac.decode_float_cdf(cdf_float=cdf, byte_stream=strings[i])
            sub_x.append(to_data(symbols, x_min=x_min, dtype=ctx.dtype).to(ctx.device))
        return self.spatial_merge(sub_x)

    @staticmethod
    def spatial_split(x: torch.Tensor) -> tuple:
        upper_left = x[:, :, ::2, ::2]
        upper_right = x[:, :, ::2, 1::2]
        bottom_left = x[:, :, 1::2, ::2]
        bottom_right = x[:, :, 1::2, 1::2]
        return upper_left, bottom_right, upper_right, bottom_left

    @staticmethod
    def spatial_merge(sub_list: list) -> torch.Tensor:
        assert len(sub_list) == _NUM_SUB_IMG
        upper_left, bottom_right, upper_right, bottom_left = sub_list
        B, C, H_half, W_half = upper_left.shape
        x = torch.zeros(B, C, H_half * 2, W_half * 2, dtype=upper_left.dtype, device=upper_left.device)
        x[:, :, ::2, ::2] = upper_left
        x[:, :, ::2, 1::2] = upper_right
        x[:, :, 1::2, ::2] = bottom_left
        x[:, :, 1::2, 1::2] = bottom_right
        return x
