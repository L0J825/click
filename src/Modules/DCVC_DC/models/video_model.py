# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time

import torch
from torch import nn
import numpy as np

from .common_model import CompressionModel
from .video_net import ME_Spynet, ResBlock, UNet, bilinearupsacling, bilineardownsacling, \
    get_hyper_enc_dec_models, flow_warp
from .layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x + 3 + 2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.repeat(1, self.offset_num, 1, 1)
        x = x.view(B * self.group_num * self.offset_num, C // self.group_num, H, W)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, x, context, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, context), dim=1))
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, x, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        mv = self.dec_3(out)
        return mv, feature


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        channel_N = 64

        self.optic_flow = ME_Spynet()
        self.align = OffsetDiversity(inplace=inplace)

        self.mv_encoder = MvEnc(2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N, inplace=inplace)

        self.mv_y_prior_fusion_adaptor_0 = DepthConvBlock(channel_mv * 1, channel_mv * 2,
                                                          inplace=inplace)
        self.mv_y_prior_fusion_adaptor_1 = DepthConvBlock(channel_mv * 2, channel_mv * 2,
                                                          inplace=inplace)

        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(channel_mv * 2, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
        )

        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 2, inplace=inplace),
        )

        self.mv_decoder = MvDec(2, channel_mv, inplace=inplace)

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(3)])
        self.feature_extractor = FeatureExtractor(inplace=inplace)
        self.context_fusion_net = MultiScaleContextFusion(inplace=inplace)

        self.contextual_encoder = ContextualEncoder(inplace=inplace)

        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(g_ch_16x, g_ch_16x, True, inplace=inplace)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.y_prior_fusion_adaptor_0 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_1 = DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3,
                                                       inplace=inplace)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace)
        self.recon_generation_net = ReconGeneration(inplace=inplace)

        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            mv_y_q_scale_enc_fine = np.linspace(np.log(self.mv_y_q_scale_enc[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_enc[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_enc_fine = np.exp(mv_y_q_scale_enc_fine)
            mv_y_q_scale_dec_fine = np.linspace(np.log(self.mv_y_q_scale_dec[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_dec[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_dec_fine = np.exp(mv_y_q_scale_dec_fine)

            y_q_scale_enc_fine = np.linspace(np.log(self.y_q_scale_enc[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_enc[3, 0, 0, 0]), 64)
            self.y_q_scale_enc_fine = np.exp(y_q_scale_enc_fine)
            y_q_scale_dec_fine = np.linspace(np.log(self.y_q_scale_dec[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_dec[3, 0, 0, 0]), 64)
            self.y_q_scale_dec_fine = np.exp(y_q_scale_dec_fine)

    def multi_scale_feature_extractor(self, dpb, index):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            index = index % 4
            index_map = [0, 1, 0, 2]
            index = index_map[index]
            feature = self.feature_adaptor[index](dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv, index):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, index)
        context1_init = flow_warp(ref_feature1, mv)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv), dim=1), mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        mv_y_q_scale_enc = ckpt["mv_y_q_scale_enc"].reshape(-1)
        mv_y_q_scale_dec = ckpt["mv_y_q_scale_dec"].reshape(-1)
        return y_q_scale_enc, y_q_scale_dec, mv_y_q_scale_enc, mv_y_q_scale_dec

    def mv_prior_param_decoder(self, mv_z_hat, dpb, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            mv_params = self.mv_y_prior_fusion_adaptor_0(mv_params)
        else:
            mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
            mv_params = self.mv_y_prior_fusion_adaptor_1(mv_params)
        mv_params = self.mv_y_prior_fusion(mv_params)
        return mv_params

    def res_prior_param_decoder(self, z_hat, dpb, context3, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            params = torch.cat((temporal_params, hierarchical_params), dim=1)
            params = self.y_prior_fusion_adaptor_0(params)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        params = self.y_prior_fusion(params)
        return params

    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def motion_estimation_and_mv_encoding(self, x, dpb, mv_y_q_enc):
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        ref_mv_feature = dpb["ref_mv_feature"]
        mv_y = self.mv_encoder(est_mv, ref_mv_feature, mv_y_q_enc)
        return mv_y

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, self.mv_y_q_basic_enc, q_index=q_index)
        mv_y_q_scale_dec = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, self.mv_y_q_basic_dec, q_index=q_index)

        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec

    def compress(self, x, dpb, q_in_ckpt, q_index, frame_idx):
        """
        压缩函数，执行视频帧的压缩处理。

        :param x: 待压缩的视频帧。
        :param dpb: 解码图像缓冲区，用于存储已解码的帧。
        :param q_in_ckpt: 量化参数的检查点。
        :param q_index: 量化索引。
        :param frame_idx: 帧索引，用于标识当前处理的帧。
        :return: 包含压缩后帧信息和位流的结果字典。
        """

        # 获取量化器用于推理的参数
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        # q_index = 0
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        # 运动估计和运动向量编码
        mv_y = self.motion_estimation_and_mv_encoding(x, dpb, mv_y_q_enc)

        # 对运动向量进行padding以适应后续处理
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)

        # 通过对运动向量进行超前编码来获得mv_z
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)

        # 对mv_z进行量化
        mv_z_hat = torch.round(mv_z)

        # 解码运动向量的参数
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)

        # 使用解码后的运动向量参数压缩四个部分的先验
        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
            mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = \
            self.compress_four_part_prior(
                mv_y, mv_params,
                self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
                self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        # 使用运动向量解码器解码运动向量
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)

        # 通过运动补偿获得上下文信息
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        # 使用上下文编码器对帧进行编码
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)

        # 对编码后的帧进行padding
        y_pad, slice_shape = self.pad_for_y(y)

        # 对上下文进行超前编码
        z = self.contextual_hyper_prior_encoder(y_pad)

        # 对z进行量化
        z_hat = torch.round(z)

        # 解码上下文参数
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)

        # 使用解码后的上下文参数压缩四个部分的先验
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        # 初始化熵编码器
        self.entropy_coder.reset()

        # 对编码后的运动向量和上下文进行熵编码
        self.bit_estimator_z_mv.encode(mv_z_hat)
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        self.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        self.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)

        # 刷新熵编码器，完成编码
        self.entropy_coder.flush()

        # 获取重构图像和特征
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        # 获取编码后的位流
        bit_stream = self.entropy_coder.get_encoded_stream()

        # 构建结果字典，包含重构图像、特征和位流
        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb, string, height, width, q_in_ckpt, q_index, frame_idx):
        """
        解压函数，用于从给定的压缩字节串中恢复视频帧。

        :param dpb: 解码图像缓冲区，存储已解码的帧。
        :param string: 压缩字节串，待解压的数据。
        :param height: 原始视频帧的高度。
        :param width: 原始视频帧的宽度。
        :param q_in_ckpt: 量化参数的检查点。
        :param q_index: 量化参数的索引。
        :param frame_idx: 当前帧的索引。
        :return: 更新后的解码图像缓冲区。
        """
        # 获取量化参数用于解码
        _, mv_y_q_dec, _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        # 设置熵编码器的输入流
        self.entropy_coder.set_stream(string)
        # 获取模型参数的数据类型和设备
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        # 计算下采样后的尺寸
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        # 计算slice的形状
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        # 从字节流中解码MV的z_hat
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device)
        # 从字节流中解码z_hat
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        # 解码MV参数
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        # 使用解码后的MV参数解压四个部分的先验
        mv_y_hat = self.decompress_four_part_prior(mv_params,
                                                   self.mv_y_spatial_prior_adaptor_1,
                                                   self.mv_y_spatial_prior_adaptor_2,
                                                   self.mv_y_spatial_prior_adaptor_3,
                                                   self.mv_y_spatial_prior)

        # 解码MV并获取特征
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        # 进行动态补偿，获取四个上下文
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        # 解码残差参数
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        # 使用解码的残差参数解压四个部分的先验
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        # 获取重建图像和特征
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        # 返回更新后的解码图像缓冲区
        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
        }

    def encode_decode(self, x, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=0):
        """
        对视频帧进行编码和解码。

        如提供输出路径，将进行帧的编码、解码操作，并计算编码与解码时间及比特率。
        若未提供输出路径，则仅执行编码操作。

        参数:
        x: 输入的视频帧。
        dpb: 已解码图片缓存。
        q_in_ckpt: 内环滤波器的量化参数。
        q_index: 量化参数索引。
        output_path: 保存编码位流的路径。如未提供，则仅执行编码。
        pic_width: 图片宽度。当图片尺寸与输入不同，用于解码时。
        pic_height: 图片高度。当图片尺寸与输入不同，用于解码时。
        frame_idx: 当前帧的索引。

        返回:
        包含已解码图片缓存、比特率、编码时间和解码时间的字典。
        """
        # 检查是否提供了输出路径以执行编码和解码
        # pic_width 和 pic_height 可能与 x 的尺寸不同。x 经过填充后使用
        # x_hat 与 x 尺寸相同
        # output_path = '/root/7.3T/RJ/MC_new/logs/modelsTest/00.bin'
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()

            # 执行压缩（编码）
            encoded = self.compress(x, dpb, q_in_ckpt, q_index, frame_idx)

            # 保存编码后的位流到指定路径
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, frame_idx, output_path)

            # 计算位流大小（以比特为单位）
            bits = filesize(output_path) * 8

            torch.cuda.synchronize(device=device)
            t1 = time.time()

            # 执行解码准备：从文件中读取量化参数等信息
            q_in_ckpt, q_index, frame_idx, string = decode_p(output_path)

            # 使用解码得到的信息进行解压（解码）
            decoded = self.decompress(dpb, string, pic_height, pic_width,
                                      q_in_ckpt, q_index, frame_idx)

            torch.cuda.synchronize(device=device)
            t2 = time.time()

            # 组织结果数据
            result = {
                "dpb": decoded["dpb"],  # 解码后图片缓存
                "bit": bits,  # 比特率
                "encoding_time": t1 - t0,  # 编码时间
                "decoding_time": t2 - t1,  # 解码时间
            }
            return result, encoded['bit_stream']

        # 无输出路径时，仅执行单帧的前向编码操作
        encoded = self.forward_one_frame(x, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index,
                                         frame_idx=frame_idx)

        # 组织仅编码情况下的结果数据
        result = {
            "dpb": encoded['dpb'],  # 编码后更新的dpb
            "bit": encoded['bit'].item(),  # 比特数
            "encoding_time": 0,  # 编码时间（此处默认为0，实际应根据具体实现计算）
            "decoding_time": 0,  # 解码时间（此处默认为0，因未执行解码）
        }
        return result

    def forward_one_frame(self, x, dpb, q_in_ckpt=False, q_index=None, frame_idx=0):
        """
        前向传播一个视频帧。

        该方法通过对给定的视频帧应用编码和解码过程，来生成重构的视频帧和相关的码流信息。
        它包括运动估计、运动向量编码、上下文建模和量化等多个步骤。

        :param x: 待编码的视频帧。
        :param dpb: 解码图像缓冲区，包含先前解码的帧和相关特征。
        :param q_in_ckpt: 是否在检查点进行量化。
        :param q_index: 量化索引。
        :param frame_idx: 帧索引。
        :return: 重构的视频帧、码流信息和相关特征。
        """

        # 获取运动向量和量化参数
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        # 运动估计，获取参考帧的光流
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        # 运动向量编码
        mv_y = self.mv_encoder(est_mv, dpb["ref_mv_feature"], mv_y_q_enc)

        # 对运动向量进行padding以适应后续处理
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        # 编码运动向量的超前向
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        # 量化运动向量的超前向
        mv_z_hat = self.quant(mv_z)
        # 解码运动向量的先验参数
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        # 应用四部分先验来进一步处理运动向量
        _, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        # 解码运动向量
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)

        # 运动补偿，生成三个不同尺度的上下文
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        # 上下文编码，结合运动补偿的信息来编码当前帧
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        # 对编码后的上下文进行padding
        y_pad, slice_shape = self.pad_for_y(y)
        # 编码上下文的超前向
        z = self.contextual_hyper_prior_encoder(y_pad)
        # 量化上下文的超前向
        z_hat = self.quant(z)
        # 解码上下文的先验参数
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        # 应用四部分先验来进一步处理上下文
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        # 获取重构的视频帧和相关特征
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        # 计算每个像素的码率
        _, _, H, W = x.size()
        pixel_num = H * W

        # 准备用于编码的量化的特征
        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat

        # 计算各个部分的码流
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        # 计算每像素的码率
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        # 计算总码率
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        # 返回重构帧、码流信息和相关特征
        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_mv_feature": mv_feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }


