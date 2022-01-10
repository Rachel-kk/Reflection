#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：default.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/9 下午8:10 
'''
# Define network components here
import torch
from torch import nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d,
                 se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class UNet_SE(torch.nn.Module):
    def __init__(self, in_channels, channel=32, out_channels=3):
        super(UNet_SE, self).__init__()
        # Initial convolution layers
        conv  = nn.Conv2d
        lrelu = nn.LeakyReLU(0.2)
        self.conv1 = ConvLayer(conv, in_channels, channel, kernel_size=1, stride=1, act=lrelu)
        self.conv2 = ConvLayer(conv, channel, channel, kernel_size=3, stride=1, padding=1, act=lrelu)

        self.down1 = Down(channel, channel * 2)
        self.down2 = Down(channel * 2, channel * 4)
        self.down3 = Down(channel * 4, channel * 8)
        self.down4 = Down(channel * 8, channel * 16)
        self.conv5 = ConvLayer(conv, channel * 16, channel * 16, kernel_size=3,stride=1, padding=1, act=lrelu)
        self.fcat  = AttentionChannel(channel * 16, channel * 16)

        self.up6   = Bilinear_up_and_concat(channel * 8, channel * 16)
        self.conv6 = DoubleConv(channel * 16, channel * 8)
        self.up7   = Bilinear_up_and_concat(channel * 4, channel * 8)
        self.conv7 = DoubleConv(channel * 8, channel * 4)
        self.up8   = Bilinear_up_and_concat(channel * 2, channel * 4)
        self.conv8 = DoubleConv(channel * 4, channel * 2)
        self.up9   = Bilinear_up_and_concat(channel, channel * 2)
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(channel, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        with torch.no_grad():
            conv1 = self.conv1(x)
            conv1 = self.conv2(conv1)
            conv2 = self.down1(conv1)
            conv3 = self.down2(conv2)
            conv4 = self.down3(conv3)
            conv5 = self.down4(conv4)
            conv5 = self.conv5(conv5)
            # (BCHW) => (BHWC)
            global_pooling = torch.mean(conv5, dim=(0, 2, 3), keepdim=True)
            attention_channel = self.fcat(global_pooling).unsqueeze(2).unsqueeze(3)
            conv5 = conv5 * attention_channel

            up6   = self.up6(conv5, conv4)
            conv6 = self.conv6(up6)
            up7   = self.up7(conv6, conv3)
            conv7 = self.conv7(up7)
            up8   = self.up8(conv7, conv2)
            conv8 = self.conv8(up8)
            up9   = self.up9(conv8, conv1)
            conv9 = self.conv9(up9)
            out   = self.conv10(conv9)
        return out


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None,
                 act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class DoubleConv(nn.Module):
    """
    (convolution => LReLU(0.2)) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Bilinear_up_and_concat(nn.Module):
    def __init__(self, out_channels, in_channels):
        super(Bilinear_up_and_concat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        upconv = self.up(x1) # [b, c, 2*h, 2*w]
        upconv = self.conv(upconv)
        upconv_output = torch.cat([upconv, x2], dim=1)
        return upconv_output


class AttentionChannel(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(AttentionChannel, self).__init__()
        self.attention_channel = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            # nn.Sigmoid()
        )


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm,
                               act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
