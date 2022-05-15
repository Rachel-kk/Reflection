#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：__init__.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/9 下午8:09 
'''
from .default import DRNet
from .default import UNet_SE
from .swintransformer import SwinRR


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels,
                 out_channels,
                 256,
                 13,
                 norm=None,
                 res_scale=0.1,
                 bottom_kernel_size=1,
                 **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels,
                 out_channels,
                 256,
                 13,
                 norm=None,
                 res_scale=0.1,
                 se_reduction=8,
                 bottom_kernel_size=1,
                 pyramid=True,
                 **kwargs)


def runet(in_channels, out_channels, **kwargs):
    return UNet_SE(in_channels,
                   out_channels = out_channels,
                   **kwargs)

# 训练Swin-Transformer网络，估计传射图
def swintranet(in_channels, out_channels, **kwargs):
   return SwinRR(in_channels,
                 out_channels,
                 upscale=1,
                 img_size=224,
                 window_size=8,
                 img_range=1.,
                 depths=[6, 6],
                 embed_dim=180,
                 num_heads=[6, 6],
                 mlp_ratio=2,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs)