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

def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


def runet(in_channels, out_channels, **kwargs):
    return UNet_SE(in_channels, out_channels = out_channels)