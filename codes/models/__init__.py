#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：__init__.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/9 下午8:17 
'''
from .errnet_model import ERRNetModel
from .errnet_model import RUNetModel

def errnet_model():
    return ERRNetModel()

def runet_model():
    return RUNetModel()