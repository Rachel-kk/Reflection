#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：enums.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/9 下午8:17 
'''
from enum import Enum


class Distance(Enum):
    L2 = 0
    DotProduct = 1


class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3