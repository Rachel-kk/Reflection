#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Reflection 
@File    ：swintranet_model.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2022/4/16 下午3:39 
'''
import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import utils.util as util
import utils.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from .errnet_model import ERRNetBase, EdgeMap
from .errnet_model import tensor2im
from PIL import Image
from os.path import join

class SwinTraNetModel(ERRNetBase):
    def name(self):
        return 'swintranet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### SwinTransformer #####################')
        networks.print_network(self.net_s)

    def _eval(self):
        self.net_s.eval()

    def _train(self):
        self.net_s.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        self.net_s = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)

        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            #define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize((losses.VGGLoss(self.vgg)))
            self.loss_dic['t_vgg'] = vggloss
            cxloss = losses.ContentLoss()
            self.loss_dic['t_cx'] = cxloss

            self.optimizer = torch.optim.Adam(self.net_s.parameters(),
                                              lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
            self._init_optimizer([self.optimizer])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.no_verbose is False:
            self.print_network()

    def backward(self):
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None


        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)

            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel + self.loss_icnn_vgg * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)

            self.loss_G += self.loss_CX

        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input
        input_i = [input_i]
        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                           feature in hypercolumn]
            # input_i = [input_i]
            input_i.extend(hypercolumn)

        input_i = torch.cat(input_i, dim=1)
        output_i = self.net_s(input_i)

        self.output_i = output_i

        return output_i

    def optimize_parameters(self):
        self._train()
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_s.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer.load_state_dict(state_dict['opt'])
        else:
            state_dict = torch.load(icnn_path, map_location='cuda:0')
            model.net_s.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']

        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict ={
            'icnn': self.net_s.state_dict(),
            'opt': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iterations': self.iterations
        }

        return state_dict

