#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：train_runet.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/25 下午3:19 
'''

from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import utils.util as util
import data
import os


def set_learning_rate(lr, engine):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

def main():
    opt = TrainOptions().parse()
    cudnn.benchmark = True
    opt.display_freq = 10

    if opt.debug:
        opt.display_id = 1
        opt.display_freq = 20
        opt.print_freq = 20
        opt.nEpochs = 40
        opt.max_dataset_size = 100
        opt.no_log = False
        opt.nThreads = 0
        opt.decay_iter = 0
        opt.serial_batches = True
        opt.no_flip = True

# modify the following code to
# datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'
# datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
# datadir_real = join(datadir, 'real_train')

# modify the following code to
    datadir = '/home/iv/Annotation/KX/data/Reflection'
    datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
    datadir_real = join(datadir, 'real_train')

    train_dataset = datasets.CEILDataset(
        datadir_syn, read_fns('../VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
        low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
        low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)


    train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

    train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

    train_dataloader_fusion = datasets.DataLoader(
        train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=opt.nThreads, pin_memory=True)

    eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testsets/testdata_CEILNET_table2'))

    eval_dataset_real = datasets.CEILTestDataset(
        join(datadir, 'testsets/real20'),
    fns=read_fns('../real_test.txt'), enable_transforms=True)

    eval_dataloader_ceilnet = datasets.DataLoader(
        eval_dataset_ceilnet, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    eval_dataloader_real = datasets.DataLoader(
        eval_dataset_real, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)


    """Main Loop"""
    engine = Engine(opt)

    if opt.resume:
        res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

    # define training strategy
    engine.model.opt.lambda_gan = 0
    # engine.model.opt.lambda_gan = 0.01
    set_learning_rate(1e-4, engine)
    while engine.epoch < 150:
        # if engine.epoch == 20:
        #     engine.model.opt.lambda_gan = 0.01 # gan loss is added after epoch 20
        # if engine.epoch == 30:
        #     set_learning_rate(5e-5, engine)
        # if engine.epoch == 40:
        #     set_learning_rate(1e-5, engine)
        # if engine.epoch == 45:
        #     ratio = [0.5, 0.5]
        #     print('[i] adjust fusion ratio to {}'.format(ratio))
        #     train_dataset_fusion.fusion_ratios = ratio
        #     set_learning_rate(5e-5, engine)
        # if engine.epoch == 50:
        #     set_learning_rate(1e-5, engine)

        engine.train(train_dataloader_fusion)

        if engine.epoch % 5 == 0:
            engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
            engine.eval(eval_dataloader_real, dataset_name='testdata_real20')


if __name__ == '__main__':
    # os.system('python train_errnet.py --name errnet --hyper')
    # print('222222')
    main()