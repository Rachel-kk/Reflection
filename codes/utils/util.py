#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：util.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/5 下午3:30 
'''
import os
import sys
import time

import torch
import numpy as np

from PIL import Image
import yaml

def get_config(config): # .yaml文件
    with open(config, 'r') as stream:
        return yaml.load(stream) # 转换yaml数据为字典或列表


def tensor2im(image_tensor, imtype=np.uint8):
    '''
    Converts a Tensor into a Numpy array and Denormalization
    （将tensor的数据类型转成numpy类型，并反归一化）
    :param image_tensor:  The input image tensor array(输入的图像tensor数组)，shape: (N, C, H, W)
    :param imtype: The desired type of the converted numpy array(转换后的numpy数组)
    :return: numpy array, (H, W, C),RGB
    '''
    # shape: (C, H, W)
    image_numpy = image_tensor[0].cpu().float().numpy() # convert it into a numpy array
    if image_numpy.shape[0] == 1: # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # shape: (H, W, C), post-processing: tranpose and scaling
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # 将image_tensor里面的数值为(-1, 1)，将其转换成(0, 255)
    image_numpy = image_numpy.astype(imtype)
    if image_numpy.shape[-1] == 6:
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:]], axis=1)
    if image_numpy.shape[-1] == 7:
        edge_map = np.tile(image_numpy[:, :, 6:7], (1, 1, 3))
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:6], edge_map], axis=1)

    return image_numpy

def tensor2numpy(image_tensor):
    '''
    :param image_tensor: The input image tensor array(输入的图像tensor数组)，shape: (N, C, H, W)
    :return: numpy array
    '''
    # Returns a tensor with all the dimensions of input of size 1 removed(降维)
    image_numpy = torch.squeeze(image_tensor).cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.float32)
    return image_numpy

def get_model_list(dirname, key, epoch=None):
    '''
    Get model list for resume
    :param dirname: Folder to store weights
    :param key: first Word
    :param epoch: Number of iterations
    :return: The weight path of the current iteration
    '''
    if epoch is None:
        return os.path.join(dirname, key+'_latest.pt')
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f and 'latest' not in f]
    if gen_models is None:
        return None

    epoch_index = [int(os.path.basename(model_name).split('_')[-2]) for model_name in gen_models
                   if 'latest' not in model_name]
    print('[i] available epoch list: %s' %epoch_index, gen_models)
    i = epoch_index.index(int(epoch))

    return gen_models[i]

def vgg_preprocess(batch):
    '''
    normalize using imagenet mean and std
    :param batch: Batch size
    :return:
    '''
    mean = batch.new(batch.size())
    std = batch.new(batch.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = (batch + 1) / 2
    batch -= mean
    batch = batch / std
    return batch

def diagnose_network(net, name='network'):
    '''
    :param net:
    :param name:
    :return:
    '''
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    '''
    :param image_numpy:
    :param image_path:
    :return:
    '''
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, median = %3.3f, std = %3.3f'
              %(np.mean(x), np.min(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """
    :param paths:
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    '''
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value

def vis(x):
    '''
    Visualization(可视化)
    :param x:
    :return:
    '''
    if isinstance(x, torch.Tensor):
        Image.fromarray(tensor2im(x)).show()
    elif isinstance(x, np.ndarray):
        Image.fromarray(x.astype(np.uint8)).show()
    else:
        raise NotImplementedError('vis for type [%s] is not implemented', type(x))

"""tensorboard"""
from tensorboardX import SummaryWriter
from datetime import datetime


def get_summary_writer(log_dir):
    '''
    get log
    :param log_dir:
    :return:
    '''
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    return writer

class AverageMeters(object):
    def __init__(self, dic=None, total_num=None):
        '''
        :param dic: dictionary type
        :param total_num: dictionary type
        '''
        self.dic = dic or {}
        self.total_num = total_num or {}

    def update(self, new_dic):
        '''
        :param new_dic: dictionary type
        :return:
        '''
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1

    def __getitem__(self, key): # 实例对象P做p[key]运算时，会调用该方法
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ') # self[key]

        return res

    def keys(self):
        return self.dic.keys() # Return all the keys of a dictionary as a list

def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(os.path.join(prefix, key), meter, iteration)

"""progress bar"""
import socket # 套接字


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# 运行上面两句会报错，使用term_width=80替代
term_width = 80
TOTAL_BAR_LENGTH = 65
last_time = time.time()
begin_time = last_time


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'

    return f


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time() # Reset for new bar

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 2):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def parse_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args





# if __name__ == '__main__':
    # a = np.array([0, 1])
    # print(a.ndim)
    # print(np.tile(a, 2))
    # print(np.tile(a, (1, 2)).shape)
    # print(np.tile(a, (2, 1, 2)).shape)
    # print(np.tile(a, (3, 1, 1)))
    #
    # b = np.array([[1, 2], [3, 4]])
    # print(np.tile(b, 2).shape)
    # print(np.tile(b, (2, 2)))
    # print(np.tile(b, (2, 1, 2)))

    # dic = {'xiaomi': 12, 'huawei': 33, 'samsung': 44}
    # total_num = {'xiaomi': 1, 'huawei': 1, 'samsung': 1}
    # avm = AverageMeters(dic, total_num)
    # print(avm.__str__())


