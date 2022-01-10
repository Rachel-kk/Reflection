#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch 
@File    ：image_folder.py
@IDE     ：PyCharm 
@Author  ：Rachel_kx
@Date    ：2021/12/5 上午11:41 
'''

from PIL import Image
import os
import os.path

# 图像扩展名
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM',
    '.bmp', '.BMP'
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]

    return fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, filenames in sorted(os.walk(dir)):
            for fname in filenames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


if __name__ == '__main__':
    datadir = '/home/rackel_kk/Project/GitWorkspace/data/Reflection/VOCdevkit/VOC2012/PNGImages'
    sortkey = lambda key: os.path.split(key)[-1]
    paths = sorted(make_dataset(datadir, read_fns('../../VOC2012_224_train_png.txt')), key=sortkey)
    print(paths)