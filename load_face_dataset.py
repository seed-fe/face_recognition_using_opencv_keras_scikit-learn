# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:24:17 2018

@author: 123
"""

import os
import numpy as np
import cv2

IMAGE_SIZE = 160 # 指定图像大小

# 按指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)
    
    # 获取图片尺寸
    h, w, _ = image.shape
    
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h,w)
    
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。
    
    # RGB颜色
    BLACK = [0,0,0]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))

# path_name是当前工作目录，后面会由os.getcwd()获得
def load_dataset(data_dir):
    dataset = {}
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        person_images = []
        for f in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, f))
            if img is None: # 遇到部分数据有点问题，报错'NoneType' object has no attribute 'shape'
                pass
            else:
                img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
            person_images.append(img)
        dataset[person] = person_images
    keys = []
    for key in dataset:
        keys.append(key)
    images = []
    labels = []
    for me in dataset[keys[0]]:
        images.append(me)
        labels.append(0)
    for other in dataset[keys[1]]:
        images.append(other)
        labels.append(1)
    images = np.array(images)
    labels = np.array(labels)
    print(len(dataset)) # 2
    print(len(dataset[keys[0]])) # 938
    print(len(dataset[keys[1]])) # 1311
    return images, labels


if __name__ == '__main__':
    path_name = os.getcwd() # 获取当前工作目录
    images = load_dataset('./dataset/')
#    print(labels)
#    print(labels.shape)