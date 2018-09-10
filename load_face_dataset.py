# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:24:17 2018

@author: 123
"""

import os
import numpy as np
import cv2

IMAGE_SIZE = 64 # 指定图像大小

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

# 读取训练数据到内存
images = []
labels = []

# path_name是当前工作目录，后面会由os.getcwd()获得
def read_path(path_name):
    for dir_item in os.listdir(path_name): # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path): # 如果是文件夹，继续递归调用，去读取文件夹里的内容
            read_path(full_path)
        else: # 如果是文件了
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                images.append(image)
                labels.append(path_name)
    return images, labels
# 读取训练数据并完成标注
def load_dataset(path_name):
    images,labels = read_path(path_name)
    # 将lsit转换为numpy array
    images = np.array(images, dtype='float') # 注意这里要将数据类型设为float，否则后面face_train_keras.py里图像归一化的时候会报错，TypeError: No loop matching the specified signature and casting was found for ufunc true_divide
    print(images.shape) # (1969, 64, 64, 3)
    # 标注数据，me文件夹下是我，指定为0，其他指定为1，这里的0和1不是logistic regression二分类输出下的0和1，而是softmax下的多分类的类别
    labels = np.array([0 if label.endswith('me') else 1 for label in labels])
    return images, labels

if __name__ == '__main__':
    path_name = os.getcwd() # 获取当前工作目录
    images, labels = load_dataset(path_name)