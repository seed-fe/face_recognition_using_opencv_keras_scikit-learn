# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:33:17 2018

@author: 123
"""

import os
import numpy as np
import cv2
from keras.models import load_model
import pickle

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
    images = [] # 用来存放图片
    labels = [] # 用来存放类别标签
    sample_nums = [] # 用来存放不同类别的人脸数据量
    classes = os.listdir(data_dir) # 通过数据集路径下文件夹的数量得到所有类别
    category = 0 # 分类标签计数
    for person in classes: # person是不同分类人脸的文件夹名
        person_dir = os.path.join(data_dir, person) # person_dir是某一分类人脸的路径名
        person_pics = os.listdir(person_dir) # 某一类人脸路径下的全部人脸数据文件
        for face in person_pics: # face是某一分类文件夹下人脸图片数据的文件名
            img = cv2.imread(os.path.join(person_dir, face)) # 通过os.path.join得到人脸图片的绝对路径
            if img is None: # 遇到部分数据有点问题，报错'NoneType' object has no attribute 'shape'
                pass
            else:
                img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img) # 得到某一分类下的所有图片
            labels.append(category) # 给某一分类下的所有图片赋予分类标签值
        sample_nums.append(len(person_pics)) # 得到某一分类下的样本量
        category += 1
    images = np.array(images)
    labels = np.array(labels)
    print("Number of classes: ", len(classes)) # 输出分类数
    for i in range(len(sample_nums)):
        print("Number of the sample of class ", i, ": ", sample_nums[i]) # 输出每个类别的样本量
    return images, labels


# 建立facenet模型
facenet = load_model('./model/facenet_keras.h5') # bad marshal data (unknown type code)，用Python2实现的模型时会报这个错
#facenet.summary()

def img_to_encoding(images, model):
    # 这里image的格式就是opencv读入后的格式
    images = images[...,::-1] # Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. 这里的操作实际是对channel这一dim进行reverse，从BGR转换为RGB
    images = np.around(images/255.0, decimals=12) # np.around是四舍五入，其中decimals是保留的小数位数,这里进行了归一化
    # https://stackoverflow.com/questions/44972565/what-is-the-difference-between-the-predict-and-predict-on-batch-methods-of-a-ker
    if images.shape[0] > 1:
        embedding = model.predict(images, batch_size = 128) # predict是对多个batch进行预测，这里的128是尝试后得出的内存能承受的最大值
    else:
        embedding = model.predict_on_batch(images) # predict_on_batch是对单个batch进行预测
    # 报错，operands could not be broadcast together with shapes (2249,128) (2249,)，因此要加上keepdims = True
    embedding = embedding / np.linalg.norm(embedding, axis = 1, keepdims = True) # 注意这个项目里用的keras实现的facenet模型没有l2_norm，因此要在这里加上
    
    return embedding

# 注意这里必须加上if __name__ == "__main__":，否则运行face_knn_classifier.py的时候也会运行load_dataset函数，而不是直接加载存好的数据，会很慢
# 参考  http://blog.konghy.cn/2017/04/24/python-entry-program/
if __name__ == "__main__":
    images, labels = load_dataset('./dataset_image/')
    # 生成128维特征向量
    X_embeddings = img_to_encoding(images, facenet) # 考虑这里分批执行，否则可能内存不够，这里在img_to_encoding函数里通过predict的batch_size参数实现

    # pickle保存数据
    file_embeddings = open('./dataset_pkl/embeddings.pkl', 'wb')
    pickle.dump(X_embeddings, file_embeddings)
    file_embeddings.close()

    file_labels = open('./dataset_pkl/labels.pkl', 'wb')
    pickle.dump(labels, file_labels)
    file_labels.close