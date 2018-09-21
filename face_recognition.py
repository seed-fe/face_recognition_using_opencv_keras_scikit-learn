# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:33:58 2018

@author: 123
"""
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2

from load_face_dataset import IMAGE_SIZE, resize_image

import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks import *

# 建立模型
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

# 定义triplet loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive))) # reduce_sum Computes the sum of elements across dimensions of a tensor. If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
    # print(pos_dist.shape)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # print(basic_loss.shape)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    # loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    loss = tf.maximum(basic_loss, 0)
    ### END CODE HERE ###
    
    return loss

# 载入训练好的权重
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# 建立人脸数据库，将已获取的人脸图片编码为128维的向量
database = {}
database["Bill"] = img_to_encoding(cv2.imread("images/Bill.jpg"), FRmodel)
database["Lin"] = img_to_encoding(cv2.imread("images/Lin.jpg"), FRmodel)

# Face Recognition
def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding()
    encoding = img_to_encoding(image, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    ### END CODE HERE ###
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity
   
              
#框住人脸的矩形边框颜色       
cv2.namedWindow('Detecting your face.') # 创建窗口
color = (0, 255, 0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # 加载分类器
#捕获指定摄像头的实时视频流
cap = cv2.VideoCapture(0)
while cap.isOpened():
        ok, frame = cap.read() # type(frame) <class 'numpy.ndarray'>
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度化
        faceRects=classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                '''
                改用facenet的思路：
                1、准备几张我的人脸照片（和其他几张人脸照片？）
                2、载入训练好的facenet模型
                3、参考吴恩达编程作业里的：建立人脸和ID数据库，用里面的who is it函数实现人脸识别
                '''
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                min_dist, identity = who_is_it(image, database, FRmodel)
#                print(faceID) # [0]
#                print(type(faceID)) # <class 'numpy.ndarray'>
#                print(faceID.shape) # (1,)
                # 如果在数据库
                if min_dist < 0.7:
                     #如果是“我”
                     if identity == "Bill":                                                        
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                        #文字提示是谁
                        cv2.putText(frame,'Bill', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
               
        cv2.imshow("Detecting your face.", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

#释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()