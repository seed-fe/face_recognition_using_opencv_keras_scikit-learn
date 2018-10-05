# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:53:58 2018

@author: 123
"""
# 生成facenet模型
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
from fr_utils import load_weights_from_FaceNet
from inception_blocks import faceRecoModel

# 建立facenet模型
FRmodel = faceRecoModel(input_shape=(3, 96, 96)) # faceRecoModel在inception_blocks里定义
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

FRmodel.save('./model/facenet.model.h5')