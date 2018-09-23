# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:03:03 2018

@author: 123
"""
import os
import cv2
import numpy as np


num = 0
finished = False
def read_lfw(lfw_path):
    global num, finished
    for dir_item in os.listdir(lfw_path): # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(lfw_path, dir_item))
        
        if os.path.isdir(full_path): # 如果是文件夹，继续递归调用，去读取文件夹里的内容
            read_lfw(full_path)
        else: # 如果是文件了
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # 加载分类器
                path_name = 'dataset/training_data_others/'
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度化
                faceRects=classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        
                if len(faceRects) > 0:
                    for faceRect in faceRects:
                        x,y,w,h = faceRect
                        # 捕捉到的图片的名字，这里用到了格式化字符串的输出
                        image_name = '%s%d.jpg' % (path_name, num) # 注意这里图片名一定要加上扩展名，否则后面imwrite的时候会报错：could not find a writer for the specified extension in function cv::imwrite_ 参考：https://stackoverflow.com/questions/9868963/cvimwrite-could-not-find-a-writer-for-the-specified-extension
                        image = image[y:y+h, x:x+w] # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位
                        cv2.imwrite(image_name, image)
                        num += 1
                        if num > 3000:
                            finished = True
                            break
        if finished:
            print('Finished.')
            break
#                images.append(image)
#    return images
#def crop_lfw(lfw_path):
#    images = np.array(read_lfw(lfw_path))
#    
    

if __name__ =='__main__':
    print ('Processing lfw dataset...')
    read_lfw('lfw/') # 注意这里的training_data 文件夹就在程序工作目录下