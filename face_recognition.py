# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:33:58 2018

@author: 123
"""
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2

from face_knn_classifier import Knn_Model
model = Knn_Model()
model.load_model('./model/knn_classifier.model')

# Face Recognition              
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
                if image is None: # 有的时候可能是人脸探测有问题，会报错 error (-215) ssize.width > 0 && ssize.height > 0 in function cv::resize，所以这里要判断一下image是不是None，防止极端情况 https://blog.csdn.net/qq_30214939/article/details/77432167
                    break
                else:
                    faceID = model.predict(image)
#                print(faceID) # [0]
#                print(type(faceID)) # <class 'numpy.ndarray'>
#                print(faceID.shape) # (1,)
#                #如果是“我”
                    if faceID == 0:                                                        
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    #文字提示是谁
                        cv2.putText(frame,'Bill', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                    else:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                            #文字提示是谁
                        cv2.putText(frame,'Unknown', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
#                    pass
        cv2.imshow("Detecting your face.", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

#释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()