# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:00:25 2018

@author: 123
"""

# =============================================================================
# facenet+Knn的思路：
# 1、用facenet将所有图片生成128维向量，准备训练数据，将数据集划分为训练集和测试集（70%/30%）
# 2、建立KNN模型，进行训练和测试
# 3、用训练好的模型进行实时人脸识别
# =============================================================================


import tensorflow as tf

#from keras import backend as K
#K.set_image_data_format('channels_first')
from keras.models import load_model
from load_face_dataset import load_dataset, IMAGE_SIZE, resize_image


import numpy as np


from fr_utils import img_to_encoding
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics

# 建立facenet模型
#with CustomObjectScope({'tf': tf}):
facenet = load_model('./model/facenet_keras.h5') # bad marshal data (unknown type code)，这个模型只能用在python2环境

class Dataset:
    # http://www.runoob.com/python3/python3-class.html
    # 很多类都倾向于将对象创建为有初始状态的。
    # 因此类可能会定义一个名为 __init__() 的特殊方法（构造方法），类定义了 __init__() 方法的话，类的实例化操作会自动调用 __init__() 方法。
    # __init__() 方法可以有参数，参数通过 __init__() 传递到类的实例化操作上，比如下面的参数path_name。
    # 类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。
    # self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类。
    def __init__(self, path_name): 
        # 训练集
        self.X_train = None
        self.y_train = None
        
        # 验证集
#        self.valid_images = None
#        self.valid_labels = None
        
        # 测试集
        self.X_test = None
        self.y_test = None
        
        # 数据集加载路径
        self.path_name = path_name
        
        # 当前库采用的维度顺序，包括rows，cols，channels，用于后续卷积神经网络模型中第一层卷积层的input_shape参数
        self.input_shape = None 
    
    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, model = facenet):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)
        # tensorflow 作为后端，数据格式约定是channel_last，与这里数据本身的格式相符，如果是channel_first，就要对数据维度顺序进行一下调整
#        if K.image_data_format == 'channel_first':
#            images = images.reshape(images.shape[0],img_channels, img_rows, img_cols)
#            self.input_shape = (img_channels, img_rows, img_cols)
#        else:
#            images = images.reshape(images.shape[0], img_rows, img_cols, img_channels)
#            self.input_shape = (img_rows, img_cols, img_channels)
#        X_encoding = []
##        在img_to_encoding函数里已经进行了归一化，因此这里不需要再归一化了
#        for image in images:
#            encoding = img_to_encoding(image, model)
#            X_encoding.append(encoding[0])
#        X_encoding = np.array(X_encoding)
#        print(X_encoding.shape)
        X_embedding = img_to_encoding(images, model) # 考虑这里分批执行，否则可能内存不够？
        X_train, X_test, y_train, y_test = train_test_split(X_embedding, labels, test_size = 0.3, random_state = random.randint(0, 100))
#        print(test_labels) # 确认了每次都不一样
        
        # 输出训练集、验证集和测试集的数量
        print('X_train shape', X_train.shape)
        print('y_train shape', y_train.shape)
        print(X_train.shape[0], 'train samples')
#        print(valid_images.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')
        
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

# 定义并训练KNN Classifier模型
class Knn_Model:
    # 初始化构造方法
    def __init__(self):
        self.model = None
    def build_model(self):
        self.model = KNeighborsClassifier()
    def train(self, dataset):
        self.model.fit(dataset.X_train, dataset.y_train)
    def save_model(self, file_path):
        #save model
        joblib.dump(self.model, file_path)
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
    def evaluate(self, dataset):
        predict = self.model.predict(dataset.X_test)
        accuracy = metrics.accuracy_score(dataset.y_test, predict)
        print ('accuracy: %.2f%%' % (100 * accuracy))
    def predict(self, image):
        image = resize_image(image)
        image_embedding = img_to_encoding(np.array([image]), facenet)
        label = self.model.predict(image_embedding)
        return label[0]

# https://teamtreehouse.com/community/getting-a-syntax-error-at-main
# 注意这里遇到了一个bug，在下面的代码上提示invalid syntax，其实是因为上面的print语句少了最后一个圆括号
# The most likely reason for you to get a syntax error at the end of the following valid code
# is because you have a syntax error earlier in the code, 
# but it's not until the interpreter gets to the end of this statement that it realises that there is a problem. 
# A common syntax error you might have is a missing close parenthesis on the last line of your code before this statement.
if __name__ == "__main__":
    dataset = Dataset('./dataset/')
    dataset.load()
    model = Knn_Model()
    model.build_model()
    model.train(dataset)
    model.evaluate(dataset)
    model.save_model('./model/knn_classifier.model')