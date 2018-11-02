#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####

import numpy as np






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