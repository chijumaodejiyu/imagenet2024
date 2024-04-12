#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

# 打印使用的python库的版本信息
print(tf.__version__)
print(sys.version_info)

# 常量的定义
train_dir = "F:\chinese-lion/train"
valid_dir = "F:\chinese-lion/train"
print(os.path.exists(train_dir))
print(os.path.exists(valid_dir))
print(os.listdir(train_dir))
print(os.listdir(valid_dir))

# 定义常量
height = 224  # resnet50的处理的图片大小
width = 224  # resnet50的处理的图片大小
channels = 3
batch_size = 32  # 因为处理的图片变大,batch_size变小一点 32->24
num_classes = 2
epochs = 20

# 使用keras中ImageDataGenerator读取数据
# 实例化ImageDataGenerator
# 对于图片数据,在keras里有更高层的封装.读取数据且做数据增强 -> Generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,
    # 此函数是是现在keras中,而非tf.keras中,在tf中,实现数据做归一化,数据取值在-1~1之间.
    # rescale = 1./255,   # 由于preprocessing_function已做了归一化,此处注释; 图像中的每个像素点都是在0~255之间,得到一个0~1之间的数
    rotation_range=40,  # 图片增强的方法,把图片随机旋转一个角度,旋转的角度就在-40~40之间
    width_shift_range=0.2,  # 做水平位移 - 增加位移鲁棒性(如果0~1之间则位移比例随机选数做位移;如果大于1,则是具体的像素大小)
    height_shift_range=0.2,  # 做垂直位移 - 增加位移鲁棒性(如果0~1之间则位移比例随机选数做位移;如果大于1,则是具体的像素大小)
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True,  # 是否随机做水平翻转
    fill_mode='nearest',  # 填充像素规则,用离其最近的像素点做填充
)
# 使用ImageDataGenerator读取图片
# 从训练集的文件夹中读取图片
train_generator = train_datagen.flow_from_directory(train_dir,  # 图片的文件夹位置
                                                    target_size=(height, width),  # 将图片缩放到的大小
                                                    batch_size=batch_size,  # 多少张为一组
                                                    seed=7,  # 随机数种子
                                                    shuffle=True,  # 是否做混插
                                                    class_mode="categorical"
                                                    )  # 控制目标值label的形式-选择onehot编码后的形式
# 从验证集的文件夹中读取图片
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(height, width),
                                                    batch_size=batch_size,
                                                    seed=7,
                                                    shuffle=False,
                                                    class_mode="categorical"
                                                    )
# 查看训练家和验证集分别有多少张数据
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

# 从ImageDataGenerator中读取数据
for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    print(y)

# 构建模型
# ResNet50中50层参数均不变
print("----------------------------构建模型------------------------------------")
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(keras.applications.ResNet50(include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                                                   pooling='avg',  # resnet50模型倒数第二层的输出是三维矩阵-卷积层的输出,做pooling或展平
                                                   weights='imagenet'))  # 参数有两种imagenet和None,None为从头开始训练,imagenet为从网络下载已训练好的模型开始训练
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation='softmax'))  # 因为include_top = False,所以需要自己定义最后一层
resnet50_fine_tune.layers[0].trainable = False  # 因为参数是从imagenet初始化的,所以我们可以只调整最后一层的参数

resnet50_fine_tune.compile(loss="categorical_crossentropy",
                           optimizer="sgd", metrics=['accuracy'])
resnet50_fine_tune.summary()

# 训练模型
checkpoint_save_path = "checkpoint/model.h5"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-------------------')
    resnet50_fine_tune.load_weights(checkpoint_save_path)

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path, verbose=1, save_best_only=True, save_weights_only=True)
print("---------------------------开始训练-----------------------------------")
resnet50_fine_tune.fit(train_generator,
                       steps_per_epoch=train_num // batch_size,
                       epochs=epochs,
                       validation_data=valid_generator,
                       validation_steps=valid_num // batch_size,
                       callbacks=[checkpointer]
                       )
print("-------------------------正在保存model------------------------------")
resnet50_fine_tune.save(
    "checkpoint\model.h5"
)
