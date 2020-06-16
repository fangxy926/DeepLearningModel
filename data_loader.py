#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :  yangman
@Contact :  fangxy926@gmail.com
@File    :  data_loader.py
@Time    :  2020/6/16 下午4:11
@Desc    :  

'''

import numpy as np
import tensorflow as tf

class MNISTLoader():
    def __init__(self):
        mnist = np.load('dataset/mnist.npz')
        self.x_train, self.x_test = mnist['x_train'], mnist['x_test']
        self.y_train, self.y_test = mnist['y_train'], mnist['y_test']

        # mnist 中图像默认为unit8 (0-255数字)，将其归一化为0-1的浮点数，并增加一维颜色通道
        self.x_train = np.expand_dims(
            self.x_train.astype(np.float32) / 255,
            axis=-1
        )  # [60000, 28, 28, 1]

        self.x_test = np.expand_dims(
            self.x_test.astype(np.float32) / 255,
            axis=-1
        )  # [10000, 28, 28, 1]

        self.y_train = self.y_train.astype(np.int32)
        self.y_test = self.y_test.astype(np.int32)
        self.train_num, self.test_num = self.x_train.shape[0], self.x_test.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.train_num, batch_size)
        return self.x_train[index, :], self.y_train[index]


class AlexNetMNISTLoader():
    def __init__(self):
        mnist = np.load('dataset/mnist.npz')
        self.x_train, self.x_test = mnist['x_train'], mnist['x_test']
        self.y_train, self.y_test = mnist['y_train'], mnist['y_test']

        # mnist 中图像默认为unit8 (0-255数字)，将其归一化为0-1的浮点数，并增加一维颜色通道
        self.x_train = np.expand_dims(
            self.x_train.astype(np.float32) / 255,
            axis=-1
        )  # [60000, 28, 28, 1]

        self.x_test = np.expand_dims(
            self.x_test.astype(np.float32) / 255,
            axis=-1
        )  # [10000, 28, 28, 1]

        self.y_train = self.y_train.astype(np.int32)
        self.y_test = self.y_test.astype(np.int32)
        self.train_num, self.test_num = self.x_train.shape[0], self.x_test.shape[0]

    def get_batch_train(self,batch_size):
        index = np.random.randint(0, self.train_num, batch_size)
        # need to resize images to (227,227)
        resized_images = tf.image.resize_with_pad(self.x_train[index], 227, 227, )
        return resized_images.numpy(), self.y_train[index]
