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


class ImageNetLoader():
    def __init__(self, train_dir, valid_dir, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1)

        self.valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    def get_batch(self, batch_size):
        train_generator = self.train_datagen.flow_from_directory(self.train_dir,
                                                                 target_size=(self.image_height, self.image_width),
                                                                 color_mode="rgb",
                                                                 batch_size=batch_size,
                                                                 seed=1,
                                                                 shuffle=True,
                                                                 class_mode="categorical")
        valid_generator = self.valid_datagen.flow_from_directory(self.valid_dir,
                                                                 target_size=(self.image_height, self.image_width),
                                                                 color_mode="rgb",
                                                                 batch_size=batch_size,
                                                                 seed=7,
                                                                 shuffle=True,
                                                                 class_mode="categorical"
                                                                 )
        train_num = train_generator.samples
        valid_num = valid_generator.samples

        return train_generator, valid_generator, train_num, valid_num

