#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :  yangman
@Contact :  fangxy926@gmail.com
@File    :  AlexNet.py
@Time    :  2020/6/16 下午3:24
@Desc    :  

'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout


# AlexNet model
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11, 11), strides=4,
                        padding='valid', activation='relu',
                        input_shape=input_shape))
        self.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                           padding='valid', data_format=None))
        self.add(BatchNormalization())
        self.add(Conv2D(256, kernel_size=(5, 5), strides=1,
                        padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                           padding='valid', data_format=None))
        self.add(BatchNormalization())
        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        ))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        ))

        self.add(Conv2D(256, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        ))

        self.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                           padding='valid', data_format=None))

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
