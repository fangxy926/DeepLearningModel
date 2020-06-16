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
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout


class AlexNet(tf.keras.Model):

    def __init__(self, output_class_units):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2D(
            filters=96,
            kernel_size=11,
            strides=4,
            padding='valid',
            activation='relu'
        )
        self.pool1 = MaxPool2D(
            pool_size=3,
            strides=2
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=256,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.pool2 = MaxPool2D(
            pool_size=3,
            strides=2
        )
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(
            filters=384,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.conv4 = Conv2D(
            filters=384,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.conv5 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.pool3 = MaxPool2D(
            pool_size=3,
            strides=2
        )
        self.flatten = Flatten()
        self.dense1 = Dense(
            units=4096,
            activation='relu'
        )
        self.dropout1 = Dropout(
            rate=0.5  # drop rate
        )
        self.dense2 = Dense(
            units=4096,
            activation='relu'
        )
        self.dropout2 = Dropout(
            rate=0.5  # drop rate
        )
        self.dense3 = Dense(
            units=output_class_units,
            activation='softmax'
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        output = self.dense3(x)
        return output
