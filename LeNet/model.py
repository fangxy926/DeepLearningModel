#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :  yangman
@Contact :  fangxy926@gmail.com
@File    :  model.py
@Time    :  2020/6/15 下午2:36
@Desc    :  

'''

import tensorflow as tf


class LeNet(tf.keras.Model):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",  # no padding
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=2
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding="valid",
            strides=(1, 1),
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=2
        )
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(
            units=120,
            activation=tf.nn.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=84,
            activation=tf.nn.relu
        )
        self.dense3 = tf.keras.layers.Dense(
            units=10
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        output = tf.nn.softmax(x)
        return output

    # fix the bug: mutiple shape
    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        print(input_shape_nobatch)
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
