#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :  yangman
@Contact :  fangxy926@gmail.com
@File    :  main.py
@Time    :  2020/6/16 下午4:09
@Desc    :  

'''

from models.AlexNet import AlexNet
from models.LeNet import LeNet
from data_loader import MNISTLoader, ImageNetLoader
import os
import tensorflow as tf
from logger import Logger
import sys

model_save_dir = "modelfiles/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)


def train_LeNet():
    EPOCHS = 5
    BATCH_SIZE = 50
    LR = 0.001

    data_loader = MNISTLoader()
    model = LeNet()
    # model.build_graph(input_shape=(None, 32, 32, 1))
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    num_train_batches = int(data_loader.train_num // BATCH_SIZE)

    # 训练
    for epoch in range(EPOCHS):
        for batch in range(num_train_batches):
            X, y = data_loader.get_batch(BATCH_SIZE)
            with tf.GradientTape() as tape:
                y_pred = model(X)

                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=y,
                    y_pred=y_pred
                )
                loss = tf.reduce_mean(loss)
                print("epoch {}, batch {}: loss {}".format(epoch + 1, batch + 1, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # 测试
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_test_batches = int(data_loader.test_num // BATCH_SIZE)
    for batch in range(num_test_batches):
        start_index, end_index = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
        y_pred = model.predict(x=data_loader.x_test[start_index:end_index])
        sparse_categorical_accuracy.update_state(
            y_true=data_loader.y_test[start_index:end_index],
            y_pred=y_pred
        )
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
    model.save_weights(filepath=model_save_dir + "lenet_mnist.h5")


def train_AlexNet():
    EPOCHS = 100
    BATCH_SIZE = 32
    image_height = 227
    image_width = 227
    train_dir = "dataset/ImageNet/train"
    valid_dir = "dataset/ImageNet/validation"
    model_dir = model_save_dir + "alexnet_mnist.h5"

    # set gpu
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    image_dataloader = ImageNetLoader(train_dir=train_dir, valid_dir=valid_dir, image_height=image_height,
                                      image_width=image_width)
    train_generator, valid_generator, train_num, valid_num = image_dataloader.get_batch(BATCH_SIZE)

    model = AlexNet(input_shape=(227, 227, 3), num_classes=2)
    model.summary()

    # start training
    model.fit(train_generator,
              epochs=EPOCHS,
              steps_per_epoch=train_num // BATCH_SIZE,
              validation_data=valid_generator,
              validation_steps=valid_num // BATCH_SIZE,
              callbacks=None,
              verbose=1)
    # save the whole model
    model.save(model_dir)


if __name__ == '__main__':
    sys.stdout = Logger("train.log", sys.stdout)
    train_AlexNet()
