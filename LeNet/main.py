#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Author  :  yangman
@Contact :  fangxy926@gmail.com
@File    :  main.py
@Time    :  2020/6/15 下午3:15
@Desc    :  

'''

from LeNet.model import LeNet
from LeNet.data_loader import MNISTLoader
import tensorflow as tf
import os

EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001
model_save_dir = "modelfiles/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

data_loader = MNISTLoader()
model = LeNet()
model.build_graph(input_shape=(None, 32, 32, 1))
model.summary()
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
model.save_weights(filepath=model_save_dir, save_format="tf")
