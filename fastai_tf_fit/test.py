# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 15:12
# @Author  : Administrator
# @Email   : happiness_ws@163.com
# @File    : test.py
# @Software: PyCharm

from fastai.vision import *

from fastai_tf_fit import *


def categorical_accuracy(y_pred, y_true):
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, tf.keras.backend.argmax(y_pred, axis=-1)))


class Simple_CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv3 = tf.keras.layers.Conv2D(10, kernel_size=3, strides=(2, 2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, xb):
        xb = tf.nn.relu(self.bn1(self.conv1(xb)))
        xb = tf.nn.relu(self.bn2(self.conv2(xb)))
        xb = tf.nn.relu(self.bn3(self.conv3(xb)))
        xb = tf.nn.pool(input=xb, window_shape=(4, 4), pooling_type='AVG', padding='VALID', data_format="NCHW")
        xb = tf.reshape(xb, (-1, 10))
        return xb


if __name__ == '__main__':
    a = 1
    b = 2
    c = tf.add(1, 2)
    print(c)

    path = untar_data(URLs.CIFAR)

    print(path)

    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=8).normalize(cifar_stats)

    opt_fn = tf.optimizers.Adam
    loss_fn = tf.losses.SparseCategoricalCrossentropy
    metrics = [categorical_accuracy]

    model = Simple_CNN()

    learn = TfLearner(data, model, opt_fn, loss_fn, metrics=metrics, true_wd=True, bn_wd=True, wd=defaults.wd,
                      train_bn=True)

    learn.lr_find()
    learn.recorder.plot()
