# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
import time
import scipy.misc
import os
from numpy.random import *

class Encoder:
    def __init__(self, f_size=4):
        self.f_size = f_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        outputs = tf.convert_to_tensor(inputs)
        out = []
        with tf.variable_scope('enc', reuse=self.reuse):
            # convolution x 3
            with tf.variable_scope('enc_conv1'):
                w = tf.get_variable('w', [5, 5, 1, 64], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [64], tf.float32, tf.zeros_initializer)
                c = tf.nn.conv2d(outputs, w, [1, 2, 2, 1], 'SAME')
                mean, variance = tf.nn.moments(c, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(c, mean, variance, b, None, 1e-5))
                # print outputs
                out.append(outputs)
            with tf.variable_scope('enc_conv2'):
                w = tf.get_variable('w', [5, 5, 64, 128], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [128], tf.float32, tf.zeros_initializer)
                c = tf.nn.conv2d(outputs, w, [1, 2, 2, 1], 'SAME')
                mean, variance = tf.nn.moments(c, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(c, mean, variance, b, None, 1e-5))
                # print outputs
                out.append(outputs)
            with tf.variable_scope('fully'):
                dim = 1
                for d in outputs.get_shape()[1:].as_list():
                    dim *= d
                print dim
                w = tf.get_variable('w', [dim, 512], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [512], tf.float32, tf.zeros_initializer)
                c = tf.matmul(tf.reshape(outputs, [-1, dim]), w)
                mean, variance = tf.nn.moments(c, [0])
                outputs = tf.nn.relu(tf.nn.batch_normalization(c, mean, variance, b, None, 1e-5))
                # print outputs
                out.append(outputs)
                outputs_sigma = tf.nn.relu(tf.nn.batch_normalization(c, mean, variance, b, None, 1e-5))
                # print outputs
                out.append(outputs)
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='enc')
        return outputs, outputs_sigma

class Decoder:
    def __init__(self, f_size=4):
        self.f_size = f_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        out = []
        with tf.variable_scope('dec', reuse=self.reuse):
            with tf.variable_scope('fully'):
                w0 = tf.get_variable('w', [inputs.get_shape()[-1], 128*7*7], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b0 = tf.get_variable('b', [128], tf.float32, tf.zeros_initializer)
                fc = tf.matmul(inputs, w0)
                reshaped = tf.reshape(fc, [-1, 7, 7, 128])
                mean, variance = tf.nn.moments(reshaped, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(reshaped, mean, variance, b0, None, 1e-5))
                out.append(outputs)
                # print outputs
            # deconvolution x 3
            with tf.variable_scope('dec_deconv1'):
                w = tf.get_variable('w', [5, 5, 64, 128], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [64], tf.float32, tf.zeros_initializer)
                dc = tf.nn.conv2d_transpose(outputs, w, [int(outputs.get_shape()[0]), 7 * 2, 7 * 2 , 64], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, b, None, 1e-5))
                out.append(outputs)
                # print outputs
            with tf.variable_scope('dec_deconv2ssss'):
                w = tf.get_variable('w', [5, 5, 1, 64], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer)
                dc = tf.nn.conv2d_transpose(outputs, w, [int(outputs.get_shape()[0]), 7 * 2 * 2, 7 * 2 * 2, 1], [1, 2, 2, 1])
                outputs = tf.nn.tanh(tf.nn.bias_add(dc, b))
                out.append(outputs)
                # print outputs
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec')
        return outputs