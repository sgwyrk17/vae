# -*- coding: utf-8 -*-
from vae import VAE

import input_data

import os
import numpy as np
import scipy.misc
import re
from glob import glob
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("hidden_size", 512, "hidden_size")
flags.DEFINE_integer("epoch", 50, "epoch")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main():
	if not os.path.exists("generated"):
		os.makedirs("generated")

	print "--- load START---"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	print "--- load FINISH ---"

	with tf.Session() as sess:
		vae = VAE(sess, batch_size=FLAGS.batch_size, epoch = FLAGS.epoch, hidden_size = FLAGS.hidden_size)

		if FLAGS.train:
			vae.train(mnist)
		else:
			vae.load("model.ckpt")
			test = mnist.test.images
			# batch_num = len(test) / FLAGS.batch_size
			i = 0
			for i in range(1):
				vae.test(i, test[i*FLAGS.batch_size : i*FLAGS.batch_size+FLAGS.batch_size])

if __name__ == '__main__':
	main()