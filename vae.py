# -*- coding: utf-8 -*-

from model import Encoder, Decoder

import numpy as np
import cv2
import tensorflow as tf
import time
import scipy.misc
import os
from numpy.random import *

import matplotlib.pyplot as plt

class VAE(object):
	def __init__(self, sess, batch_size, epoch, hidden_size, f_size=4):
		self.sess = sess
		self.batch_size = batch_size
		self.epoch = epoch
		self.hidden_size = hidden_size
		self.f_size = f_size
		self.enc = Encoder(f_size=self.f_size)
		self.dec = Decoder(f_size=self.f_size)
		self.build()

	def build(self):
		# z_p =  tf.random_normal((self.batch_size, self.hidden_size), 0, 1) # normal dist for GAN
		eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1) # normal dist for VAE

		# self.z = tf.placeholder(tf.float32, [None, self.hidden_size])
		self.input = tf.placeholder(tf.float32, [None, 28*28])
		self.x = tf.reshape(self.input, [-1, 28, 28, 1])

		# encoder 
		self.z_mean, self.z_sigma = self.enc(self.x)
		# print "***{0}".format(self.z_mean)
		# print "***{0}".format(self.z_sigma)
		# decoder
		# self.z_x = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_sigma)), eps))
		self.z_x = tf.add(self.z_mean, tf.mul(tf.exp(0.5 * self.z_sigma), eps))
		self.generated_x = self.dec(self.z_x)

		# losses

  		self.kl_loss = -0.5 * tf.reduce_mean(1 + self.z_sigma - tf.square(self.z_mean) - tf.exp(self.z_sigma), reduction_indices=1)
		
		# self.reconst_loss = -tf.reduce_mean(self.x * tf.log(1e-10 + self.generated_x) + (1-self.x) * tf.log(1e-10 + 1 - self.generated_x), 1)
		self.reconst_loss = tf.reduce_sum(tf.squared_difference(self.x, self.generated_x)) / (28 * 28 * 1)

		self.loss = tf.reduce_mean(self.kl_loss + self.reconst_loss)

		# for Tensorboard
		tf.scalar_summary("total_loss", self.loss)

		self.saver = tf.train.Saver()

	def read_image(self, paths):
			images = []
			for i, data in enumerate(paths):
				img = scipy.misc.imread(data).astype(np.float32)
				img = np.asarray(img) / 127.5 - 1.0
				images.append(img)

			return images

	def train(self, mnist):

		# train_input = self.read_image(train_path)
		train_input = mnist
	
		train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss, var_list=self.enc.variables + self.dec.variables)
	
		self.sess.run(tf.initialize_all_variables())
		#for Tensorboard
		summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter("log", graph_def=self.sess.graph_def)

		# for matplotlib
		kl_loss_list = []
		reconst_loss_list = []
		total_loss_list = []

		start_time = time.time()
		for step in range(self.epoch):
			for i in range(1000):
				batch = self.batch_size * i

				# train_input = self.read_image(train_path[batch : batch+self.batch_size])
				train_input, _ = mnist.train.next_batch(self.batch_size)
				train_input = np.reshape(train_input, (-1, 28, 28, 1))
				self.sess.run(train_step, feed_dict={self.x : train_input})

			rand = randint(500)
			# train_input = self.read_image(train_path[rand * self.batch_size : (rand + 1) * self.batch_size])
			train_input, _ = mnist.train.next_batch(self.batch_size)
			train_input = np.reshape(train_input, (-1, 28, 28, 1))

			print "epoch:{0}, time(sec):{1}".format(step, time.time() - start_time)
			# kl_loss = self.sess.run(self.kl_loss, feed_dict={self.x : train_input})
			# print "kl_loss : {0}".format(kl_loss)

			# reconst_loss = self.sess.run(self.reconst_loss, feed_dict={self.x : train_input})
			# print "reconst_loss : {0}".format(reconst_loss)

			total_loss = self.sess.run(self.loss, feed_dict={self.x : train_input})
			print "total_loss : {0}".format(total_loss)
				
			# kl_loss_list.append(kl_loss)
			# reconst_loss_list.append(reconst_loss)
			total_loss_list.append(total_loss)

			recon_z = self.sess.run(self.z_mean, feed_dict = {self.x : train_input})
			# print recon_z
			recon_x = self.sess.run(self.generated_x, {self.z_x: recon_z})
			# print recon_x
			gen_images = self.make88_data_image(recon_x)
			cv2.imwrite(os.path.join("generated", "{0}_epoch.jpg".format(step)), gen_images[:, :, ::-1].copy())
			# with open(os.path.join("generated", "{0}_epoch.jpg".format(step)), 'wb') as f:
			# 	f.write(gen_images)

			inp = self.make88_data_image(train_input)
			# scipy.misc.imsave(os.path.join("generated", "{0}_input.jpg".format(step)), inp)
			# scipy.misc.imsave(os.path.join("generated", "{0}_answer.jpg".format(step)), ans)
			# cv2.imwrite(os.path.join("generated", "{0}_answer.jpg".format(step)), ans[:, :, ::-1].copy())
			# cv2.imwrite(os.path.join("generated", "{0}_input.jpg".format(step)), inp[:, :, ::-1].copy())
			# cv2.imwrite(os.path.join("generated", "{0}_answer.jpg".format(step)), ans[:, :, ::-1].copy())
			cv2.imwrite(os.path.join("generated", "{0}_input.jpg".format(step)), inp[:, :, ::-1].copy())
			print "{0}_save image".format(step)

			#for tensorboard
			summary_str = self.sess.run(summary_op, feed_dict={self.x : train_input})
			summary_writer.add_summary(summary_str, step)
			summary_writer.flush()
			if step % 10 == 0:
				self.saver.save(self.sess, "model{0}.ckpt".format(step))
			else:
				self.saver.save(self.sess, "model.ckpt")

		# plt.plot(kl_loss_list, label = 'kl_loss')
		# plt.plot(reconst_loss_list, label = 'reconst_loss')
		plt.plot(total_loss_list, label = 'total_loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.savefig("loss.jpg")
		print "total time(sec): {0}".format(time.time() - start_time)
		print "total time(min): {0}".format((time.time() - start_time) / 60)
		print "total time(hour): {0}".format((time.time() - start_time) / 60 / 60)

	def make88_data_image(self, input, row=8, col=8):
		input_image = [0 for l in range(self.batch_size)]
		for i in range(64):
			input_image[i] = input[i] + 1.0
			input_image[i] *= 127.5
		row_img = []
		for j in range(row):
			for i in range(col - 1):
				if i == 0:
					# horizontally : axis = 0
					img = np.concatenate((input_image[0 + 8 * j], input_image[1 + 8 * j]), axis=1)
				else:
					img = np.concatenate((img, input_image[i + 1 + j * 8]), axis=1)
					if i == 6:
						row_img.append(img)

		for k in range(row - 1):
			if k == 0:
				out = np.concatenate((row_img[0], row_img[1]), axis=0)
			else:
				out = np.concatenate((out, row_img[k + 1]), axis=0)
				if k == 6:
					out = np.uint8(out)
		return out

	def make88_gen_images(self, images, row=8, col=8):
		images = [image for image in tf.split(0, self.batch_size, images)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
		image = tf.concat(1, rows)

		return tf.image.encode_jpeg(tf.squeeze(image, [0]))

	def save_gen_images(self, inputs):
		generated = self.dec(inputs)[-1]
		reconstructed = self.g2(generated, minus_vector_batch)[-1]
		gen_images = tf.cast(tf.mul(tf.add(generated, 1.0), 127.5), tf.uint8)
		re_images = tf.cast(tf.mul(tf.add(reconstructed, 1.0), 127.5), tf.uint8)

		return self.make88_gen_images(gen_images), self.make88_gen_images(re_images)

	def test(self, i, test_inputs_path):
		# test_input = self.read_image(test_inputs_path)
		test_input = test_inputs_path
		test_input = np.reshape(test_input, (-1, 28, 28, 1))

		# kl_loss = self.sess.run(self.kl_loss, feed_dict={self.x : test_input})
		# print "kl_loss : {0}".format(kl_loss)
		
		# reconst_loss = self.sess.run(self.reconst_loss, feed_dict={self.x : test_input})
		# print "reconst_loss : {0}".format(reconst_loss)

		total_loss = self.sess.run(self.loss, feed_dict={self.x : test_input})
		print "total_loss : {0}".format(total_loss)

		recon_z = self.sess.run(self.z_mean, feed_dict = {self.x : test_input})
		# print recon_z
		recon_x = self.sess.run(self.generated_x, {self.z_x: recon_z})
		# print recon_x
		# recon_x = tf.cast(tf.mul(tf.add(recon_x, 1.0), 127.5), tf.uint8)
		inp = self.make88_data_image(test_input)
		cv2.imwrite(os.path.join("generated", "input_{0}.jpg".format(i)), inp[:, :, ::-1].copy())
		gen_images = self.make88_data_image(recon_x)
		cv2.imwrite(os.path.join("generated", "test_{0}.jpg".format(i)), gen_images[:, :, ::-1].copy())

	# def feature(self, input_paths):
	# 	input_images = self.read_image(input_paths)
	# 	out = self.sess.run(self.z_x, feed_dict = {self.x : input_images})
	# 	return out

	def load(self, checkpoint_dir):
		self.saver.restore(self.sess, checkpoint_dir)