'''
Singer Voice Separator RNN

Lei Mao
University of Chicago
'''

import tensorflow.compat.v1 as tf
import numpy as np 
import os
import shutil
from datetime import datetime


class SVSRNN(object):

    def __init__(self, num_features, num_rnn_layer = 3, num_hidden_units = [256, 256, 256]):

        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        # The shape of x_mixed, y_src1, y_src2 are [batch_size, n_frames (time), n_frequencies]
        self.x_mixed = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src2')

        self.y_pred_src1, self.y_pred_src2 = self.network_initializer()

        self.gamma = 0.001
        self.loss = self.loss_initializer()
        self.optimizer = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def network(self):

        rnn_layer = [tf.nn.rnn_cell.GRUCell(size) for size in self.num_hidden_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_mixed, dtype = tf.float32)
        y_hat_src1 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src1')
        y_hat_src2 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src2')
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        # Mask with abs
        #y_tilde_src1 = tf.abs(y_hat_src1) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        #y_tilde_src2 = tf.abs(y_hat_src2) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        return y_tilde_src1, y_tilde_src2
        #return y_hat_src1, y_hat_src2
 
    def network_initializer(self):

        with tf.variable_scope('rnn_network') as scope:
            y_pred_src1, y_pred_src2 = self.network()

        return y_pred_src1, y_pred_src2


    def generalized_kl_divergence(self, y, y_hat):

        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)


    def loss_initializer(self):

        with tf.variable_scope('loss') as scope:

            # Mean Squared Error Loss
            loss = tf.reduce_mean(tf.square(self.y_src1 - self.y_pred_src1) + tf.square(self.y_src2 - self.y_pred_src2), name = 'loss')

        return loss

    def optimizer_initializer(self):

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss, global_step = self.gstep)

        return optimizer

    def train(self, x, y1, y2, learning_rate):

        #step = self.gstep.eval()

        step = self.sess.run(self.gstep)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss], 
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2, self.learning_rate: learning_rate})
        return train_loss

    def validate(self, x, y1, y2):

        y1_pred, y2_pred, validate_loss = self.sess.run([self.y_pred_src1, self.y_pred_src2, self.loss], 
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2})
        return y1_pred, y2_pred, validate_loss

    def test(self, x):

        y1_pred, y2_pred = self.sess.run([self.y_pred_src1, self.y_pred_src2], feed_dict = {self.x_mixed: x})

        return y1_pred, y2_pred

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


