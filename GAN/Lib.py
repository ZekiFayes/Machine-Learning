import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


model_path = 'Model/GAN.ckpt'
save_path = 'Model/'

param_gan = {
    'learning_rate': 0.0001,
    'num_steps': 30000,
    'batch_size': 128,
    'num_inputs': 784,
    'gen_hidden': 256,
    'disc_hidden': 256,
    'num_noise': 100,
}


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


weights = {
    'gen_hidden1': tf.Variable(glorot_init([param_gan['num_noise'], param_gan['gen_hidden']])),
    'gen_out': tf.Variable(glorot_init([param_gan['gen_hidden'], param_gan['num_inputs']])),
    'disc_hidden1': tf.Variable(glorot_init([param_gan['num_inputs'], param_gan['disc_hidden']])),
    'disc_out': tf.Variable(glorot_init([param_gan['disc_hidden'], 1])),
}

biases = {
    'gen_hidden1': tf.Variable(tf.zeros([param_gan['gen_hidden']])),
    'gen_out': tf.Variable(tf.zeros([param_gan['num_inputs']])),
    'disc_hidden1': tf.Variable(tf.zeros([param_gan['disc_hidden']])),
    'disc_out': tf.Variable(tf.zeros([1])),
}
