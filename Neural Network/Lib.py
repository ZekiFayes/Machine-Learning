import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os


default_mode = 'SNN'


""" Parameters of Simple Neural Network """
snn_param = {
    'learning_rate': 0.01,
    'num_steps': 1000,
    'batch_size': 128,
    'display_step': 100,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 256,
    'hidden2': 64
}

weights_snn = {
    'h1': tf.Variable(tf.random_normal([snn_param['num_inputs'], snn_param['hidden1']])),
    'h2': tf.Variable(tf.random_normal([snn_param['hidden1'], snn_param['hidden2']])),
    'out': tf.Variable(tf.random_normal([snn_param['hidden2'], snn_param['num_classes']]))
}

biases_snn = {
    'b1': tf.Variable(tf.random_normal([snn_param['hidden1']])),
    'b2': tf.Variable(tf.random_normal([snn_param['hidden2']])),
    'out': tf.Variable(tf.random_normal([snn_param['num_classes']]))
}


""" Parameters of Convolutional Neural Network """
cnn_param = {
    'learning_rate': 0.01,
    'num_steps': 1000,
    'batch_size': 128,
    'display_step': 100,
    'num_inputs': 784,
    'channel': 1,
    'num_classes': 10,
    'filter1': 5,
    'hidden1': 32,
    'filter2': 5,
    'hidden2': 64,
    'fc1': 3136,
    'fc2': 1024,
    'dropout': 0.75
}

weights_cnn = {
    'wc1': tf.Variable(tf.random_normal([cnn_param['filter1'], cnn_param['filter1'],
                                         cnn_param['channel'], cnn_param['hidden1']])),
    'wc2': tf.Variable(tf.random_normal([cnn_param['filter2'], cnn_param['filter2'],
                                         cnn_param['hidden1'], cnn_param['hidden2']])),
    'wd1': tf.Variable(tf.random_normal([cnn_param['fc1'], cnn_param['fc2']])),
    'out': tf.Variable(tf.random_normal([cnn_param['fc2'], cnn_param['num_classes']]))
}

biases_cnn = {
    'bc1': tf.Variable(tf.random_normal([cnn_param['hidden1']])),
    'bc2': tf.Variable(tf.random_normal([cnn_param['hidden2']])),
    'bd1': tf.Variable(tf.random_normal([cnn_param['fc2']])),
    'out': tf.Variable(tf.random_normal([cnn_param['num_classes']]))
}

""" Parameters of AutoEncoder """
ae_param = {
    'learning_rate': 0.04,
    'num_steps': 3000,
    'batch_size': 256,
    'display_step': 100,
    'num_inputs': 784,
    'examples_to_show': 10,
    'hidden1': 256,
    'hidden2': 128,
}

weights_ae = {
    'encoder_h1': tf.Variable(tf.random_normal([ae_param['num_inputs'], ae_param['hidden1']])),
    'encoder_h2': tf.Variable(tf.random_normal([ae_param['hidden1'], ae_param['hidden2']])),
    'decoder_h1': tf.Variable(tf.random_normal([ae_param['hidden2'], ae_param['hidden1']])),
    'decoder_h2': tf.Variable(tf.random_normal([ae_param['hidden1'], ae_param['num_inputs']])),
}

biases_ae = {
    'encoder_b1': tf.Variable(tf.random_normal([ae_param['hidden1']])),
    'encoder_b2': tf.Variable(tf.random_normal([ae_param['hidden2']])),
    'decoder_b1': tf.Variable(tf.random_normal([ae_param['hidden1']])),
    'decoder_b2': tf.Variable(tf.random_normal([ae_param['num_inputs']])),
}
