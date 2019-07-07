import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


model_path = 'Model/VAE.ckpt'

param_vae = {
    'leanring_rate': 0.001,
    'num_steps': 10000,
    'batch_size': 128,
    'num_inputs': 784,
    'hidden1': 512,
    'latent': 2
}


def initialization(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


weights = {
    'encoder_h1': tf.Variable(initialization([param_vae['num_inputs'], param_vae['hidden1']])),
    'z_mean': tf.Variable(initialization([param_vae['hidden1'], param_vae['latent']])),
    'z_std': tf.Variable(initialization([param_vae['hidden1'], param_vae['latent']])),
    'decoder_h1': tf.Variable(initialization([param_vae['latent'], param_vae['hidden1']])),
    'decoder_out': tf.Variable(initialization([param_vae['hidden1'], param_vae['num_inputs']]))
}

biases = {
    'encoder_b1': tf.Variable(initialization([param_vae['hidden1']])),
    'z_mean': tf.Variable(initialization([param_vae['latent']])),
    'z_std': tf.Variable(initialization([param_vae['latent']])),
    'decoder_b1': tf.Variable(initialization([param_vae['hidden1']])),
    'decoder_out': tf.Variable(initialization([param_vae['num_inputs']]))
}
