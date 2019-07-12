import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

param_rbm = {
    'learning_rate': 0.1,
    'batch_size': 1,
    'num_epochs': 5,
    'num_v_nodes': 3952*5,
    'num_h_nodes': 200,
    'initial_mu': 0.5,
    'final_mu': 0.9,
    'weight_cost': 0.0002,
    'cd_k': 10,
}

wb = {
    'w': tf.random_normal([param_rbm['num_v_nodes'], param_rbm['num_h_nodes']]),
    'v': tf.random_normal([param_rbm['num_v_nodes']]),
    'h': tf.random_normal([param_rbm['num_h_nodes']]),

    'w_inc': tf.zeros([param_rbm['num_v_nodes'], param_rbm['num_h_nodes']]),
    'v_inc': tf.zeros([param_rbm['num_v_nodes']]),
    'h_inc': tf.zeros([param_rbm['num_h_nodes']]),
}
