import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


model_path = 'Model/DCGAN.ckpt'
save_path = 'Model/'

param_dcgan = {
    'learning_rate': 0.0002,
    'batch_size': 128,
    'num_noise': 200,
    'num_steps': 1000
}
