import tensorflow as tf


# parameters for complicated model
cm_param = {
    'learning_rate': 0.01,
    'num_steps': 20000,
    'batch_size': 60,
    'display_step': 200,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 512,
    'hidden2': 256,
    'hidden3': 128,
    'eposilion': 1e-6
}

weights_cm = {
    'h1': tf.Variable(tf.random_normal([cm_param['num_inputs'], cm_param['hidden1']])),
    'h2': tf.Variable(tf.random_normal([cm_param['hidden1'], cm_param['hidden2']])),
    'h3': tf.Variable(tf.random_normal([cm_param['hidden2'], cm_param['hidden3']])),
    'out': tf.Variable(tf.random_normal([cm_param['hidden3'], cm_param['num_classes']]))
}

biases_cm = {
    'b1': tf.Variable(tf.random_normal([cm_param['hidden1']])),
    'b2': tf.Variable(tf.random_normal([cm_param['hidden2']])),
    'b3': tf.Variable(tf.random_normal([cm_param['hidden3']])),
    'out': tf.Variable(tf.random_normal([cm_param['num_classes']]))
}

bn_cm = {
    'rb1': tf.Variable(tf.random_normal([2, 1])),
    'rb2': tf.Variable(tf.random_normal([2, 1])),
    'rb3': tf.Variable(tf.random_normal([2, 1]))
}


# parameters for simple model

sm_param = {
    'learning_rate': 0.01,
    'num_steps': 5000,
    'batch_size': 60,
    'display_step': 1000,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 100,
    'hidden2': 100,
}

weights_sm = {
    'h1': tf.Variable(tf.random_normal([sm_param['num_inputs'], sm_param['hidden1']])),
    'h2': tf.Variable(tf.random_normal([sm_param['hidden1'], sm_param['hidden2']])),
    'out': tf.Variable(tf.random_normal([sm_param['hidden2'], sm_param['num_classes']]))
}

biases_sm = {
    'b1': tf.Variable(tf.random_normal([sm_param['hidden1']])),
    'b2': tf.Variable(tf.random_normal([sm_param['hidden2']])),
    'out': tf.Variable(tf.random_normal([sm_param['num_classes']]))
}
