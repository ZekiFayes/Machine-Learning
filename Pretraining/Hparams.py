import tensorflow as tf

sm_param = {
    'learning_rate': 0.01,
    'num_steps': 25000,
    'batch_size': 60,
    'display_step': 1000,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 200,
    'hidden2': 200,
}

weights_sm = {
    'h1': tf.Variable(tf.random_normal([sm_param['num_inputs'], sm_param['hidden1']])),
    'h11': tf.Variable(tf.random_normal([sm_param['hidden1'], sm_param['num_inputs']])),

    'h2': tf.Variable(tf.random_normal([sm_param['hidden1'], sm_param['hidden2']])),
    'h22': tf.Variable(tf.random_normal([sm_param['hidden2'], sm_param['hidden1']])),

    'out': tf.Variable(tf.random_normal([sm_param['hidden2'], sm_param['num_classes']])),
    'out1': tf.Variable(tf.random_normal([sm_param['num_classes'], sm_param['hidden2']]))
}

biases_sm = {
    'b1': tf.Variable(tf.random_normal([sm_param['hidden1']])),
    'b11': tf.Variable(tf.random_normal([sm_param['num_inputs']])),

    'b2': tf.Variable(tf.random_normal([sm_param['hidden2']])),
    'b22': tf.Variable(tf.random_normal([sm_param['hidden1']])),

    'out': tf.Variable(tf.random_normal([sm_param['num_classes']])),
    'out1': tf.Variable(tf.random_normal([sm_param['hidden2']]))
}
