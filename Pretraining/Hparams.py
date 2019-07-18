import tensorflow as tf


sm_param = {
    'learning_rate': 0.01,
    'num_steps': 25000,
    'batch_size': 100,
    'display_step': 600,
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

# pretrain with RBM

param_rbm = {
    'learning_rate': 0.1,
    'initial_mu': 0.5,
    'final_mu': 0.9,
    'batch_size': 100,
    'weight_cost': 0.0002,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 200,
    'hidden2': 200,
    'restart': tf.ones([1])
}

weights_rbm = {
    'w1': tf.Variable(tf.random_normal([param_rbm['num_inputs'], param_rbm['hidden1']], mean=0, stddev=0.01)),
    'v1': tf.Variable(tf.random_normal([param_rbm['num_inputs']], mean=0, stddev=0.01)),
    'h1': tf.Variable(tf.random_normal([param_rbm['hidden1']], mean=0, stddev=0.01)),

    'dw1': tf.Variable(tf.zeros([param_rbm['num_inputs'], param_rbm['hidden1']])),
    'dv1': tf.Variable(tf.zeros([param_rbm['num_inputs']])),
    'dh1': tf.Variable(tf.zeros([param_rbm['hidden1']])),

    'w2': tf.Variable(tf.random_normal([param_rbm['hidden1'], param_rbm['hidden2']], mean=0, stddev=0.01)),
    'v2': tf.Variable(tf.random_normal([param_rbm['hidden1']], mean=0, stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([param_rbm['hidden2']], mean=0, stddev=0.01)),

    'dw2': tf.Variable(tf.zeros([param_rbm['hidden1'], param_rbm['hidden2']])),
    'dv2': tf.Variable(tf.zeros([param_rbm['hidden1']])),
    'dh2': tf.Variable(tf.zeros([param_rbm['hidden2']])),

    'w3': tf.Variable(tf.random_normal([param_rbm['hidden2'], param_rbm['num_classes']], mean=0, stddev=0.01)),
    'v3': tf.Variable(tf.random_normal([param_rbm['hidden2']], mean=0, stddev=0.01)),
    'h3': tf.Variable(tf.random_normal([param_rbm['num_classes']], mean=0, stddev=0.01)),

    'dw3': tf.Variable(tf.zeros([param_rbm['hidden2'], param_rbm['num_classes']])),
    'dv3': tf.Variable(tf.zeros([param_rbm['hidden2']])),
    'dh3': tf.Variable(tf.zeros([param_rbm['num_classes']]))
}
