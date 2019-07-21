import tensorflow as tf


param = {
    'learning_rate': 0.001,
    'batch_size': 100,
    'num_steps': 10000,
    'display_step': 100,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 500,
    'hidden2': 500,
    'weight_cost': 0.0002,
    'initial_mu': 0.1,
    'final_mu': 0.3
}


weights = {
    'h1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden1']])),
    'dh1': tf.Variable(tf.zeros([param['num_inputs'], param['hidden1']])),
    'h2': tf.Variable(tf.random_normal([param['hidden1'], param['hidden2']])),
    'dh2': tf.Variable(tf.zeros([param['hidden1'], param['hidden2']])),
    'out': tf.Variable(tf.random_normal([param['hidden2'], param['num_classes']])),
    'do': tf.Variable(tf.random_normal([param['hidden2'], param['num_classes']]))
}


biases = {
    'b1': tf.Variable(tf.random_normal([param['hidden1']])),
    'db1': tf.Variable(tf.zeros([param['hidden1']])),
    'b2': tf.Variable(tf.random_normal([param['hidden2']])),
    'db2': tf.Variable(tf.zeros([param['hidden2']])),
    'out': tf.Variable(tf.random_normal([param['num_classes']])),
    'do': tf.Variable(tf.zeros([param['num_classes']])),
}
