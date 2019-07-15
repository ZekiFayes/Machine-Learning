import tensorflow as tf

snn_param = {
    'learning_rate': 0.01,
    'num_steps': 5000,
    'batch_size': 60,
    'display_step': 100,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden1': 100,
    'hidden2': 100,
    'hidden3': 100,
    'eposilion': 1e-6
}

weights_snn = {
    'h1': tf.Variable(tf.random_normal([snn_param['num_inputs'], snn_param['hidden1']])),
    'h2': tf.Variable(tf.random_normal([snn_param['hidden1'], snn_param['hidden2']])),
    'h3': tf.Variable(tf.random_normal([snn_param['hidden2'], snn_param['hidden3']])),
    'out': tf.Variable(tf.random_normal([snn_param['hidden3'], snn_param['num_classes']]))
}

biases_snn = {
    'b1': tf.Variable(tf.random_normal([snn_param['hidden1']])),
    'b2': tf.Variable(tf.random_normal([snn_param['hidden2']])),
    'b3': tf.Variable(tf.random_normal([snn_param['hidden3']])),
    'out': tf.Variable(tf.random_normal([snn_param['num_classes']]))
}

bn_snn = {
    'rb1': tf.Variable(tf.random_normal([2, 1])),
    'rb2': tf.Variable(tf.random_normal([2, 1])),
    'rb3': tf.Variable(tf.random_normal([2, 1]))
}
