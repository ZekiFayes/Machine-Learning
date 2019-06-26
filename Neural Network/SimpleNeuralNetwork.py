from Lib import *
from Evaluation import snn_evaluation_fn


def simple_neural_net_naf(x):

    layer_1 = tf.add(tf.matmul(x, weights_snn['h1']), biases_snn['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights_snn['h2']), biases_snn['b2'])
    out_layer = tf.matmul(layer_2, weights_snn['out']) + biases_snn['out']

    return out_layer


def simple_neural_net_af(x):

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights_snn['h1']), biases_snn['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights_snn['h2']), biases_snn['b2'])
    out_layer = tf.matmul(layer_2, weights_snn['out']) + biases_snn['out']

    return out_layer


def snn_cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=snn_param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def snn_model_fn(x, y):

    logits = simple_neural_net_af(x)
    prediction = tf.nn.softmax(logits)
    train_op, loss_op = snn_cost_fn(logits, y)
    accuracy = snn_evaluation_fn(prediction, y)

    return train_op, loss_op, accuracy
