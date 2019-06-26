from Lib import *


def encoder(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_ae['encoder_h1']),
                                   biases_ae['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_ae['encoder_h2']),
                                   biases_ae['encoder_b2']))
    return layer_2


def decoder(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_ae['decoder_h1']),
                                   biases_ae['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_ae['decoder_h2']),
                                   biases_ae['decoder_b2']))

    return layer_2


def ae_cost_fn(y_true, y_pred):

    loss_op = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(ae_param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def ae_model_fn(x):

    encoder_op = encoder(x)
    y_true, y_pred = x, decoder(encoder_op)
    train_op, loss_op = ae_cost_fn(y_true, y_pred)

    return train_op, loss_op, y_pred
