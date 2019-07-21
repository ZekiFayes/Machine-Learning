from Lib import *
from Hparams import *


def binomial(dim, rate):
    u = tf.random_uniform(shape=dim, minval=0, maxval=1)
    return tf.where(u > rate, tf.ones(dim), tf.zeros(dim))


def bpblock(dz, h):
    dim0 = tf.shape(dz)
    dim1 = tf.shape(h)

    dz_re = tf.reshape(dz, [dim0[0], 1, dim0[1]])
    dz_re = tf.tile(dz_re, [1, dim1[1], 1])

    h_re = tf.reshape(h, [dim1[0], 1, dim1[1]])
    h_re = tf.tile(h_re, [1, dim0[1], 1])
    h_re = tf.transpose(h_re, (0, 2, 1))

    dw = tf.reduce_mean(dz_re * h_re, 0)
    db = tf.reduce_mean(dz, 0)
    return dw, db


def fnn_with_dropout_fn(x, y, epoch, mode):
    # forward pass
    r1 = binomial(dim=[param['num_inputs']], rate=0.1)
    h0_ = x * r1
    z1 = tf.matmul(h0_, weights['h1']) + biases['b1']
    h1 = tf.nn.sigmoid(z1)

    r2 = binomial(dim=[param['hidden1']], rate=0.2)
    h1_ = h1 * r2
    z2 = tf.matmul(h1_, weights['h2']) + biases['b2']
    h2 = tf.nn.sigmoid(z2)

    r3 = binomial(dim=[param['hidden2']], rate=0.2)
    h2_ = h2 * r3
    z3 = tf.matmul(h2_, weights['out']) + biases['out']
    y_ = tf.nn.softmax(z3)

    if mode == 'test':
        accuracy = evaluation_fn(y_, y)
        return accuracy

    loss_op = - tf.reduce_mean(y * tf.log(y_))

    # backward pass
    mu = tf.where(epoch > 5000, [param['final_mu']], [param['initial_mu']])

    y1 = tf.reshape(y, [param['batch_size'], 1, param['num_classes']])
    y1 = tf.tile(y1, [1, param['num_classes'], 1])

    y1_ = tf.reshape(y_, [param['batch_size'], 1, param['num_classes']])
    y1_ = tf.tile(y1_, [1, param['num_classes'], 1])
    y1_ = tf.transpose(y1_, (0, 2, 1))

    dz3 = tf.reduce_sum(y1 * (y1_ - tf.matrix_diag(tf.ones(param['num_classes']))), 2)

    dw, db = bpblock(dz3, h2_)

    w = mu * weights['do'] + param['learning_rate'] * (dw - weights['out'] * param['weight_cost'])
    b = mu * biases['do'] + param['learning_rate'] * db
    weights['do'] = weights['do'].assign(w)
    biases['do'] = biases['do'].assign(b)
    weights['out'] = weights['out'].assign(weights['out'] - w)
    biases['out'] = biases['out'].assign(biases['out'] - b)

    dh2 = tf.matmul(dz3, tf.transpose(weights['out'])) * r3
    dz2 = dh2 * h2 * (1 - h2)

    dw, db = bpblock(dz2, h1_)
    w = mu * weights['dh2'] + param['learning_rate'] * (dw - weights['h2'] * param['weight_cost'])
    b = mu * biases['db2'] + param['learning_rate'] * db
    weights['dh2'] = weights['dh2'].assign(w)
    biases['db2'] = biases['db2'].assign(b)
    weights['h2'] = weights['h2'].assign(weights['h2'] - w)
    biases['b2'] = biases['b2'].assign(biases['b2'] - b)

    dh1 = tf.matmul(dz2, tf.transpose(weights['h2'])) * r2
    dz1 = dh1 * h1 * (1 - h1)

    dw, db = bpblock(dz1, h0_)

    w = mu * weights['dh1'] + param['learning_rate'] * (dw - weights['h1'] * param['weight_cost'])
    b = mu * biases['db1'] + param['learning_rate'] * db
    weights['dh1'] = weights['dh1'].assign(w)
    biases['db1'] = biases['db1'].assign(b)
    weights['h1'] = weights['h1'].assign(weights['h1'] - w)
    biases['b1'] = biases['b1'].assign(biases['b1'] - b)

    gd = dz3

    return gd, loss_op


def evaluation_fn(prediction, y):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def model_fn(x, y, epoch, mode):
    if mode == 'train':
        gd, loss_op = fnn_with_dropout_fn(x, y, epoch, mode)
        return gd, loss_op

    elif mode == 'test':
        accuracy = fnn_with_dropout_fn(x, y, epoch, mode)
        return accuracy
