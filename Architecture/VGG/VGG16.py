from Hparams import *
from Lib import *


def vgg_block(input_x, n, mode):

    l = 'l' + str(n)
    if mode == 'conv':
        layer = tf.nn.relu(tf.nn.conv2d(input_x, weights[l], strides=[1, 1, 1, 1],
                                        padding='SAME') + biases[l])
        return layer
    elif mode == 'pool':
        layer = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')
        return layer
    elif mode == 'dense':
        layer = tf.layers.flatten(input_x)
        return layer
    elif mode == 'fc':
        layer = tf.nn.relu(tf.matmul(input_x, weights[l]) + biases[l])
        return layer
    elif mode == 'out':
        layer = tf.nn.softmax(tf.matmul(input_x, weights[l]) + biases[l])
        return layer


def vgg(x, mode):

    conv1 = vgg_block(x, 1, 'conv')
    conv2 = vgg_block(conv1, 2, 'conv')
    pool1 = vgg_block(conv2, 2, 'pool')

    conv3 = vgg_block(pool1, 3, 'conv')
    conv4 = vgg_block(conv3, 4, 'conv')
    pool2 = vgg_block(conv4, 4, 'pool')

    conv5 = vgg_block(pool2, 5, 'conv')
    conv6 = vgg_block(conv5, 6, 'conv')
    conv7 = vgg_block(conv6, 7, 'conv')
    pool3 = vgg_block(conv7, 7, 'pool')

    conv8 = vgg_block(pool3, 8, 'conv')
    conv9 = vgg_block(conv8, 9, 'conv')
    conv10 = vgg_block(conv9, 10, 'conv')
    pool4 = vgg_block(conv10, 10, 'pool')

    conv11 = vgg_block(pool4, 11, 'conv')
    conv12 = vgg_block(conv11, 12, 'conv')
    conv13 = vgg_block(conv12, 13, 'conv')
    pool5 = vgg_block(conv13, 13, 'pool')
    dense = vgg_block(pool5, 13, 'dense')

    fc1 = vgg_block(dense, 14, 'fc')
    fc2 = vgg_block(fc1, 15, 'fc')
    logits = vgg_block(fc2, 16, 'out')

    return logits


def evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=param['lr'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def model_fn(x, y, mode):

    logits = vgg(x, mode)
    prediction = tf.nn.softmax(logits)
    accuracy = evaluation_fn(prediction, y)

    if mode == 'train':
        train_op, loss_op = cost_fn(logits, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        return accuracy

