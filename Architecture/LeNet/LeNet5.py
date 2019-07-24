from Hparams import *


def cnn_block(input_x, n, mode):

    l = 'l' + str(n)
    if mode == 'conv':
        layer = tf.nn.relu(tf.nn.conv2d(input_x, weights[l], strides=[1, 1, 1, 1], padding='SAME') + biases[l])
        return layer
    elif mode == 'pool':
        layer = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return layer
    elif mode == 'fc':
        layer = tf.nn.relu(tf.matmul(input_x, weights[l]) + biases[l])
        return layer
    elif mode == 'dense':
        layer = tf.layers.flatten(input_x)
        return layer
    elif mode == 'out':
        layer = tf.matmul(input_x, weights[l]) + biases[l]
        return layer


def LeNet(x, mode):

    conv1 = cnn_block(x, 1, 'conv')
    pool1 = cnn_block(conv1, 1, 'pool')
    conv2 = cnn_block(pool1, 2, 'conv')
    pool2 = cnn_block(conv2, 2, 'pool')
    dense = cnn_block(pool2, 2, 'dense')
    fc1 = cnn_block(dense, 3, 'fc')
    if mode == 'train':
        fc1 = tf.nn.dropout(fc1, rate=0.5)
    logits = cnn_block(fc1, 4, 'out')

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

    logits = LeNet(x, mode)
    prediction = tf.nn.softmax(logits)
    accuracy = evaluation_fn(prediction, y)

    if mode == 'train':
        train_op, loss_op = cost_fn(logits, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        return accuracy
