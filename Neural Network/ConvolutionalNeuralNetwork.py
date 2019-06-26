from Lib import *
from Evaluation import cnn_evaluation_fn


def conv2d(x, w, b, strides=1):

    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, dropout):

        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = conv2d(x, weights_cnn['wc1'], biases_cnn['bc1'])
        conv1 = maxpool2d(conv1, k=2)

        conv2 = conv2d(conv1, weights_cnn['wc2'], biases_cnn['bc2'])
        conv2 = maxpool2d(conv2, k=2)

        fc1 = tf.reshape(conv2, [-1, weights_cnn['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights_cnn['wd1']), biases_cnn['bd1'])
        fc1 = tf.nn.relu(fc1)

        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.add(tf.matmul(fc1, weights_cnn['out']), biases_cnn['out'])

        return out


def cnn_cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=cnn_param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def cnn_model_fn(x, y, dropout):

    logits = conv_net(x, dropout)
    prediction = tf.nn.softmax(logits)
    train_op, loss_op = cnn_cost_fn(logits, y)
    accuracy = cnn_evaluation_fn(prediction, y)

    return train_op, loss_op, accuracy
