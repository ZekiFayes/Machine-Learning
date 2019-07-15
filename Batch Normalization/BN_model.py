from Hparams import *


def snn_train_model(x):

    a1 = tf.matmul(x, weights_snn['h1']) + biases_snn['b1']
    a1_mean, a1_var = tf.nn.moments(a1, 1)
    a1_mean_re = tf.transpose(tf.tile([a1_mean], (snn_param['hidden1'], 1)))
    a1_var_re = tf.transpose(tf.tile([a1_var], (snn_param['hidden1'], 1)))
    a1_ = (a1 - a1_mean_re) / tf.sqrt(a1_var_re + snn_param['eposilion'])
    a1_ = bn_snn['rb1'][0] * a1_ + bn_snn['rb1'][1]
    layer1 = tf.nn.sigmoid(a1_)

    a2 = tf.matmul(layer1, weights_snn['h2']) + biases_snn['b2']
    a2_mean, a2_var = tf.nn.moments(a2, 1)
    a2_mean_re = tf.transpose(tf.tile([a2_mean], (snn_param['hidden2'], 1)))
    a2_var_re = tf.transpose(tf.tile([a2_var], (snn_param['hidden2'], 1)))
    a2_ = (a2 - a2_mean_re) / tf.sqrt(a2_var_re + snn_param['eposilion'])
    a2_ = bn_snn['rb2'][0] * a2_ + bn_snn['rb2'][1]
    layer2 = tf.nn.sigmoid(a2_)

    a3 = tf.matmul(layer2, weights_snn['h3']) + biases_snn['b3']
    a3_mean, a3_var = tf.nn.moments(a3, 1)
    a3_mean_re = tf.transpose(tf.tile([a3_mean], (snn_param['hidden3'], 1)))
    a3_var_re = tf.transpose(tf.tile([a3_var], (snn_param['hidden3'], 1)))
    a3_ = (a3 - a3_mean_re) / tf.sqrt(a3_var_re + snn_param['eposilion'])
    a3_ = bn_snn['rb3'][0] * a3_ + bn_snn['rb3'][1]
    layer3 = tf.nn.sigmoid(a3_)

    out_layer = tf.matmul(layer3, weights_snn['out']) + biases_snn['out']

    return out_layer


def snn_cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=snn_param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def snn_evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def snn_bn_model_fn(x, y, mode):

    if mode == 'train':
        logits = snn_train_model(x)
        train_op, loss_op = snn_cost_fn(logits, y)
        prediction = tf.nn.softmax(logits)
        accuracy = snn_evaluation_fn(prediction, y)
        return train_op, loss_op, accuracy
    elif mode == 'test':
        logits = snn_train_model(x)
        prediction = tf.nn.softmax(logits)
        accuracy = snn_evaluation_fn(prediction, y)
        return accuracy
    else:
        logits = snn_train_model(x)
        return logits
