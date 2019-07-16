from Hparams import *


# Complicated Model
def bn_fn(x):

    dim0 = tf.shape(x)
    dim = dim0[1]
    x_mean, x_var = tf.nn.moments(x, 1)
    x_mean_re = tf.transpose(tf.tile([x_mean], (dim, 1)))
    x_var_re = tf.transpose(tf.tile([x_var], (dim, 1)))
    x_ = (x - x_mean_re) / tf.sqrt(x_var_re + cm_param['eposilion'])

    return x_


def cm_model_fn(x):

    a1 = tf.matmul(x, weights_cm['h1']) + biases_cm['b1']
    a1_ = bn_fn(a1)
    a1_ = bn_cm['rb1'][0] * a1_ + bn_cm['rb1'][1]
    layer1 = tf.nn.sigmoid(a1_)

    a2 = tf.matmul(layer1, weights_cm['h2']) + biases_cm['b2']
    a2_ = bn_fn(a2)
    a2_ = bn_cm['rb2'][0] * a2_ + bn_cm['rb2'][1]
    layer2 = tf.nn.sigmoid(a2_)

    a3 = tf.matmul(layer2, weights_cm['h3']) + biases_cm['b3']
    a3_ = bn_fn(a3)
    a3_ = bn_cm['rb3'][0] * a3_ + bn_cm['rb3'][1]
    layer3 = tf.nn.sigmoid(a3_)

    out_layer = tf.matmul(layer3, weights_cm['out']) + biases_cm['out']

    return out_layer


def cm_cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=cm_param['learning_rate'])

    cm_var = [weights_cm['h1'], weights_cm['h2'], weights_cm['h3'], weights_cm['out'],
              biases_sm['b1'], biases_sm['b2'], biases_cm['b3'], biases_sm['out'],
              bn_cm['rb1'], bn_cm['rb2'], bn_cm['rb3']]

    train_op = optimizer.minimize(loss_op, var_list=cm_var)

    return train_op, loss_op


def evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy
# complicated model


# simple model
def sm_model_fn(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_sm['h1']), biases_sm['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_sm['h2']), biases_sm['b2']))
    out_layer = tf.matmul(layer_2, weights_sm['out']) + biases_sm['out']

    return out_layer


def softmax(x):

    temperature = 2.5
    dim = tf.shape(x)
    x_exp = tf.exp(x/temperature)
    x_sum = tf.transpose(tf.tile([tf.reduce_sum(x_exp, 1)], (dim[1], 1)))
    x_ = x_exp / x_sum

    return x_


def distill_cost_fn(logits_cm, logits_sm):

    soft_cm = softmax(logits_cm)
    soft_sm = softmax(logits_sm)
    loss_op = -tf.reduce_mean(2.5 * 2.5 * soft_cm * tf.log(soft_sm))
    optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
    sm_var = [weights_sm['h1'], weights_sm['h2'], weights_sm['out'],
              biases_sm['b1'], biases_sm['b2'], biases_sm['out']]
    train_op = optimizer.minimize(loss_op, var_list=sm_var)

    return train_op, loss_op


def model_fn(x, y, mode):

    if mode == 'train':
        logits = cm_model_fn(x)
        train_op, loss_op = cm_cost_fn(logits, y)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)

        return train_op, loss_op, accuracy

    elif mode == 'distill':
        logits_cm = cm_model_fn(x)
        logits_sm = sm_model_fn(x)
        dis_train_op, dis_loss_op = distill_cost_fn(logits_cm, logits_sm)

        return dis_train_op, dis_loss_op

    elif mode == 'train_distill':

        logits_cm = cm_model_fn(x)
        logits_sm = sm_model_fn(x)
        cm_train_op, cm_loss_op = cm_cost_fn(logits_cm, y)
        dis_train_op, dis_loss_op = distill_cost_fn(logits_cm, logits_sm)

        return cm_train_op, cm_loss_op, dis_train_op, dis_loss_op

    elif mode == 'test_cm':
        logits = cm_model_fn(x)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)

        return accuracy

    elif mode == 'test_distill':
        logits = sm_model_fn(x)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)

        return accuracy


