from Hparams import *


def evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def pretrain_model_fn(x, mode):

    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_sm['h1']), biases_sm['b1']))
    if mode == 'layer1':
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights_sm['h11']), biases_sm['b11']))
        return layer1

    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights_sm['h2']), biases_sm['b2']))
    if mode == 'layer2':
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights_sm['h22']), biases_sm['b22']))
        return layer1, layer2

    out_layer = tf.matmul(layer2, weights_sm['out']) + biases_sm['out']
    if mode == 'out':
        out_layer = tf.matmul(out_layer, weights_sm['out1']) + biases_sm['out1']
        return layer2, out_layer

    return out_layer


def cost_fn(logits, y, mode):

    if mode == 'layer1':
        loss_op = tf.reduce_mean(tf.square(logits - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
        sm_var = [weights_sm['h1'], weights_sm['h11'],
                  biases_sm['b1'], biases_sm['b11']]
        train_op = optimizer.minimize(loss_op, var_list=sm_var)
        return train_op, loss_op

    elif mode == 'layer2':
        loss_op = tf.reduce_mean(tf.square(logits - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
        sm_var = [weights_sm['h2'], weights_sm['h22'],
                  biases_sm['b2'], biases_sm['b22']]
        train_op = optimizer.minimize(loss_op, var_list=sm_var)
        return train_op, loss_op

    elif mode == 'out':
        loss_op = tf.reduce_mean(tf.square(logits - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
        sm_var = [weights_sm['out'], weights_sm['out1'],
                  biases_sm['out'], biases_sm['out1']]
        train_op = optimizer.minimize(loss_op, var_list=sm_var)
        return train_op, loss_op

    elif mode == 'train':
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
        train_op = optimizer.minimize(loss_op)

        return train_op, loss_op


def model_fn(x, y, mode):

    if mode == 'layer1':
        logits = pretrain_model_fn(x, mode)
        train_op, loss_op = cost_fn(logits, x, mode)
        return train_op, loss_op

    elif mode == 'layer2':
        logits1, logits2 = pretrain_model_fn(x, mode)
        train_op, loss_op = cost_fn(logits1, logits2, mode)
        return train_op, loss_op

    elif mode == 'out':
        logits1, logits2 = pretrain_model_fn(x, mode)
        train_op, loss_op = cost_fn(logits1, logits2, mode)
        return train_op, loss_op

    elif mode == 'train':
        logits = pretrain_model_fn(x, mode)
        train_op, loss_op = cost_fn(logits, y, mode)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        logits = pretrain_model_fn(x, 'train')
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return accuracy
