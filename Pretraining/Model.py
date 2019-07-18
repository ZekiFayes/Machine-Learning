from Hparams import *

"""Pretrain with AutoEncoder"""


def evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def ae_pretrain_model_fn(x, mode):

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


def ae_cost_fn(logits, y, mode):

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


def ae_model_fn(x, y, mode):

    if mode == 'layer1':
        logits = ae_pretrain_model_fn(x, mode)
        train_op, loss_op = ae_cost_fn(logits, x, mode)
        return train_op, loss_op

    elif mode == 'layer2':
        logits1, logits2 = ae_pretrain_model_fn(x, mode)
        train_op, loss_op = ae_cost_fn(logits1, logits2, mode)
        return train_op, loss_op

    elif mode == 'out':
        logits1, logits2 = ae_pretrain_model_fn(x, mode)
        train_op, loss_op = ae_cost_fn(logits1, logits2, mode)
        return train_op, loss_op

    elif mode == 'train':
        logits = ae_pretrain_model_fn(x, mode)
        train_op, loss_op = ae_cost_fn(logits, y, mode)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        logits = ae_pretrain_model_fn(x, 'train')
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return accuracy


""" Pretrain with AutoEncoder"""


# RBM
def binomial(prob):

    dim = tf.shape(prob)
    u = tf.random_uniform(shape=dim, minval=0, maxval=1)

    return tf.where(prob > u, tf.ones(dim), tf.zeros(dim))


def calculate(vis, probs):

    prods = tf.matmul(tf.transpose(vis), probs)
    hid = tf.reduce_sum(probs, 0)
    vis = tf.reduce_sum(vis, 0)

    return prods, hid, vis


def rbm_pretrain_model_fn(x, mode):

    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_sm['h1']), biases_sm['b1']))
    if mode == 'layer1':
        return layer1

    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights_sm['h2']), biases_sm['b2']))
    if mode == 'layer2':
        return layer2

    out_layer = tf.matmul(layer2, weights_sm['out']) + biases_sm['out']
    return out_layer


def rbm1(vis, epoch):

    # positive phase
    pos_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w1']) + weights_rbm['h1'])
    pos_prods, pos_hid, pos_vis = calculate(vis, pos_h_probs)
    pos_h_states = binomial(pos_h_probs)

    # negative phase
    neg_data = tf.nn.sigmoid(tf.matmul(pos_h_states, tf.transpose(weights_rbm['w1'])) + weights_rbm['v1'])
    neg_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w1']) + weights_rbm['h1'])
    neg_prods, neg_hid, neg_vis = calculate(neg_data, neg_h_probs)

    # update
    mu = tf.where(epoch > 5, [param_rbm['final_mu']], [param_rbm['initial_mu']])

    dw = mu * weights_rbm['dw1'] + param_rbm['learning_rate'] * ((pos_prods - neg_prods)/param_rbm['batch_size'] -
                                                                 param_rbm['weight_cost'] * weights_rbm['w1'])
    dv = mu * weights_rbm['dv1'] + param_rbm['learning_rate'] * (pos_vis - neg_vis) / param_rbm['batch_size']
    dh = mu * weights_rbm['dh1'] + param_rbm['learning_rate'] * (pos_hid - neg_hid) / param_rbm['batch_size']

    weights_rbm['dw1'] = weights_rbm['dw1'].assign(dw)
    weights_rbm['dv1'] = weights_rbm['dv1'].assign(dv)
    weights_rbm['dh1'] = weights_rbm['dh1'].assign(dh)

    w = weights_rbm['w1'] + weights_rbm['dw1']
    v = weights_rbm['v1'] + weights_rbm['dv1']
    h = weights_rbm['h1'] + weights_rbm['dh1']

    weights_rbm['w1'] = weights_rbm['w1'].assign(w)
    weights_rbm['v1'] = weights_rbm['v1'].assign(v)
    weights_rbm['h1'] = weights_rbm['h1'].assign(h)

    # cost
    loss_op = tf.reduce_sum(tf.square(vis - neg_data))

    return loss_op


def rbm2(vis, epoch):

    # positive phase
    pos_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w2']) + weights_rbm['h2'])
    pos_prods, pos_hid, pos_vis = calculate(vis, pos_h_probs)
    pos_h_states = binomial(pos_h_probs)

    # negative phase
    neg_data = tf.nn.sigmoid(tf.matmul(pos_h_states, tf.transpose(weights_rbm['w2'])) + weights_rbm['v2'])
    neg_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w2']) + weights_rbm['h2'])
    neg_prods, neg_hid, neg_vis = calculate(neg_data, neg_h_probs)

    # update
    mu = tf.where(epoch > 5, [param_rbm['final_mu']], [param_rbm['initial_mu']])

    dw = mu * weights_rbm['dw2'] + param_rbm['learning_rate'] * ((pos_prods - neg_prods) / param_rbm['batch_size'] -
                                                                 param_rbm['weight_cost'] * weights_rbm['w2'])
    dv = mu * weights_rbm['dv2'] + param_rbm['learning_rate'] * (pos_vis - neg_vis) / param_rbm['batch_size']
    dh = mu * weights_rbm['dh2'] + param_rbm['learning_rate'] * (pos_hid - neg_hid) / param_rbm['batch_size']

    weights_rbm['dw2'] = weights_rbm['dw2'].assign(dw)
    weights_rbm['dv2'] = weights_rbm['dv2'].assign(dv)
    weights_rbm['dh2'] = weights_rbm['dh2'].assign(dh)

    w = weights_rbm['w2'] + weights_rbm['dw2']
    v = weights_rbm['v2'] + weights_rbm['dv2']
    h = weights_rbm['h2'] + weights_rbm['dh2']

    weights_rbm['w2'] = weights_rbm['w2'].assign(w)
    weights_rbm['v2'] = weights_rbm['v2'].assign(v)
    weights_rbm['h2'] = weights_rbm['h2'].assign(h)

    # cost
    loss_op = tf.reduce_sum(tf.square(vis - neg_data))

    return loss_op


def rbm3(vis, epoch):

    # positive phase
    pos_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w3']) + weights_rbm['h3'])
    pos_prods, pos_hid, pos_vis = calculate(vis, pos_h_probs)
    pos_h_states = binomial(pos_h_probs)

    # negative phase
    neg_data = tf.nn.sigmoid(tf.matmul(pos_h_states, tf.transpose(weights_rbm['w3'])) + weights_rbm['v3'])
    neg_h_probs = tf.nn.sigmoid(tf.matmul(vis, weights_rbm['w3']) + weights_rbm['h3'])
    neg_prods, neg_hid, neg_vis = calculate(neg_data, neg_h_probs)

    # update
    mu = tf.where(epoch > 5, [param_rbm['final_mu']], [param_rbm['initial_mu']])

    dw = mu * weights_rbm['dw3'] + param_rbm['learning_rate'] * ((pos_prods - neg_prods) / param_rbm['batch_size'] -
                                                                 param_rbm['weight_cost'] * weights_rbm['w3'])
    dv = mu * weights_rbm['dv3'] + param_rbm['learning_rate'] * (pos_vis - neg_vis) / param_rbm['batch_size']
    dh = mu * weights_rbm['dh3'] + param_rbm['learning_rate'] * (pos_hid - neg_hid) / param_rbm['batch_size']

    weights_rbm['dw3'] = weights_rbm['dw3'].assign(dw)
    weights_rbm['dv3'] = weights_rbm['dv3'].assign(dv)
    weights_rbm['dh3'] = weights_rbm['dh3'].assign(dh)

    w = weights_rbm['w3'] + weights_rbm['dw3']
    v = weights_rbm['v3'] + weights_rbm['dv3']
    h = weights_rbm['h3'] + weights_rbm['dh3']

    weights_rbm['w3'] = weights_rbm['w3'].assign(w)
    weights_rbm['v3'] = weights_rbm['v3'].assign(v)
    weights_rbm['h3'] = weights_rbm['h3'].assign(h)

    # cost
    loss_op = tf.reduce_sum(tf.square(vis - neg_data))

    return loss_op


def rbm_cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=sm_param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def rbm_model_fn(x, y, epoch, mode):

    if mode == 'layer1':
        if param_rbm['restart'] == 1:
            param_rbm['restart'] = param_rbm['restart'] + 1

        loss_op = rbm1(x, epoch)

        return loss_op

    elif mode == 'layer2':
        if param_rbm['restart'] == 2:
            param_rbm['restart'] = param_rbm['restart'] + 1
            weights_sm['h1'] = weights_sm['h1'].assign(weights_rbm['w1'])
            biases_sm['b1'] = biases_sm['b1'].assign(weights_rbm['h1'])

        x = rbm_pretrain_model_fn(x, 'layer1')
        loss_op = rbm2(x, epoch)

        return loss_op

    elif mode == 'out':
        if param_rbm['restart'] == 3:
            param_rbm['restart'] = param_rbm['restart'] + 1
            weights_sm['h2'] = weights_sm['h2'].assign(weights_rbm['w2'])
            biases_sm['b2'] = weights_sm['b2'].assign(weights_rbm['h2'])

        x = rbm_pretrain_model_fn(x, 'layer2')
        loss_op = rbm3(x, epoch)

        return loss_op

    elif mode == 'train':
        if param_rbm['restart'] == 4:
            param_rbm['restart'] = param_rbm['restart'] + 1
            weights_sm['out'] = weights_sm['out'].assign(weights_rbm['w3'])
            biases_sm['out'] = biases_sm['out'].assign(weights_rbm['h3'])

        logits = rbm_pretrain_model_fn(x, mode)
        train_op, loss_op = rbm_cost_fn(logits, y)
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        logits = rbm_pretrain_model_fn(x, 'train')
        prediction = tf.nn.softmax(logits)
        accuracy = evaluation_fn(prediction, y)
        return accuracy


# RBM
