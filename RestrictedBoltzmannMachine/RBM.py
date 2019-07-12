from Lib import *


def sample_h_given_v(vis):

    return tf.nn.sigmoid(tf.matmul(vis, wb['w']) + wb['h'])


def binomial(prob):

    dim = tf.shape(prob)
    one = tf.ones(dim)
    zero = tf.zeros(dim)
    u = tf.random_uniform(shape=dim, minval=0, maxval=1)

    return tf.where(prob > u, one, zero)


# v -> h
def positive_phase(vis):

    pos_h_probs = sample_h_given_v(vis)
    pos_prods = tf.matmul(tf.transpose(vis), pos_h_probs)
    pos_hid = tf.reduce_sum(pos_h_probs, 0)
    pos_vis = tf.reduce_sum(vis, 0)
    pos_h_states = binomial(pos_h_probs)

    return pos_h_states, pos_prods, pos_hid, pos_vis


# h -> v
def negtive_phase(hid, mask):

    neg_data = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(wb['w'])) + wb['v'])
    neg_data_re = tf.reshape(neg_data, (-1, 5))
    neg_data_re = tf.one_hot(tf.argmax(neg_data_re, 1), 5)
    mask_re = tf.reshape(mask, (-1, 5))
    neg_data_re = neg_data_re * mask_re
    neg_data = tf.reshape(neg_data_re, tf.shape(neg_data))

    neg_h_probs = sample_h_given_v(neg_data)
    neg_prods = tf.matmul(tf.transpose(neg_data), neg_h_probs)
    neg_hid = tf.reduce_sum(neg_h_probs)
    neg_vis = tf.reduce_sum(neg_data)

    return neg_data, neg_prods, neg_hid, neg_vis


def cost_fn(vis, neg_data):

    return tf.reduce_sum(tf.pow(vis - neg_data, 2))


def param_update(epoch, pos_prods, pos_hid, pos_vis, neg_prods, neg_hid, neg_vis):

    mu = tf.where(epoch > 5, [param_rbm['final_mu']], [param_rbm['initial_mu']])

    wb['w_inc'] = mu * wb['w_inc'] + param_rbm['learning_rate'] * ((pos_prods - neg_prods)/param_rbm['batch_size'] -
                                                                   param_rbm['weight_cost'] * wb['w'])
    wb['v_inc'] = mu * wb['v_inc'] + param_rbm['learning_rate'] * (pos_vis - neg_vis) / param_rbm['batch_size']
    wb['h_inc'] = mu * wb['h_inc'] + param_rbm['learning_rate'] * (pos_hid - neg_hid) / param_rbm['batch_size']

    wb['w'] = wb['w'] + wb['w_inc']
    wb['v'] = wb['v'] + wb['v_inc']
    wb['h'] = wb['h'] + wb['h_inc']


def rbm(vis, mask, epoch):

    pos_h_states, pos_prods, pos_hid, pos_vis = positive_phase(vis)
    neg_data, neg_prods, neg_hid, neg_vis = negtive_phase(pos_h_states, mask)
    cost = cost_fn(vis, neg_data)
    param_update(epoch, pos_prods, pos_hid, pos_vis, neg_prods, neg_hid, neg_vis)

    return cost
