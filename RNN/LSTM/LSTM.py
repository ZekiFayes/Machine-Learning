from Hparams import *


def lstm_block(x, state, C, t):

    time = 't' + str(t)
    f = tf.nn.sigmoid(tf.matmul(state, Wf[time]) + tf.matmul(x, Uf[time]) + Bf[time])
    i = tf.nn.sigmoid(tf.matmul(state, Wi[time]) + tf.matmul(x, Ui[time]) + Bi[time])
    C_ = tf.nn.tanh(tf.matmul(state, Wc[time]) + tf.matmul(x, Uc[time]) + Bc[time])
    C1 = f * C + i * C_
    o = tf.nn.sigmoid(tf.matmul(state, Wo[time]) + tf.matmul(x, Uo[time]) + Bo[time])
    state1 = o * tf.nn.tanh(C1)
    out = tf.nn.softmax(tf.matmul(state1, V[time]) + Bs[time])

    return state1, C1, out


def lstm(x):

    dim = tf.shape(x)
    t = 1
    state0 = tf.zeros([dim[0], param['hidden']])
    C0 = tf.zeros([dim[0], param['hidden']])
    state1, C1, out1 = lstm_block(x, state0, C0, t)
    t += 1

    x_in = tf.zeros([dim[0], param['num_inputs']])
    state2, C2, out2 = lstm_block(x_in, state1, C1, t)
    t += 1
    _, _, out3 = lstm_block(x_in, state2, C2, t)

    out = (out1 + out2 + out3) / 2

    return out


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

    logits = lstm(x)
    prediction = tf.nn.softmax(logits)
    accuracy = evaluation_fn(prediction, y)

    if mode == 'train':
        train_op, loss_op = cost_fn(logits, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        return accuracy
