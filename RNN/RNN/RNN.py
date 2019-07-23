from Hparams import *


def evaluation_fn(prediction, y):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def rnn_block(x, state_0, time):

    t_step = 't' + str(time)
    state = tf.nn.sigmoid(tf.matmul(x, u[t_step]) + tf.matmul(state_0, w[t_step]) + bs[t_step])
    out = tf.nn.sigmoid(tf.matmul(state, v[t_step]) + bo[t_step])
    # out = tf.nn.softmax(tf.matmul(state, v[t_step]) + bo[t_step])
    return state, out


# This is just to know how RNN works with matrix operations.
# We only feed the input at time t = 1
# The input at other time will zero.
def rnn(x):

    dim = tf.shape(x)
    t = 1
    state0 = tf.zeros([dim[0], param['hidden']])
    state1, out1 = rnn_block(x, state0, t)
    t = t + 1

    x_in = tf.zeros([dim[0], param['num_inputs']])
    state2, out2 = rnn_block(x_in, state1, t)
    t = t + 1

    x_in = tf.zeros([dim[0], param['num_inputs']])
    state3, out3 = rnn_block(x_in, state2, t)

    out = (out1 + out2 + out3) / 2

    return out


def cost_fn(logits, y):

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=param['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op


def model_fn(x, y, mode):

    logits = rnn(x)
    prediction = tf.nn.softmax(logits)
    accuracy = evaluation_fn(prediction, y)

    if mode == 'train':
        train_op, loss_op = cost_fn(logits, y)
        return train_op, loss_op, accuracy

    elif mode == 'test':
        return accuracy
