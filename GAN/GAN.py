from Lib import *


def gan_generator(x):

    with tf.variable_scope('layer1'):
        hidden_layer = tf.nn.relu(tf.matmul(x, weights['gen_hidden1']) + biases['gen_hidden1'])
        out_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights['gen_out']) + biases['gen_out'])

    return out_layer


def gan_discriminator(x):

    with tf.variable_scope('layer2'):
        hidden_layer = tf.nn.relu(tf.matmul(x, weights['disc_hidden1']) + biases['disc_hidden1'])
        out_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights['disc_out']) + biases['disc_out'])

    return out_layer


def gan_cost_fn(disc_fake, disc_real):

    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    return gen_loss, disc_loss


def gan(gen_input, disc_input):

    gen_sample = gan_generator(gen_input)

    disc_real = gan_discriminator(disc_input)
    disc_fake = gan_discriminator(gen_sample)

    gen_loss, disc_loss = gan_cost_fn(disc_fake, disc_real)

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=param_gan['learning_rate'])
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=param_gan['learning_rate'])

    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]

    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                 biases['disc_hidden1'], biases['disc_out']]

    train_gen_op = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc_op = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    return gen_loss, disc_loss, train_gen_op, train_disc_op
