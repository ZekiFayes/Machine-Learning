from Lib import *


def dcgan_generator(x, reuse=False):

    with tf.variable_scope('Generator', reuse=reuse):

        layer1 = tf.nn.tanh(tf.layers.dense(x, units=6 * 6 * 128))
        layer1 = tf.reshape(layer1, shape=[-1, 6, 6, 128])
        layer2 = tf.layers.conv2d_transpose(layer1, 64, 4, strides=2)
        layer3 = tf.layers.conv2d_transpose(layer2, 1, 2, strides=2)
        out = tf.nn.sigmoid(layer3)

        return out


def dcgan_discriminator(x, reuse=False):

    with tf.variable_scope('Discriminator', reuse=reuse):

        layer1 = tf.nn.tanh(tf.layers.conv2d(x, 64, 5))
        layer2 = tf.layers.average_pooling2d(layer1, 2, 2)
        layer3 = tf.nn.tanh(tf.layers.conv2d(layer2, 128, 5))
        layer4 = tf.layers.average_pooling2d(layer3, 2, 2)
        layer4 = tf.contrib.layers.flatten(layer4)
        layer5 = tf.nn.tanh(tf.layers.dense(layer4, 1024))
        out = tf.layers.dense(layer5, 2)

    return out


def dcgan_cost_fn(disc_concat, disc_target, stacked_gan, gen_target):

    disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_concat, labels=disc_target))
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=stacked_gan, labels=gen_target))

    return disc_loss, gen_loss


def dcgan(noise_input, real_image_input, disc_target, gen_target):

    gen_sample = dcgan_generator(noise_input)

    disc_real = dcgan_discriminator(real_image_input)
    disc_fake = dcgan_discriminator(gen_sample, reuse=True)
    disc_concat = tf.concat([disc_real, disc_fake], axis=0)
    stacked_gan = dcgan_discriminator(gen_sample, reuse=True)

    disc_loss, gen_loss = dcgan_cost_fn(disc_concat, disc_target, stacked_gan, gen_target)

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=param_dcgan['learning_rate'])
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=param_dcgan['learning_rate'])

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    return train_gen, train_disc, gen_loss, disc_loss
