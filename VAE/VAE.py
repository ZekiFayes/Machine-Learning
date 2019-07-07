from Lib import *


def vae_encoder(x):

    encoder = tf.nn.tanh(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])

    z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
    z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']
    eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                           name='epsilon')
    z = z_mean + tf.exp(z_std / 2) * eps

    return z_mean, z_std, z


def vae_decoder(x):

    decoder = tf.nn.tanh(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    decoder = tf.nn.sigmoid(tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out'])

    return decoder


def vae_cost_fn(x_true, x_reconstructed, z_mean, z_std):

    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


def vae(x):

    z_mean, z_std, encoder_op = vae_encoder(x)
    decoder_op = vae_decoder(encoder_op)

    loss_op = vae_cost_fn(x, decoder_op, z_mean, z_std)
    optimizer = tf.train.AdamOptimizer(param_vae['leanring_rate'])
    train_op = optimizer.minimize(loss_op)

    return loss_op, train_op

