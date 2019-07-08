from VAE import vae, vae_decoder
from Lib import *


def train(x_inputs):

    x = tf.placeholder(tf.float32, shape=[None, param_vae['num_inputs']])

    loss_op, train_op = vae(x)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, model_path)

        for i in range(1, param_vae['num_steps'] + 1):
            batch_x, _ = x_inputs.train.next_batch(param_vae['batch_size'])
            _, loss = sess.run([train_op, loss_op], feed_dict={x: batch_x})

            if i % 1000 == 0 or i == 1:
                print('Step %i, Loss: %f' % (i, loss))

        saver.save(sess, model_path)
