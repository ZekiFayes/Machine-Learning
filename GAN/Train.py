from Lib import *
from GAN import gan, gan_generator


def train(x):

    gen_input = tf.placeholder(tf.float32, shape=[None, param_gan['num_noise']], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, param_gan['num_inputs']], name='disc_input')

    gen_loss, disc_loss, train_gen_op, train_disc_op = gan(gen_input, disc_input)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        for i in range(1, param_gan['num_steps'] + 1):
            batch_x, _ = x.train.next_batch(param_gan['batch_size'])
            z = np.random.uniform(-1., 1., size=[param_gan['batch_size'], param_gan['num_noise']])
            _, _, gl, dl = sess.run([train_gen_op, train_disc_op, gen_loss, disc_loss],
                                    feed_dict={disc_input: batch_x, gen_input: z})

            if i % 1000 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        saver.save(sess, model_path)
