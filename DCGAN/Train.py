from Lib import *
from DCGAN import dcgan


def train(x):

    noise_input = tf.placeholder(tf.float32, shape=[None, param_dcgan['num_noise']])
    real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    disc_target = tf.placeholder(tf.int32, shape=[None])
    gen_target = tf.placeholder(tf.int32, shape=[None])

    train_gen, train_disc, gen_loss, disc_loss = dcgan(noise_input, real_image_input, disc_target, gen_target)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        # saver.restore(sess, model_path)

        for i in range(1, param_dcgan['num_steps'] + 1):

            batch_x, _ = x.train.next_batch(param_dcgan['batch_size'])
            batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
            z = np.random.uniform(-1., 1., size=[param_dcgan['batch_size'], param_dcgan['num_noise']])
            batch_disc_y = np.concatenate(
                [np.ones([param_dcgan['batch_size']]), np.zeros([param_dcgan['batch_size']])], axis=0)
            batch_gen_y = np.ones([param_dcgan['batch_size']])

            feed_dict = {real_image_input: batch_x, noise_input: z,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)

            if i % 100 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        saver.save(sess, model_path)
