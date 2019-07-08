from Lib import *
from DCGAN import dcgan_generator


def evaluate():

    noise_input = tf.placeholder(tf.float32, shape=[None, param_dcgan['num_noise']])
    gen_sample = dcgan_generator(noise_input)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_path)
            f, a = plt.subplots(4, 10, figsize=(10, 4))
            for i in range(10):
                # Noise input.
                z = np.random.uniform(-1., 1., size=[4, param_dcgan['num_noise']])
                g = sess.run(gen_sample, feed_dict={noise_input: z})
                for j in range(4):
                    # Generate image from noise. Extend to 3 channels for matplot figure.
                    img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                     newshape=(28, 28, 3))
                    a[j][i].imshow(img)

            f.show()
            plt.show()
        else:
            print('No checkpoint file found!')
            return


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
