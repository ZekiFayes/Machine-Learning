from VAE import vae_decoder
from Lib import *


def evaluate():

    x = tf.placeholder(tf.float32, shape=[None, param_vae['latent']])
    decoder = vae_decoder(x)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_path)
            n = 20
            x_axis = np.linspace(-3, 3, n)
            y_axis = np.linspace(-3, 3, n)

            canvas = np.empty((28 * n, 28 * n))
            for i, yi in enumerate(x_axis):
                for j, xi in enumerate(y_axis):
                    z_mu = np.array([[xi, yi]] * param_vae['batch_size'])
                    x_mean = sess.run(decoder, feed_dict={x: z_mu})
                    canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_axis, y_axis)
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.show()
        else:
            print('No checkpoint file found!')
            return


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
