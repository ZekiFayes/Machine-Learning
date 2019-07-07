from VAE import vae, vae_decoder
from Lib import *


def train(x_inputs):

    x = tf.placeholder(tf.float32, shape=[None, param_vae['num_inputs']])

    loss_op, train_op = vae(x)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # saver.restore(sess, model_path)

        for i in range(1, param_vae['num_steps'] + 1):
            batch_x, _ = x_inputs.train.next_batch(param_vae['batch_size'])
            _, loss = sess.run([train_op, loss_op], feed_dict={x: batch_x})

            if i % 1000 == 0 or i == 1:
                print('Step %i, Loss: %f' % (i, loss))

        saver.save(sess, model_path)

        noise_input = tf.placeholder(tf.float32, shape=[None, param_vae['latent']])
        decoder = vae_decoder(noise_input)
        n = 20
        x_axis = np.linspace(-3, 3, n)
        y_axis = np.linspace(-3, 3, n)

        canvas = np.empty((28 * n, 28 * n))
        for i, yi in enumerate(x_axis):
            for j, xi in enumerate(y_axis):
                z_mu = np.array([[xi, yi]] * param_vae['batch_size'])
                x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
                canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_axis, y_axis)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()
