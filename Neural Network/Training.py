from Lib import *
from SimpleNeuralNetwork import snn_model_fn
from ConvolutionalNeuralNetwork import cnn_model_fn
from AutoEncoder import ae_model_fn
from Storage import get_path


def train(mnist, mode):

    model_path = get_path(mode)

    if mode == 'SNN':
        train_snn(mnist, model_path)
    elif mode == 'CNN':
        train_cnn(mnist, model_path)
    elif mode == 'AE':
        train_ae(mnist, model_path)


def train_snn(mnist, model_path):

    x = tf.placeholder("float", [None, snn_param['num_inputs']])
    y = tf.placeholder("float", [None, snn_param['num_classes']])

    train_op, loss_op, accuracy = snn_model_fn(x, y)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

        for step in range(1, snn_param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(snn_param['batch_size'])
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            if step % snn_param['display_step'] == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

        print("Training Finished!")

        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels}))

        # save_path = saver.save(sess, model_path)
        # print("Model saved in file: %s" % save_path)


def train_cnn(mnist, model_path):

    x = tf.placeholder("float", [None, cnn_param['num_inputs']])
    y = tf.placeholder("float", [None, cnn_param['num_classes']])
    keep_prob = tf.placeholder(tf.float32)

    train_op, loss_op, accuracy = cnn_model_fn(x, y, keep_prob)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

        for step in range(1, cnn_param['num_steps'] + 1):

            batch_x, batch_y = mnist.train.next_batch(cnn_param['batch_size'])
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

            if step % cnn_param['display_step'] == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x, y: batch_y,
                                                                     keep_prob: 1.0})

                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

        print("Training Finished!")
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                                 y: mnist.test.labels[:256],
                                                                 keep_prob: 1.0}))

        # save_path = saver.save(sess, model_path)
        # print("Model saved in file: %s" % save_path)


def train_ae(mnist, model_path):

    x = tf.placeholder("float", [None, ae_param['num_inputs']])

    train_op, loss_op,  decoder_op = ae_model_fn(x)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

        for step in range(1, ae_param['num_steps'] + 1):
            batch_x, _ = mnist.train.next_batch(ae_param['batch_size'])
            sess.run(train_op, feed_dict={x: batch_x})

            if step % ae_param['display_step'] == 0 or step == 1:
                loss = sess.run(loss_op, feed_dict={x: batch_x})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss))

        print("Training Finished!")

        # save_path = saver.save(sess, model_path)
        # print("Model saved in file: %s" % save_path)

        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))

        for i in range(n):

            batch_x, _ = mnist.test.next_batch(n)
            g = sess.run(decoder_op, feed_dict={x: batch_x})

            for j in range(n):
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])

            for j in range(n):
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        # plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()
