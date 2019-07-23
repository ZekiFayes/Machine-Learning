from Lib import *
from RNN import model_fn
from Hparams import *


def train(mnist):

    x = tf.placeholder("float", [None, param['num_inputs']])
    y = tf.placeholder("float", [None, param['num_classes']])

    train_op, loss_op, accuracy = model_fn(x, y, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    fig = plt.figure()
    ax1 = fig.add_subplot(5, 2, 1)
    ax1.set_title('u[1]')
    ax2 = fig.add_subplot(5, 2, 2)
    ax2.set_title('u[2]')
    ax3 = fig.add_subplot(5, 2, 3)
    ax3.set_title('w[1]')
    ax4 = fig.add_subplot(5, 2, 4)
    ax4.set_title('w[2]')
    ax5 = fig.add_subplot(5, 2, 5)
    ax5.set_title('bs[1]')
    ax6 = fig.add_subplot(5, 2, 6)
    ax6.set_title('bs[2]')
    ax7 = fig.add_subplot(5, 2, 7)
    ax7.set_title('v[1]')
    ax8 = fig.add_subplot(5, 2, 8)
    ax8.set_title('v[2]')
    ax9 = fig.add_subplot(5, 2, 9)
    ax9.set_title('bo[1]')
    ax0 = fig.add_subplot(5, 2, 10)
    ax0.set_title('bo[2]')
    plt.subplots_adjust(wspace=0.7, hspace=0.7)

    with tf.Session() as sess:
        sess.run(init)

        # print("Model restore in file: %s" % model_path)
        # saver.restore(sess, model_path)

        for step in range(1, param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(param['batch_size'])

            _, loss, U, W, BS, V, BO = sess.run([train_op, loss_op, u, w, bs, v, bo], feed_dict={x: batch_x, y: batch_y})
            if step % param['display_step'] == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))
                ax1.plot(U['t1'].reshape(784 * 200))
                ax2.plot(U['t2'].reshape(784 * 200))

                ax3.plot(W['t1'].reshape(200 * 200))
                ax4.plot(W['t2'].reshape(200 * 200))

                ax5.plot(BS['t1'])
                ax6.plot(BS['t2'])

                ax7.plot(V['t1'].reshape(200 * 10))
                ax8.plot(V['t2'].reshape(200 * 10))

                ax9.plot(BO['t1'])
                ax0.plot(BO['t2'])

                plt.draw()
                plt.pause(0.1)
                name = 'Figure/fig' + str(int(step / param['display_step'])) + '.png'
                plt.savefig(name)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
