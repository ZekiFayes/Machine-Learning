from Lib import *
from Model import ae_model_fn, rbm_model_fn
from Hparams import *


def train(mnist, option):

    if option == 'pretrain_with_AE':
        train_ae(mnist)

    elif option == 'pretrain_with_RBM':
        train_rbm(mnist)


def train_ae(mnist):

    x = tf.placeholder("float", [None, sm_param['num_inputs']])
    y = tf.placeholder("float", [None, sm_param['num_classes']])

    train_op1, loss_op1 = ae_model_fn(x, y, 'layer1')
    train_op2, loss_op2 = ae_model_fn(x, y, 'layer2')
    train_op3, loss_op3 = ae_model_fn(x, y, 'out')
    train_op, loss_op, accuracy = ae_model_fn(x, y, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title('Weight[h1]')
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title('Bias[b1]')
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title('Weight[h2]')
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title('Bias[b2]')
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title('Weight[out]')
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.set_title('Bias[out]')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    with tf.Session() as sess:
        sess.run(init)

        print("Model restore in file: %s" % model_path)
        saver.restore(sess, model_path)

        for step in range(1, sm_param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(sm_param['batch_size'])

            if step <= 5000:
                _, loss, w, b = sess.run([train_op1, loss_op1, weights_sm, biases_sm], feed_dict={x: batch_x, y: batch_y})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", layer 1 Minibatch Loss= " + "{:.4f}".format(loss))
                    ax1.plot(w['h1'].reshape(784 * 200))
                    ax2.plot(b['b1'])
                    ax3.plot(w['h2'].reshape(200 * 200))
                    ax4.plot(b['b2'])
                    ax5.plot(w['out'].reshape(200 * 10))
                    ax6.plot(b['out'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 5000 < step <= 10000:
                _, loss, w, b = sess.run([train_op2, loss_op2, weights_sm, biases_sm], feed_dict={x: batch_x, y: batch_y})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", layer 2 Minibatch Loss= " + "{:.4f}".format(loss))
                    ax1.plot(w['h1'].reshape(784 * 200))
                    ax2.plot(b['b1'])
                    ax3.plot(w['h2'].reshape(200 * 200))
                    ax4.plot(b['b2'])
                    ax5.plot(w['out'].reshape(200 * 10))
                    ax6.plot(b['out'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 10000 < step <= 15000:
                _, loss, w, b = sess.run([train_op3, loss_op3, weights_sm, biases_sm], feed_dict={x: batch_x, y: batch_y})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", Out layer Minibatch Loss= " + "{:.4f}".format(loss))
                    ax1.plot(w['h1'].reshape(784 * 200))
                    ax2.plot(b['b1'])
                    ax3.plot(w['h2'].reshape(200 * 200))
                    ax4.plot(b['b2'])
                    ax5.plot(w['out'].reshape(200 * 10))
                    ax6.plot(b['out'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 15000 < step:
                _, loss, acc, w, b = sess.run([train_op, loss_op, accuracy, weights_sm, biases_sm],
                                              feed_dict={x: batch_x, y: batch_y})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " +
                          "{:.4f}".format(acc))

                    ax1.plot(w['h1'].reshape(784*200))
                    ax2.plot(b['b1'])
                    ax3.plot(w['h2'].reshape(200 * 200))
                    ax4.plot(b['b2'])
                    ax5.plot(w['out'].reshape(200 * 10))
                    ax6.plot(b['out'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig' + str(int(step/1000)) + '.png'
                    plt.savefig(name)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)


def train_rbm(mnist):

    x = tf.placeholder("float", [None, sm_param['num_inputs']])
    y = tf.placeholder("float", [None, sm_param['num_classes']])
    epoch = tf.placeholder('float', [1])
    loss_op1 = rbm_model_fn(x, y, epoch, 'layer1')
    loss_op2 = rbm_model_fn(x, y, epoch, 'layer2')
    loss_op3 = rbm_model_fn(x, y, epoch, 'out')
    train_op, loss_op, accuracy = rbm_model_fn(x, y, epoch, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_title('Weight[w1]')
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_title('Weight[v1]')
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.set_title('Weight[h1]')
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_title('Weight[w2]')
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_title('Weight[v2]')
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_title('Weight[h2]')
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_title('Weight[w3]')
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_title('Weight[v3]')
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_title('Weight[h3]')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    with tf.Session() as sess:
        sess.run(init)

        # print("Model restore in file: %s" % model_path)
        # saver.restore(sess, model_path)

        for step in range(1, sm_param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(sm_param['batch_size'])

            if step <= 5000:
                loss, w = sess.run([loss_op1, weights_rbm], feed_dict={x: batch_x, y: batch_y, epoch: [step]})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", layer 1 Minibatch Loss= " + "{:.4f}".format(loss))
                    ax1.plot(w['w1'].reshape(784 * 200))
                    ax2.plot(w['v1'])
                    ax3.plot(w['h1'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig_rbm_' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 5000 < step <= 10000:
                loss, w = sess.run([loss_op2, weights_rbm],
                                   feed_dict={x: batch_x, y: batch_y, epoch: [step]})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", layer 2 Minibatch Loss= " + "{:.4f}".format(loss))
                    ax4.plot(w['w2'].reshape(200 * 200))
                    ax5.plot(w['v2'])
                    ax6.plot(w['h2'])

                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig_rbm_' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 10000 < step <= 15000:
                loss, w = sess.run([loss_op3, weights_rbm],
                                   feed_dict={x: batch_x, y: batch_y, epoch: [step]})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", Out layer Minibatch Loss= " + "{:.4f}".format(loss))
                    ax7.plot(w['w3'].reshape(200 * 10))
                    ax8.plot(w['v3'])
                    ax9.plot(w['h3'])
                    plt.draw()
                    plt.pause(0.1)
                    name = 'Figure/fig' + str(int(step / 1000)) + '.png'
                    plt.savefig(name)

            elif 15000 < step:
                _, loss, acc = sess.run([train_op, loss_op, accuracy],
                                        feed_dict={x: batch_x, y: batch_y, epoch: [step]})
                if step % sm_param['display_step'] == 0:
                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " +
                          "{:.4f}".format(acc))

                    # plt.draw()
                    # plt.pause(0.1)
                    # name = 'Figure/fig' + str(int(step / 1000)) + '.png'
                    # plt.savefig(name)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
