from Lib import *
from KD_model import model_fn
from Hparams import *


def train(mnist):

    x = tf.placeholder("float", [None, cm_param['num_inputs']])
    y = tf.placeholder("float", [None, cm_param['num_classes']])

    train_op, loss_op, accuracy = model_fn(x, y, 'train')
    dis_train_op, dis_loss_op = model_fn(x, y, 'distill')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    r1, r2, r3 = [], [], []
    b1, b2, b3 = [], [], []

    with tf.Session() as sess:
        sess.run(init)
        print("Model restore in file: %s" % model_path)
        saver.restore(sess, model_path)

        for step in range(1, cm_param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(cm_param['batch_size'])

            if step < 10000:
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
            else:
                sess.run(dis_train_op, feed_dict={x: batch_x, y: batch_y})

            if step % cm_param['display_step'] == 0 or step == 1:
                if step < 10000:
                    loss, acc, bn = sess.run([loss_op, accuracy, bn_cm], feed_dict={x: batch_x, y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " +
                          "{:.4f}".format(acc))
                else:
                    loss, bn = sess.run([dis_loss_op, bn_cm], feed_dict={x: batch_x, y: batch_y})
                    print("Step " + str(step) + ", Distill Loss= " +
                          "{:.4f}".format(loss))

                r1.append(bn['rb1'][0]), b1.append(bn['rb1'][1])
                r2.append(bn['rb2'][0]), b2.append(bn['rb2'][1])
                r3.append(bn['rb3'][0]), b3.append(bn['rb3'][1])

        print('--------------------------------------------------')
        print("Training Finished!")

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        plt.plot(np.arange(len(r1)), r1), plt.plot(np.arange(len(b1)), b1)
        plt.plot(np.arange(len(r2)), r2), plt.plot(np.arange(len(b2)), b2)
        plt.plot(np.arange(len(r3)), r3), plt.plot(np.arange(len(b3)), b3)
        plt.legend(['r1', 'b1', 'r2', 'b2', 'r3', 'b3'])
        plt.title('BN: gamma & beta')
        plt.show()
