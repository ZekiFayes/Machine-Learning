from Lib import *
from Model import model_fn
from Hparams import *


def train(mnist):

    x = tf.placeholder("float", [None, param['num_inputs']])
    y = tf.placeholder("float", [None, param['num_classes']])
    epoch = tf.placeholder("float", [1])

    r, loss_op = model_fn(x, y, epoch, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # print("Model restore in file: %s" % model_path)
        # saver.restore(sess, model_path)

        for step in range(1, param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(param['batch_size'])

            one, loss = sess.run([r, loss_op], feed_dict={x: batch_x, y: batch_y, epoch: [step]})

            if step == 1 or step % param['display_step'] == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

        saver.save(sess, model_path)
        print('Model stored in file: %s ' % model_path)
