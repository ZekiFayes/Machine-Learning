from Hparams import *
from LSTM import model_fn
from Lib import *


def train(mnist):

    x = tf.placeholder('float', [None, param['num_inputs']])
    y = tf.placeholder('float', [None, param['num_classes']])

    train_op, loss_op, accuracy = model_fn(x, y, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        for step in range(1, param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(param['batch_size'])

            _, loss = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
            if step % param['display_step'] == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

        print('Finished!')
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
