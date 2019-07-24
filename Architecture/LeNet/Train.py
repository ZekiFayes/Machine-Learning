from Lib import *
from LeNet5 import model_fn
from Hparams import *


def train(mnist):

    x = tf.placeholder("float", [param['batch_size'], param['num_inputs'], param['num_inputs'], param['num_channels']])
    y = tf.placeholder("float", [param['batch_size'], param['num_classes']])

    train_op, loss_op, accuracy = model_fn(x, y, 'train')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        print("Model restore in file: %s" % model_path)
        saver.restore(sess, model_path)

        for step in range(1, param['num_steps'] + 1):
            batch_x, batch_y = mnist.train.next_batch(param['batch_size'])
            batch_x = np.reshape(batch_x, [param['batch_size'], param['num_inputs'],
                                           param['num_inputs'], param['num_channels']])

            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={x: batch_x, y: batch_y})
            if step % param['display_step'] == 0:
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Accuracy= " + "{:.4f}".format(acc))

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
