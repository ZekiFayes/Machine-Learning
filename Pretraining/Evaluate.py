from Lib import *
from Hparams import *
from Model import model_fn
from Preprocessing import load_data


def evaluate(mnist):

    x = tf.placeholder("float", [None, sm_param['num_inputs']])
    y = tf.placeholder("float", [None, sm_param['num_classes']])

    accuracy = model_fn(x, y, 'test')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(dir_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_path)
            print("Testing Accuracy:",
                  sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y: mnist.test.labels}))
        else:
            print('No checkpoint file found!')
            return


def main(argv=None):
    x = load_data()
    evaluate(x)


if __name__ == '__main__':
    tf.app.run()
