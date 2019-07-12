from Lib import *
from RBM import rbm


def train(data, mask):

    print('---------------------------------------------')
    print('Starting training')

    v_input = tf.placeholder('float', [None, param_rbm['num_v_nodes']])
    v_mask = tf.placeholder('float', [None, param_rbm['num_v_nodes']])
    epoch = tf.placeholder('float', [1])

    loss_op = rbm(v_input, v_mask, epoch)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        errsum = 0.0
        for step in range(param_rbm['num_epochs']):
            for batch in range(1, 2016):
                start = (step * param_rbm['batch_size']) % 2015
                end = min(start + param_rbm['batch_size'], 2015)

                x = data[start:end]
                m = mask[start:end]

                loss = sess.run(loss_op, feed_dict={v_input: x, v_mask: m, epoch: [step]})
                errsum = errsum + loss
                print('epoch=%d, batch=%d, loss=%f' % (step, batch, loss))

            print('epoch=%d, loss=%f' % (step, errsum / 2015))

        print('Training Finished!')
        print('---------------------------------------------')
