from Lib import *
from Train import train
from Preprocessing import load_data

"""
This is to demonstrate how RNN works. We build a two-step RNN.
"""


def main(argv=None):

    print('This is to build a simple RNN.')
    print('Loading data')
    x = load_data()
    print('Finished!')
    print('Starting training')
    train(x)


if __name__ == '__main__':
    tf.app.run()
