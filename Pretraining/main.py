from Lib import *
from Train import train
from Preprocessing import load_data


def main(argv=None):

    print('This is to pretrain a shallow neural network using autoencoder.')
    print('Loading data')
    x = load_data()
    print('Finished!')
    print('Starting training')
    train(x)


if __name__ == '__main__':
    tf.app.run()
