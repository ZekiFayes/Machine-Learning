from Lib import *
from Train import train
from Preprocessing import load_data


def main(argv=None):

    print('This is to pretrain a shallow neural network using autoencoder or restricted boltzmann machine.'
          'There are two options.'
          '1 -> pretrain_with_RBM'
          '2 -> pretrain_with_AE')
    print('Loading data')
    x = load_data()
    print('Finished!')
    print()
    option = 'pretrain_with_RBM'
    print('Starting training')
    train(x, option)


if __name__ == '__main__':
    tf.app.run()
