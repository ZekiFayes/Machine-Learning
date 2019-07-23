from Lib import *
from Preprocessing import load_data
from Train import train


def main(argv=None):

    print('load data')
    mnist = load_data()
    print('Finished!')
    print('Start training!')
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
