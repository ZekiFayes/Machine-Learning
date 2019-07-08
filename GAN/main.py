from Train import train
from Lib import *


def main(argv=None):
    mnist = input_data.read_data_sets("Dataset/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
