from Train import train
from Lib import *


def run():

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':

    run()
