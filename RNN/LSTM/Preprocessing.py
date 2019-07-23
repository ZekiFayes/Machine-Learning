from Lib import *


def load_data():

    filename = 'Dataset/'
    mnist = input_data.read_data_sets(filename, one_hot=True)

    return mnist
