from Lib import *


def load_data():

    filename = "DataSet/"
    mnist = input_data.read_data_sets(filename, one_hot=True)

    return mnist
