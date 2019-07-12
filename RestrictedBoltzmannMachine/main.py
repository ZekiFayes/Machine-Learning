from Lib import *
from Train import train
from LoadData import load_data


def main(argv=None):

    new_train_set, new_train_mask = load_data()
    print('Loading data finished!')
    train(new_train_set, new_train_mask)


if __name__ == "__main__":

    tf.app.run()
