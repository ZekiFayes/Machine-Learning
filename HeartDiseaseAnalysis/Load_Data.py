from Libs import *


def load_data():

    filename = 'DataSet/heart.csv'
    df = pd.read_csv(filename)

    return df
