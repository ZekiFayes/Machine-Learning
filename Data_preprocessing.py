from Lib import *


def load_split_standardize_data(filename=None, test_size=0.3, random_state=None):

    if filename != None:
        # load data
        df = pd.read_csv(filename)
        x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    else:
        iris = datasets.load_iris()
        x, y = iris.data[:, [2, 3]], iris.target

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    # standardize data
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test

