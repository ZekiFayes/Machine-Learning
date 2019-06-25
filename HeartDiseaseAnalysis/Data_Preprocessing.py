from Libs import *


def preprocessing(df):

    columns = list(df.columns)

    names = []
    for column in columns:

        if len(np.unique(df[column])) > 10:
            continue
        else:
            names.append(column)

    return df[names]
