from Lib import *


def converter():

    # initialization
    data = {}
    max_user_id, min_user_id = 0, 1000000
    max_movie_id, min_movie_id = 0, 1000000

    # load data
    print('Starting data conversion')
    with open("Dataset/ratings.dat", "r") as fin:

        for line in fin:
            lyst = [int(x) for x in line.split('::')]
            user, movie, rate = lyst[0], lyst[1], lyst[2]

            max_user_id = max(max_user_id, user)
            min_user_id = min(min_user_id, user)
            max_movie_id = max(max_movie_id, movie)
            min_movie_id = min(min_movie_id, movie)

            if user not in data:
                data[user] = [(movie, rate)]
            else:
                data[user].append((movie, rate))

    if min_user_id == 1:
        max_user_id = max_user_id - 1
        min_user_id = min_user_id - 1

    if min_movie_id == 1:
        max_movie_id = max_movie_id - 1
        min_movie_id = min_movie_id - 1

    train_set = np.zeros((max_user_id + 1, max_movie_id + 1))

    for k, v in data.items():
        for m, r in v:
            train_set[k - 1][m - 1] = r

    with open("Dataset/data.pkl", "wb") as fout:
        pickle.dump((min_user_id, max_user_id, min_movie_id, max_movie_id, train_set), fout)

    print('Data conversion finished!')
    print('Data stored in data.pkl')
    print('---------------------------------------------')
    print('Basic information:')
    print('min movie ID:', min_movie_id)
    print('max movie ID:', max_movie_id)
    print('min user ID:', min_user_id)
    print('max user ID:', max_user_id)
    print('---------------------------------------------')


def load_data():

    converter()

    print('Loading data')
    with open("Dataset/data.pkl", "rb") as fin:
        min_user_id, max_user_id, min_movie_id, max_movie_id, train_set = pickle.load(fin)

    rows, cols = train_set.shape
    new_train_set = np.zeros((rows, cols * 5))
    new_train_mask = np.zeros((rows, cols * 5))

    for row in range(rows):
        for col in range(cols):
            r = int(train_set[row][col])
            if r == 0:
                continue
            new_train_set[row][col * 5 + r - 1] = 1
            new_train_mask[row][col * 5:col * 5 + 5] = 1

    return new_train_set, new_train_mask
