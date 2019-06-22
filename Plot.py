from Lib import *


def plot_decision_regions(X, y, classifer, resolution=0.02, test_idx=None):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    zz = classifer.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    zz = zz.reshape(xx1.shape)

    plt.contourf(xx1, xx2, zz, alpha=0.4, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=cl)

    if test_idx:
        x_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1], c='gray',
                    alpha=0.2, linewidths=1, marker='^',
                    s=55, label='test set')


def plot(x, y, classifer, test_idx, label, title):
    plot_decision_regions(x, y, classifer, resolution=0.02, test_idx=test_idx)
    # plt.xlabel(label['x'])
    # plt.ylabel(label['y'])
    plt.legend()
    plt.title(title)
