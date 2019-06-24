from CA_Lib import *


def statistic_analysis(X, y):

    data = X
    df = {}

    for label in np.unique(y):
        df[label] = data[y == label]

    print('Plotting data distribution')
    cols = 4
    rows = np.ceil(len(columns) / cols)

    # plot data distribution
    fig = plt.figure()
    for i, column in enumerate(columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        for key in df:
            ax.hist(df[key][column], bins=30, label=key)
            plt.xlabel(column)
            plt.legend()

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    # print('Plotting Correlation Map')
    # # plot correlation
    # cm = np.corrcoef(data.values.T)
    # sns.set(font_scale=0.8)
    # _ = sns.heatmap(cm,
    #                 cbar=True,
    #                 annot=True,
    #                 square=True,
    #                 fmt='.2f',
    #                 annot_kws={'size': 9},
    #                 yticklabels=columns,
    #                 xticklabels=columns)
    # plt.title('Correlation')
    # plt.show()
