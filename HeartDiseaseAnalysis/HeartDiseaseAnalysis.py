from Libs import *


def correlation_plot(df):

    cols = df.columns

    cm = np.corrcoef(df.values.T)
    sns.set(font_scale=0.8)
    _ = sns.heatmap(cm, cbar=True,
                    annot=True,
                    square=True,
                    fmt='.2f',
                    annot_kws={'size': 7},
                    yticklabels=cols,
                    xticklabels=cols)
    plt.show()


def hist_plot(df):

    names = df.columns

    length = len(names)
    cols = 4
    rows = np.ceil(length / cols)
    fig = plt.figure()

    for ith_fig, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, ith_fig + 1)
        num_bins = len(np.unique(df[name]))
        df[name].hist(bins=num_bins, ax=ax)

        plt.ylabel('Amount')
        plt.title('Amount(' + name + ')')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def hist_plot_given_target(df):

    names = list(df.columns)
    names.remove('target')

    length = len(names)
    cols = 4
    rows = np.ceil(length / cols)
    fig = plt.figure()

    for ith_fig, column in enumerate(names):
        counts = pd.crosstab(df['target'], df[column])

        for i in range(len(np.unique(df['target']))):
            counts.ix[i] = counts.ix[i] / counts.ix[i].sum()

        ax = fig.add_subplot(rows, cols, ith_fig + 1)
        counts.plot(kind='bar', stacked=True, rot=0, ax=ax, sharey=True)
        plt.ylabel('Portion')
        plt.title('p(' + column + '|target)')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def hist_plot_given_sex_n_target(df):

    names = list(df.columns)
    names.remove('sex')
    names.remove('target')

    length = len(names)
    cols = 4
    rows = np.ceil(length / cols)
    fig = plt.figure()

    for i, item in enumerate(names):
        column = ['target', 'sex', item]
        data = df.groupby(column).size().unstack().fillna(0) / 1.0

        for m in range(2):
            for n in range(2):
                data.ix[m].ix[n] = data.ix[m].ix[n] / data.ix[m].ix[n].sum()

        ax = fig.add_subplot(rows, cols, i + 1)
        data.plot(kind='bar', stacked=True, rot=0, ax=ax, sharey=True)
        plt.xlabel(('target', 'sex'))
        plt.ylabel('Portion')
        plt.title('p(' + item + '|target, sex)')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def calculate_prob(prob, names, variable):

    p1 = 1
    p2 = 1

    if variable == [] or len(variable) < 9:
        sex = np.random.randint(2)
        cp = np.random.randint(4)
        fbs = np.random.randint(2)
        restecg = np.random.randint(3)
        exang = np.random.randint(2)
        slope = np.random.randint(3)
        ca = np.random.randint(5)
        thal = np.random.randint(4)
        target = np.random.randint(2)
        variable = [sex, cp, fbs, restecg, exang, slope, ca, thal, target]
    else:
        [sex, cp, fbs, restecg, exang, slope, ca, thal, target] = variable

    for v, name in zip(variable, names):
        p1 *= prob[name + '|target'][target][v]
        p2 *= prob[name][v]

    p_hd = p1 * prob['target'][target] / p2

    print('p(target=%d|sex=%d, cp=%d, fbs=%d, restecg=%d, exang=%d, slope=%d, ca=%d, thal=%d) = %f'
          % (target, sex, cp, fbs, restecg, exang, slope, ca, thal, p_hd))


def train(df, names):

    prob = {}
    tot_num = df.shape[0]

    for name in names:
        # calculate p(variable)
        prob[name] = list(df.groupby(name).size() / tot_num)

        if name == 'target':
            continue
        else:
            # calculate p(variable|target)
            p = pd.crosstab(df['target'], df[name])
            for i in range(len(np.unique(df['target']))):
                p.ix[i] = p.ix[i] / p.ix[i].sum()
            prob[name+'|target'] = np.array(p)

    return prob


def probabilistic_graph_model(df):

    names = list(df.columns)

    print('-----------------------------------------------------------------------------------------')
    print('Graphical Model Node:')
    print(names)
    print('Assuming that each variable is independent. '
          'For each variable in each node, the possible values of each variable and its probability:')
    for name in names:
        print(name, '=', np.unique(df[name]))

    print('''Naive Bayes:
                                  p(sex,...|target)p(target)
          p(target|sex, ...)  =  ----------------------------
                                         p(sex,...)''')
    print('-----------------------------------------------------------------------------------------')

    prob = train(df, names)
    names.remove('target')

    for i in range(10):
        calculate_prob(prob, names, list(df.ix[i]))


def analysis(csv_data):

    df = csv_data

    print('Plotting Correlation Matrix ......')
    correlation_plot(df)

    print('Plotting Histgram ......')
    hist_plot(df)

    print('Plotting Histgram given target ......')
    print('Target = {0, 1}, 0 = have not disease, 1 = have disease')
    hist_plot_given_target(df)
    hist_plot_given_sex_n_target(df)

    print('Starting Graphical Model ......')
    probabilistic_graph_model(df)
