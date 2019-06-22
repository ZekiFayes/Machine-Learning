from Lib import *


def optimization_and_recommendation(mode=None, filename=None):

    # load data
    x_train, x_test, y_train, y_test = load_split_standardize_data(
        filename=filename, test_size=0.3, random_state=0)

    if mode == 'Recommendation':
        x_combined = np.vstack((x_train, x_test))
        y_combined = np.hstack((y_train, y_test))

    # model name
    options = ['Perceptron', 'Logistic_Regression', 'SVM_Kernel_Linear',
               'SVM_Kernel_rbf', 'Decision Tree', 'Random Forest', 'K-Neighbors']

    # store the best parameters for each predictive model
    model_param_n_accuracy = {}

    # visualize the relationship between parameter and accuracy for each model
    if mode == 'Optimization':
        fig = plt.figure()
        cols = 4
        rows = np.ceil(len(options) / cols)
        ith_fig = 0

        print('Optimizing...')

    for option in options:
        params, acc = optimize_classifier(option, x_train, x_test, y_train, y_test)

        if mode == 'Optimization':
            ith_fig += 1
            ax = fig.add_subplot(rows, cols, ith_fig)
            ax.plot(params, acc, marker='o')
            plt.title(option)
            plt.ylabel('Accuracy')

        model_param_n_accuracy[option] = [params[acc.index(max(acc))], max(acc)]

    print('Finished!')

    if mode == 'Optimization':
        plt.show()

    if mode == 'Recommendation':

        ppn = Perceptron(n_iter=40,
                         eta0=model_param_n_accuracy['Perceptron'][0],
                         random_state=0)
        lr = LogisticRegression(C=model_param_n_accuracy['Logistic_Regression'][0],
                                random_state=0)
        svm_linear = SVC(kernel='linear',
                         C=model_param_n_accuracy['SVM_Kernel_Linear'][0],
                         random_state=0)
        svm_rbf = SVC(kernel='rbf',
                      C=model_param_n_accuracy['SVM_Kernel_rbf'][0],
                      random_state=0)
        tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=model_param_n_accuracy['Decision Tree'][0],
                                      random_state=0)
        forest = RandomForestClassifier(criterion='entropy',
                                        n_estimators=model_param_n_accuracy['Random Forest'][0],
                                        random_state=1,
                                        n_jobs=2)
        knn = KNeighborsClassifier(n_neighbors=model_param_n_accuracy['K-Neighbors'][0],
                                   p=2,
                                   metric='minkowski')

        clfs = [ppn, lr, svm_linear, svm_rbf, tree, forest, knn]

        fig = plt.figure()
        cols = 4
        rows = np.ceil(len(options) / cols)
        ith_fig = 0

        for option, clf in zip(options, clfs):
            train(x_train, y_train, clf)

            # visualize the results
            plot_labels = {'x': 'petal length',
                           'y': 'petal width'}
            plot_title = option

            ith_fig += 1
            _ = fig.add_subplot(rows, cols, ith_fig)
            plot(x_combined, y_combined, classifer=clf,
                 test_idx=range(105, 150), label=plot_labels, title=plot_title)

        plt.show()

    df = DataFrame(model_param_n_accuracy, index=['Parameter', 'Accuracy'])
    print(df.T)
