from Lib import *
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def train(x, y, clf):
    clf.fit(x, y)


def calculate_accuracy(x_train, x_test, y_train, y_test, clf):

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred)


def optimize_classifier(option, x_train, x_test, y_train, y_test):

    accuracy = []

    if option == 'Perceptron':
        eta = [0.00001, 0.0001, 0.001, 0.01, 0.1,  0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for eta0 in eta:
            ppn = Perceptron(n_iter=40, eta0=eta0, random_state=0)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, ppn)
            accuracy.append(acc)
        return eta, accuracy

    elif option == 'Logistic_Regression':
        C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        for c in C:
            lr = LogisticRegression(C=c, random_state=0)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, lr)
            accuracy.append(acc)
        return C, accuracy

    elif option == 'SVM_Kernel_Linear':
        C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        for c in C:
            svm = SVC(kernel='linear', C=c, random_state=0)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, svm)
            accuracy.append(acc)
        return C, accuracy

    elif option == 'SVM_Kernel_rbf':
        C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        for c in C:
            svm = SVC(kernel='rbf', C=c, random_state=0)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, svm)
            accuracy.append(acc)
        return C, accuracy

    elif option == 'Decision Tree':
        depth = np.arange(1, 15)
        for max_depth in depth:
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, tree)
            accuracy.append(acc)
        return depth, accuracy

    elif option == 'Random Forest':
        n_estimator = np.arange(1, 20)
        for n in n_estimator:
            forest = RandomForestClassifier(criterion='entropy', n_estimators=n, random_state=1, n_jobs=2)
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, forest)
            accuracy.append(acc)
        return n_estimator, accuracy

    elif option == 'K-Neighbors':
        K = np.arange(1, 20)
        for k in K:
            knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
            acc = calculate_accuracy(x_train, x_test, y_train, y_test, knn)
            accuracy.append(acc)
        return K, accuracy
