from Lib import *
''' sklearn '''


def run(mode=None, filename=None):
    optimization_and_recommendation(mode, filename)


def app(mode=None, Name=None):
    print('This app is to demonstrate how each predictive model works.')
    print('We use Sklearn to train model and use accuracy as an assessment.')
    print('There are some predictive models.')
    print('Perceptron, Logistic Regression, SVM_Kernel_Linear, '
          'SVM_Kernel_rbf, Decision Tree, Random Forest, K-Neighbors')
    print('It will output the best model.')
    run(mode, Name)


if __name__ == '__main__':

    # Optimization / Recommendation
    option = 'Recommendation'
    data_name = None
    app(option, data_name)
