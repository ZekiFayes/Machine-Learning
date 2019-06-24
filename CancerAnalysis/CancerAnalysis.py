"""
This is to do something about the cancer data.
The data comes from Kaggle.
We try to find some relationships inside the data, and
build a predictive model, if possible.

Statistic Analysis: plotting data distribution.
Machine Learning Analysis:
1, Training each classifier
2, observe the accuracy of each classifier with the parameter varying
3, plotting Learning curve and validation curve
4, grid search optimization

"""

from CA_Lib import *
from StatisticalAnalysis import statistic_analysis
from MLAnalysis import ml_analysis


def run():

    print('Loading Data >>> ......')
    filename = 'dataSet/cancer.csv'
    df = pd.read_csv(filename)

    X = df[columns]
    y = df['diagnosis']

    print('Finished!')
    print('Starting Statistic Analysis >>> ......')
    statistic_analysis(X, y)
    print('Finished!')

    print('Starting Machine Learning Analysis >>> ......')
    ml_analysis(X, y)
    print('Finished!')


def app():
    run()


if __name__ == '__main__':
    app()
