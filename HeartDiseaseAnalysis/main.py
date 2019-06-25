from HeartDiseaseAnalysis import analysis
from Data_Preprocessing import preprocessing
from Load_Data import load_data


def run(df):
    analysis(df)


def app():

    print('Loading Data ......')
    df = load_data()

    print('Preprocessing data ......')
    data = preprocessing(df)

    print('Analysing ......')
    run(data)


if __name__ == '__main__':
    app()
