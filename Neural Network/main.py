from Load_Data import load_data
from Training import train


def run(mnist, mode):
    train(mnist, mode)


def app():
    print('Loading Data ......')
    mnist = load_data()
    print('Training Neural Network ......')
    print('Mode Selection')
    print("""There are a few neural networks we can choose.
          Mode 1: Simple Neural Network  -> SNN
          Mode 2: Convolutional Neural Network -> CNN
          Mode 3: AutoEncoder -> AE""")
    # input_mode = input('Enter the mode:(ex -> SNN)\n')

    run(mnist, 'AE')


if __name__ == '__main__':
    app()
