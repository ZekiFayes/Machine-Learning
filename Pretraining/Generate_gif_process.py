import imageio


def generate():

    image = []

    for i in range(1, 4):
        image.append(imageio.imread('Figure/fig' + str(int(i)) + '.png'))
    imageio.mimsave('Figure/wb1.gif', image)


def main(argv=None):
    generate()


if __name__ == '__main__':
    main()
