import imageio


def generate():

    image = []

    for i in range(1, 26):
        image.append(imageio.imread('Figure/fig' + str(int(i)) + '.png'))
    imageio.mimsave('Figure/wb.gif', image, fps=2)


def main(argv=None):
    generate()


if __name__ == '__main__':
    main()
