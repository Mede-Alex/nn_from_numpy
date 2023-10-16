import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


def read_sample(n):
    PATH_IMAGE = 'data/train-images-idx3-ubyte'
    PATH_LABEL = 'data/train-labels-idx1-ubyte'

    imgs = idx2numpy.convert_from_file(PATH_IMAGE)
    labels = idx2numpy.convert_from_file(PATH_LABEL)
    
    imgs = imgs / 255    
    
    return imgs[:n], labels[:n]


def display_img(img: np.ndarray, label: np.uint8) -> None:
    plt.figure()
    plt.imshow(img, cmap='grey')
    plt.axis(False)
    plt.title(f'Label: {label}')
    plt.savefig('sample.jpg')


NO_SAMPLES = 1000

imgs, labels = read_sample(NO_SAMPLES)


display_img(imgs[2], labels[2])
