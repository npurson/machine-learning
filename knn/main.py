import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn


train_images = './train-images.idx3-ubyte'
train_labels = './train-labels.idx1-ubyte'
test_images = './t10k-images.idx3-ubyte'
test_labels = './t10k-labels.idx1-ubyte'


def load_mnist():
    X_train = np.fromfile(open(train_images), np.uint8)
    X_train = X_train[16:].reshape((60000, 28, 28)).astype(np.int)
    y_train = np.fromfile(open(train_labels), np.uint8)
    y_train = y_train[8:].reshape((60000)).astype(np.int)
    X_test = np.fromfile(open(test_images), np.uint8)[16:]
    X_test = X_test.reshape((10000, 28, 28)).astype(np.int)
    y_test = np.fromfile(open(test_labels), np.uint8)[8:]
    y_test = y_test.reshape((10000)).astype(np.int)
    return X_train[:40000], y_train[:40000], X_test[:200], y_test[:200]


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    # data binarization
    # for i in tqdm(range(len(x_train))):
    #     for j in range(28):
    #         for k in range(28):
    #             x_train[i][j][k] = 1 if x_train[i][j][k] > 177 else 0
    # for i in tqdm(range(len(x_test))):
    #     for j in range(28):
    #         x_test[i][j].squeeze()
    #         for k in range(28):
    #             x_test[i][j][k] = 1 if x_test[i][j][k] > 177 else 0

    # plot data samples
    # plot = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')[1].flatten()
    # for i in range(20):
    #     img = x_train[i]
    #     plot[i].set_title(y_train[i])
    #     plot[i].imshow(img, cmap='Greys', interpolation='nearest')
    # plot[0].set_xticks([])
    # plot[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()

    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')[1].flatten()
    for i in range(20):
        img = X_test[i]
        fig[i].set_title(y_pred[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
