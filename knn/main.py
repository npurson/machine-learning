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
    x_train = np.fromfile(open(train_images), np.uint8)
    x_train = x_train[16:].reshape((60000, 28, 28)).astype(np.int)
    y_train = np.fromfile(open(train_labels), np.uint8)
    y_train = y_train[8:].reshape((60000)).astype(np.int)
    x_test = np.fromfile(open(test_images), np.uint8)[16:]
    x_test = x_test.reshape((10000, 28, 28)).astype(np.int)
    y_test = np.fromfile(open(test_labels), np.uint8)[8:]
    y_test = y_test.reshape((10000)).astype(np.int)
    return x_train[:10000], y_train[:10000], x_test[:200], y_test[:200]

def main():
    x_train, y_train, x_test, y_test = load_mnist()

    # binarization
    # for i in tqdm(range(len(x_train))):
    #     for j in range(28):
    #         for k in range(28):
    #             x_train[i][j][k] = 1 if x_train[i][j][k] > 177 else 0
    # for i in tqdm(range(len(x_test))):
    #     for j in range(28):
    #         x_test[i][j].squeeze()
    #         for k in range(28):
    #             x_test[i][j][k] = 1 if x_test[i][j][k] > 177 else 0
    
    # plot
    # ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')[1].flatten()
    # for i in range(10):
    #     img = x_train[i]
    #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()

    knn = Knn()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    correct = sum((y_test - y_pred) == 0)
    print('==> correct:', correct)
    print('==> total:', len(x_test))
    print('==> acc:', correct / len(x_test))
    os.system("pause")

if __name__ == '__main__':
    main()
