import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F
from nn.model import DNN


n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
eta = 1e-3
reg_lambda = 0
lengths = (n_features, 512, 512, n_classes)


def load_mnist(mode='train', n_samples=None):
    images = './train-images-idx3-ubyte' if mode == 'train' else './t10k-images-idx3-ubyte'
    labels = './train-labels-idx1-ubyte' if mode == 'train' else './t10k-labels-idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape((length, 28, 28)).astype(np.int32)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape((length)).astype(np.int32)
    return (X[:n_samples].reshape(n_samples, -1), y[:n_samples]) if n_samples is not None else (X.reshape(length, -1), y)


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = DNN(lengths)
    optimizer = nn.optim.SGD(momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            delta = optimizer.step(probs - np.eye(n_classes)[y])
            model.backward(delta, eta, reg_lambda)
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f}')  # loss={criterion(probs, y):.3f}')

        for X, y in testloader:
            preds = model.predict(X)
            print(f'test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

        # if i % 5 == 0:
        #     X, y = load_mnist('test', 20)
        #     pred = model.predict(X)
        #     fig = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')[1].flatten()
        #     for i in range(20):
        #         img = X[i].reshape(28, 28)
        #         fig[i].set_title(pred[i])
        #         fig[i].imshow(img, cmap='Greys', interpolation='nearest')
        #     fig[0].set_xticks([])
        #     fig[0].set_yticks([])
        #     plt.tight_layout()
        #     plt.savefig("vis.png")
        #     plt.show()


if __name__ == '__main__':
    main()
