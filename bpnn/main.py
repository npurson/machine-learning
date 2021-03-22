import numpy as np
import sklearn.datasets
from matplotlib import pyplot as plt

from bpnn import BPNN


n_samples = 100
eta = 5e-2
reg_lambda = 0
n_iters = 300
lengths = (2, 100, 100, 2)


def CrossEntropyLoss(preds, targets, num_cls=2):
    return -np.sum(np.eye(num_cls)[targets] * np.eye(num_cls)[preds])


def plot_clf(X, y, clf):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    preds = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.contour(xx, yy, preds.reshape(xx.shape), levels=[0], colors='b')
    plt.show()
    return


def main():
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1)

    clf = BPNN(lengths)
    criterion = CrossEntropyLoss

    for i in range(n_iters):
        probs = clf.forward(X)
        clf.backward(probs - np.eye(2)[y], eta, reg_lambda)
        preds = np.argmax(probs, axis=1)
        print(f'iter {i}: acc={np.sum(preds == y) / len(y)}, loss={criterion(preds, y)}')
    plot_clf(X, y, clf)


if __name__ == '__main__':
    main()
