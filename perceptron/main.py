import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron


def gen_data(num):
    x1 = np.random.multivariate_normal([-0.5, -2], [[1, .75], [.75, 1]], num)
    x2 = np.random.multivariate_normal([0.5, 2], [[1, .75], [.75, 1]], num)
    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(num), np.ones(num)))
    return X, y


def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()


def plot_clf(clf, X, y):
    xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(xy)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='black')
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()


def main():
    num_epochs = 10
    X, y = gen_data(100)
    plot_data(X, y)

    ppn = Perceptron(eta=1e-3)
    accs = []
    losses = []
    for i in range(num_epochs):
        acc, loss = ppn.fit(X, y)
        plot_clf(ppn, X, y)
        accs.append(acc)
        losses.append(loss)
    plot_clf(ppn, X, y)

    plt.plot(range(1, len(accs) + 1), accs, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.show()

    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()
