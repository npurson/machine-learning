import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassification (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, planes=2, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        rgen = np.random.RandomState(random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + planes)
        self.iter=0


    def fit(self, X, y):
        loss = 0
        acc = 0
        for xi, target in zip(X, y):
            self.iter += 1
            update = self.eta * (target - self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update
            acc += int(update == 0.0)
            loss += abs(np.sum(update))
        return acc / len(X), loss

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def plot_clf(self, X, y):
        xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
        xy = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(xy)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors='black')
        plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
        plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
        plt.savefig(str(self.iter // 5) + '.jpg')
        plt.show()
