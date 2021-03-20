import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LogisticRegression(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ----------
    eta : float, (between 0.0 and 1.0)
        Learning rate
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.
    """

    def __init__(self, eta=0.01, C=1, n_iter=1000, random_state=None, lr_decay=None):
        self.eta = eta
        self.C = C
        self.n_iter = n_iter
        self.random_state = random_state
        self.lr_decay = lr_decay
        self.base_lr = eta

    def fit(self, X, y):
        """Fit training data."""
        if self.random_state == None:
            self.w_ = np.zeros((1 + X.shape[1]))
        else:                                           # random weight initialization
            rgen = np.random.RandomState(self.random_state)
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        # self.cost_ = []
        # self.acc_ = []
        # self.lr_ = []
        # self.max_acc = 0

        for i in tqdm(range(self.n_iter)):
            net_input = self._net_input(X)
            output = self._activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * (X.T.dot(errors) - 1. / self.C * self.w_[1:])
            self.w_[0] += self.eta * errors.sum()
            y_pred = self.predict(X)

            if self.lr_decay == 'exp':
                self.eta = self.base_lr * (0.99 ** i)
            elif self.lr_decay == 'step' and i % 160 == 0:
                self.eta *= 0.8

            # cost = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output + 1e-4)))
            # acc = accuracy_score(y, y_pred)
            # self.lr_.append(self.eta)
            # self.cost_.append(cost)
            # self.acc_.append(acc)
            # if acc > self.max_acc:
            #     self.max_acc = acc
        return self

    def _net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self._net_input(X) >= 0, 1, 0)
