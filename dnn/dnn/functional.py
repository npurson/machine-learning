import numpy as np

from . import nn


class Softmax(nn.Module):
    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)


class Sigmoid(nn.Module):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, delta, eta, **kwargs):
        return self.y * (1 - self.y)


class Tanh(nn.Module):
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, delta, eta, **kwargs):
        return delta * (1 - self.x ** 2)


class ReLU(nn.Module):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta, eta, **kwargs):
        return np.where(self.x > 0, self.x, 0)


class LeakyReLU(nn.Module):
    def forward(self, x):
        self.x = x
        ...

    def backward(self, delta, eta, **kwargs):
        ...


class ELU(nn.Module):
    def forward(self, x):
        self.x = x
        ...

    def backward(self, delta, eta, **kwargs):
        ...


class CrossEntropyLoss(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, preds, targets):
        return -np.sum(np.eye(self.n_classes)[targets] * np.eye(self.n_classes)[preds])
