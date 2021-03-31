import numpy as np

from .modules import Module


class Softmax(Module):

    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)


class CrossEntropyLoss(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, preds, targets):
        return -np.sum(np.eye(self.n_classes)[targets] * np.eye(self.n_classes)[preds])
