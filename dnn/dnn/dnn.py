import numpy as np

from . import nn
from . import functional as F


class Linear(nn.Module):

    def __init__(self, in_length: int, out_length: int) -> None:
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(out_length, in_length + 1))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w[:, 1:].T) + self.w[:, 0]        # (B, Din) x (Din, Dout) + (Dout) => (B, Dout);

    def backward(self, delta, eta, reg_lambda):
        delta_ = np.dot(delta, self.w[:, 1:])                   # (B, Dout) x (Dout, Din)
        self.w[:, 1:] *= 1 - eta * reg_lambda
        self.w[:, 1:] -= eta * np.dot(delta.T, self.x)          # (Dout, Din) - (Dout, B) x (B, Din)
        self.w[:, 0] -= eta * np.sum(delta, axis=0)             # (Dout) - ((B, Dout) => (Dout))
        return delta_


class BatchNorm1d(nn.Module):

    def __init__(self, length: int, momentum: float=0.9) -> None:
        self.running_mean = np.zeros((length,))
        self.running_var = np.zeros((length,))
        self.gamma = np.ones((length,))
        self.beta = np.zeros((length,))
        self.momentum = momentum

    def forward(self, x):
        self.mean = np.mean(x, axis = 0)
        self.var = np.var(x, axis = 0)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        self.x = (x - self.mean) / np.sqrt(self.var ** 2 + 1e-5)
        return self.gamma * self.x + self.beta

    def backward(self, delta, eta, **kwargs):
        ...


class DNN(nn.Module):

    def __init__(self, lengths: list, activation: str='RelU') -> None:
        Activation = F.get_attribute(activation)
        self.layers = []
        for i in range(len(lengths) - 1):
            self.layers.append(Linear(lengths[i], lengths[i + 1]))
            self.layers.append(BatchNorm1d(lengths[i + 1]))
            self.layers.append(Activation() if i != len(lengths) - 2 else F.Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta, eta, reg_lambda=0):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, eta, reg_lambda)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


if __name__ == '__main__':
    nn = DNN((2, 4, 4, 2))
    X = np.zeros((12, 2))
    y = np.zeros((12, 2))
    pred = nn.forward(X)
    nn.backward(y - pred, 0.1)
