import math
import numpy as np


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


class FCLayer(object):

    def __init__(self, in_length, out_length):
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=(out_length, in_length + 1))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x_ = x
        return np.dot(x, self.w_[:, 1:].T) + self.w_[:, 0]      # (B, Din) x (Din, Dout) + (Dout) => (B, Dout);

    def backward(self, delta, eta, reg_lambda):
        self.w_[:, 1:] -= eta * np.dot(delta.T, self.x_)        # (Dout, Din) - (Dout, B) x (B, Din)
        self.w_[:, 0] -= eta * np.sum(delta, axis=0)            # (Dout, B) => (Dout, 1)
        return np.dot(delta, self.w_[:, 1:])


class BPNN(object):

    def __init__(self, lengths: list):
        self.w1 = FCLayer(lengths[0], lengths[1])
        self.w2 = FCLayer(lengths[1], lengths[2])
        self.w3 = FCLayer(lengths[2], lengths[3])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x1_ = x
        p = np.tanh(self.w1(x))
        self.x2_ = p
        p = np.tanh(self.w2(p))
        self.x3_ = p
        p = softmax(self.w3(p))
        return p

    def backward(self, delta, eta, reg_lambda=None):
        delta = self.w3.backward(delta, eta, reg_lambda)
        delta *= (1 - self.x3_ ** 2)
        delta = self.w2.backward(delta, eta, reg_lambda)
        delta *= (1 - self.x2_ ** 2)
        delta = self.w1.backward(delta, eta, reg_lambda)

    def predict(self, x):
        return np.argmax(self.forward(x))


if __name__ == '__main__':
    nn = BPNN((2, 4, 4, 2))
    X = np.zeros((12, 2))
    y = np.zeros((12, 2))
    pred = nn.forward(X)
    nn.backward(y - pred, 0.1)
