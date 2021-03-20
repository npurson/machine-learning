import math
import numpy as np


def softmax(x):
    exps = np.array([math.exp(xi) for xi in x])
    return exps / np.sum(exps)


class FCLayer(object):

    def __init__(self, in_length, out_length):
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=(out_length, 1 + in_length))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x_ = x
        return np.dot(x, (self.w_[:, 1:] + self.w_[:, 0]).T)

    def backward(self, delta):
        self.w_[:, 1:] -= np.dot(self.x_, delta)
        self.w_[:, 0] -= delta
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

    def backward(self, delta):
        delta = self.w3.backward(delta)
        delta *= (1 - self.x3_ ** 2)
        delta = self.w2.backward(delta)
        delta *= (1 - self.x2_ ** 2)
        delta = self.w1.backward(delta)


    def predict(self, x):
        return np.argmax(self.forward(x))


if __name__ == '__main__':
    ...
