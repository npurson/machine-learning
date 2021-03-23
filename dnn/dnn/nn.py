import numpy as np


class Module(object):
    def __init__(self):
        ...

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ...

    def backward(self, delta, eta, **kwargs):
        return delta
