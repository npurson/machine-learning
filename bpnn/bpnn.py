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
        delta_ = np.dot(delta, self.w_[:, 1:])                  # (B, Dout) x (Dout, Din)
        self.w_[:, 1:] *= 1 - eta * reg_lambda
        self.w_[:, 1:] -= eta * np.dot(delta.T, self.x_)        # (Dout, Din) - (Dout, B) x (B, Din)
        self.w_[:, 0] -= eta * np.sum(delta, axis=0)            # (Dout) - ((B, Dout) => (Dout))
        return delta_


class BPNN(object):

    def __init__(self, lengths: list):
        self.layers = [FCLayer(lengths[i], lengths[i + 1]) for i in range(len(lengths) - 1)]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.xs = []
        for i, layer in enumerate(self.layers):
            self.xs.append(x)
            x = layer(x)
            x = np.tanh(x) if i != len(self.layers) - 1 else softmax(x)
        return x

    def backward(self, delta, eta, reg_lambda=0):
        for i, layer in enumerate(reversed(self.layers)):
            delta = delta * (1 - self.xs[-i] ** 2) if i else delta
            delta = layer.backward(delta, eta, reg_lambda)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


if __name__ == '__main__':
    nn = BPNN((2, 4, 4, 2))
    X = np.zeros((12, 2))
    y = np.zeros((12, 2))
    pred = nn.forward(X)
    nn.backward(y - pred, 0.1)
