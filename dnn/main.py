import numpy as np
from matplotlib import pyplot as plt

from dnn import DNN
from dnn import functional as F


n_samples = 100
n_features = 2
n_classes = 2
n_iters = 300
eta = 5e-2
reg_lambda = 0
lengths = (n_features, 100, 100, n_classes)


def main():
    X, y = []
    nn = DNN(lengths)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_iters):
        probs = nn.forward(X)
        nn.backward(probs - np.eye(n_classes)[y], eta, reg_lambda)
        preds = np.argmax(probs, axis=1)
        print(f'iter {i}: acc={np.sum(preds == y) / len(y)}, loss={criterion(preds, y)}')


if __name__ == '__main__':
    main()
