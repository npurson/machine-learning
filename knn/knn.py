import numpy as np
from tqdm import tqdm


class Knn(object):
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, xs):
        y_pred = []
        for x in tqdm(xs):
            dist = [np.sum((xi - x) ** 2) for xi in self.x]
            topk = self.y[np.argsort(dist)[:self.k]]            # the top k nearest ys
            y_pred.append(np.argmax(np.bincount(topk)))         # append the most frequent label in topk
        return y_pred
