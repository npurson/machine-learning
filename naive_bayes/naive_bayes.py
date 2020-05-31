import numpy as np


class NaiveBayes(object):

    def __init__(self, alpha=1.0):
        """Inits NaiveBayes class with Laplace smoothing parameter alpha.
        """
        self.alpha = alpha

    def fit(self, X, y):
        self.X = np.asanyarray(X)
        self.y = np.asanyarray(y)
        self.classes = np.unique(y)

        n_features = self.X.shape[1]
        n_classes = self.classes.shape[0]

        self.feature_count = np.full((n_classes, n_features), self.alpha)       # Laplace smoothing
        self.class_count = np.full((n_classes), n_classes)                      #

        for xi, _ in enumerate(self.X):
            self.feature_count[y[xi]] += X[xi]
            self.class_count[y[xi]] += 1

        for _, cls in enumerate(self.classes):
            self.feature_count[cls] /= self.class_count[cls]

    def predict(self, X):
        X_pred = np.asanyarray(X)
        pred = []
        for xi in X_pred:
            prob = []
            for _, cls in enumerate(self.classes):
                prob_prior = np.log(self.class_count[cls] / np.sum(self.class_count))
                prob_post = np.sum(np.log(self.feature_count[cls][xi == 1]))
                prob.append(prob_prior + prob_post)
            pred.append(self.classes[np.argmax(prob)])
        return pred


class GaussianNaiveBayes(NaiveBayes):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = np.asanyarray(X)
        self.y = np.asanyarray(y)
        self.classes = np.unique(y)

        n_features = self.X.shape[1]
        n_classes = self.classes.shape[0]

        self.mean = np.zero((n_classes, n_features))
        self.var = np.zero((n_classes, n_features))
        self.prob_prior = np.zeros((n_classes))

        for i, cls in enumerate(self.classes):
            xi = self.X[np.where(y == cls)]
            self.mean[i] = np.mean(xi, axis=0, keepdims=True)
            self.var[i] = np.var(xi, axis=0, keepdims=True)
            self.prob_prior[cls] = xi.shape[0] / self.X.shape[0]

    def predict(self, X):
        X_pred = np.asanyarray(X)
        pred = []
        for xi in X_pred:
            prob = []
            for _, cls in enumerate(self.classes):
                prob_prior = np.log(self.prob_prior[cls])
                prob_post = self._gaussian_prob_density(xi, cls)
                prob.append(prob_prior + prob_post)
            pred.append(self.classes[np.argmax(prob)])
        return pred

    def _gaussian_prob_density(self, X, cls):
        """Compute the probability density function of 1d Gaussian distribution.
        """
        eps = 1e-5                                                  # prevent denominator from being 0
        mean = self.mean[cls]
        var = self.var[cls]

        result = (np.exp(-(X - mean) ** 2 / (2 * var + eps))
                  / np.sqrt(2 * np.pi * var + eps))                 # probability density function of 1d Gaussian distribution
        result = np.sum(np.log(result), axis=1, keepdims=True)      # log(P(X|y)) = log(P(x1|y)) + log(P(x2|y)) + ...
        return result
