import numpy as np

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.base import BaseEstimator, ClassifierMixin

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=1e3):
        self.n_iter = n_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y     
        
        self.weights_ = np.random.rand((X.shape[1]))
        
        for i in range(int(self.n_iter)):
            for j in range(0, self.weights_.shape[0]):
                x = np.random.randint(self.y_.shape[0])
                if np.dot(self.weights_, self.X_[x, :]) > 0:
                    if self.y_[x] == 1:
                        self.weights_ += self.X_[x, :]
                else:
                    if self.y_[x] == 0:
                        self.weights_ -= self.X_[x, :]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return [np.dot(x, self.weights_) for x in X]


from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
# print(X.shape)
# print(y)

print(Perceptron(n_iter=1).fit(X, y).predict((X)))