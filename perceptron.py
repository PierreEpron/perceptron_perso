import numpy as np

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, classification_report

from sklearn.base import BaseEstimator, ClassifierMixin

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=1e5):
        self.n_iter = n_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y     
        
        self.weights_ = np.ones((X.shape[1]))
        
        for i in range(int(self.n_iter)):
            x = np.random.randint(self.y_.shape[0])
            if np.dot(self.weights_, self.X_[x, :]) < 0:
                if self.y_[x] == 1:
                    self.weights_ += self.X_[x, :]
            else:
                if self.y_[x] == 0:
                    self.weights_ -= self.X_[x, :]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return [0 if np.dot(x, self.weights_) < 0 else 1 for x in X]


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X = Normalizer().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


p = Perceptron(n_iter=1e5).fit(X_train, y_train)
# print(p.weights_)
y_pred = p.predict(X_test)
print(classification_report(y_test, y_pred))