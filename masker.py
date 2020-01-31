from giotto.base import TransformerResamplerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, column_or_1d
import numpy as np


class Masker(BaseEstimator, TransformerResamplerMixin):
    def __init__(self, last=False):
        self.last = last

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['_is_fitted'])
        Xt = X[:-1] if self.last else X[1:]
        return Xt

    def resample(self, y, X=None):
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)
        yt = y[:-1] if self.last else y[1:]
        return yt


class Squeezer(BaseEstimator, TransformerResamplerMixin):
    def __init__(self, dim=0):
        self.dim = dim

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['_is_fitted'])
        assert X.shape[self.dim] == 1
        return np.squeeze(X, self.dim)





