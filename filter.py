import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gtda.utils.validation import validate_params, check_diagram
from sklearn.utils.validation import check_is_fitted


class FilterBigComponents(BaseEstimator, TransformerMixin):

    _hyperparameters = {'n_filter': [int, (0, np.inf)]}

    def __init__(self, n_filter=1):
        self.n_filter = n_filter

    def fit(self, X, y=None):
        hyperparameters = {}
        if isinstance(self.n_filter, float):
            hyperparameters['n_filter'] = int(self.n_filter)
        else:
            hyperparameters['n_filter'] = self.n_filter
        check_diagram(X)
        validate_params(hyperparameters, self._hyperparameters)
        self.homology_dims_ = np.unique(X[:, :, -1]).astype(int)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['_is_fitted'])
        shape = X.shape

        def remove_comp(X_temp):
            X_temp_idx = np.argsort(X_temp[:, :, 1]-X_temp[:, :, 0], axis=1)
            return X_temp[X_temp_idx < X_temp.shape[1] - self.n_filter]\
                .reshape((shape[0], -1, 3))

        Xt = [remove_comp(X[X[:, :, -1] == hom_dim].reshape((shape[0], -1, 3)))
              for hom_dim in self.homology_dims_]
        return np.concatenate(Xt, axis=1)

