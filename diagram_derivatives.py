import numpy as np
from giotto.diagrams import Amplitude, PairwiseDistance
from giotto.homology import CubicalPersistence
from giotto.homology import VietorisRipsPersistence
from giotto.utils.validation import check_diagram, validate_params, validate_metric_params
from giotto.base import TransformerResamplerMixin
from giotto.diagrams._utils import _discretize

from sklearn.base import BaseEstimator

from joblib import Parallel, delayed, effective_n_jobs


class DiagramDerivative(BaseEstimator, TransformerResamplerMixin):

    _hyperparameters = {'order': [float, (1, np.inf)]}

    def __init__(self, metric='landscape', metric_params=None, order=2., h=None,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.h = h
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator and return it.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples_fit, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        hyperparameters = self.get_params().copy()
        print(hyperparameters)
        if self.order is not None:
            if isinstance(self.order, int):
                hyperparameters['order'] = float(self.order)
        else:
            hyperparameters['order'] = 1.  # Automatically pass validate_params

        validate_params(hyperparameters, self._hyperparameters)
        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)

        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
            self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

        self._X = X
        return self

    def transform(self, X):
        """ The method takes as input a set of diagrams and compute their 'temporal' derivative. Thus the
        diagrams should be given for fixed toroidal-coordinate and different times. If not passed in the constructor the
        delta_t is considered equal to 1.
        """
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        h = self.h if self.h is not None else 1

        def my_pairwise_distance(x, y):
            dist = PairwiseDistance(metric=self.metric, metric_params=self.metric_params,
                                    order=self.order, n_jobs=1).fit_transform(np.concatenate([np.expand_dims(x, axis=0),
                                                                                              np.expand_dims(y, axis=0)]
                                                                                             ))
            return dist[0, 1]

        metric_function = my_pairwise_distance
        derivatives = Parallel(n_jobs=n_jobs)(delayed(metric_function)(
            X[t], X[t+1]) for t in range(len(X)-1))
        return np.array(derivatives)/h








