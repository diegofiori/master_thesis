import numpy as np
from gtda.diagrams import Amplitude, PairwiseDistance
from gtda.homology import CubicalPersistence
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams._metrics import *
from gtda.utils.validation import check_diagram, validate_params, validate_metric_params
from gtda.base import TransformerResamplerMixin
from gtda.diagrams._utils import _discretize, _subdiagrams
from joblib.parallel import Parallel, delayed

from sklearn.base import BaseEstimator

from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


def _parallel_successive_pairwise(X, metric, metric_params,
                                  homology_dimensions, periodic, n_jobs):
    metric_func = implemented_metric_recipes[metric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None for dim in homology_dimensions}
    samplings = effective_metric_params.pop('samplings', none_dict)
    step_sizes = effective_metric_params.pop('step_sizes', none_dict)
    if periodic:
        X = np.concatenate([X, np.expand_dims(X[0], axis=0)])

    distance_matrices = Parallel(n_jobs=n_jobs)(delayed(metric_func)(
        _subdiagrams(X[t-1].reshape(1, X.shape[1], 3), [dim], remove_dim=True),
        _subdiagrams(X[t].reshape(1, X.shape[1], 3), [dim], remove_dim=True),
        sampling=samplings[dim], step_size=step_sizes[dim],
        **effective_metric_params) for dim in homology_dimensions
        for t in range(1, len(X)))

    distance_matrices = np.concatenate(distance_matrices, axis=1)

    ind_temp = X.shape[0] - 1
    distance_matrices = np.stack(
        [distance_matrices[:, i*ind_temp: (i+1)*ind_temp]
         for i in range(len(homology_dimensions))], axis=2)
    return distance_matrices


class DiagramDerivative(BaseEstimator, TransformerResamplerMixin):

    _hyperparameters = {'order': [float, (1, np.inf)]}

    def __init__(self, metric='landscape', metric_params=None, order=2., periodic=False,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.periodic = periodic
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

        Xt = _parallel_successive_pairwise(X, self.metric,
                                self.effective_metric_params_,
                                self.homology_dimensions_,
                                self.periodic,
                                self.n_jobs)

        if self.order is not None:
            Xt = np.linalg.norm(Xt, axis=2, ord=self.order)
            Xt = Xt.reshape((-1, 1))
        return Xt

    def resample(self, y, X=None):
        return y[1:]


class MultiDiagramsDerivative(BaseEstimator, TransformerResamplerMixin):
    def __init__(self, *args, n_jobs=1, **kwargs):
        self.n_jobs = n_jobs
        self.diagram_derivative = DiagramDerivative(*args, **kwargs)

    def fit(self, X, y=None):
        self.diagram_derivative.fit(X[0], y)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """ X must be an array of groups diagrams or list of groups of diagrams."""

        check_is_fitted(self, ['_is_fitted'])
        X_t = Parallel(n_jobs=self.n_jobs)(delayed(self.diagram_derivative.transform)(X[i]) for i in range(len(X)))
        X_t = np.concatenate(X_t)
        return X_t

    def resample(self, y, X=None):
        check_is_fitted(self, ['_is_fitted'])
        return y

    def get_params(self, deep=True):
        out = self.diagram_derivative.get_params(deep)
        out['n_jobs'] = self.n_jobs
        return out





