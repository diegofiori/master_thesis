import types
import numpy as np
from giotto.diagrams._metrics import _parallel_amplitude
from giotto.diagrams._utils import _discretize
from giotto.utils import validate_params, validate_metric_params, check_diagram

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Scaler(BaseEstimator, TransformerMixin):
    """Linear scaling of persistence diagrams.

    A positive scale factor is calculated during :meth:`fit` by considering all
    available persistence diagrams and homology dimensions. During
    :meth:`transform`, all birth-death pairs are divided by this factor.

    The value of the scale factor depends on two things:

        - A way of computing, for each homology dimension, the `amplitude
          <LINK TO GLOSSARY>`_ in that dimension of a persistence diagram
          consisting of birth-death-dimension triples [b, d, q]. Together,
          `metric` and `metric_params` define this in the same way as
          in :class:`Amplitude`.
        - A scalar-valued function which is applied to the resulting
          two-dimensional array of amplitudes.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'bottleneck'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` there are no available arguments.
        - If ``metric == 'wasserstein'`` the only argument is `p` (int,
          default: ``2``).
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_values` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_values` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``1.``) and `n_values`
          (int, default: ``100``).

    function : callable, optional, default: ``numpy.max``
        Function used to extract a positive scalar from the collection of
        amplitude vectors in :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params` as
        well as on any relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    scale_ : float
        The scaling factor used to rescale diagrams.

    See also
    --------
    Filtering, Amplitude, PairwiseDistance, \
    giotto.homology.VietorisRipsPersistence

    Notes
    -----
    To compute scaling factors without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """
    _hyperparameters = {'function': [types.FunctionType]}

    def __init__(self, metric='bottleneck', metric_params=None,
                 function=np.max, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.function = function
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator by finding the scale factor, then returns it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
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
        validate_params(self.get_params(), self._hyperparameters)

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)
        print(np.sum(X < 0))
        amplitude_array = _parallel_amplitude(X, self.metric,
                                              self.effective_metric_params_,
                                              self.homology_dimensions_,
                                              self.n_jobs)
        self.scale_ = self.function(amplitude_array)

        return self

    def transform(self, X, y=None):
        """Divide all birth and death values in `X` by :attr:`scale_`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xs : ndarray, shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self, ['scale_', 'homology_dimensions_',
                               'effective_metric_params_'])

        Xs = check_diagram(X)
        Xs[:, :, :2] /= self.scale_
        return Xs

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation. Multiplies
        by the scale found in :meth:`fit`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Data to apply the inverse transform to.

        Returns
        -------
        Xs : ndarray, shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self, ['effective_metric_params_'])

        Xs = check_diagram(X)
        Xs[:, :, :2] *= self.scale_
        return Xs
