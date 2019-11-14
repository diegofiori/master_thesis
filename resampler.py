from giotto.utils import validate_params
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn.base import BaseEstimator
from giotto.base import TransformerResamplerMixin

import numpy as np

# @TODO: the comments must be modified


class ShiftResampler(BaseEstimator, TransformerResamplerMixin):
    """Data sampling transformer that returns a sampled numpy.ndarray.

    Parameters
    ----------
    period : int, default: 2
        The sampling period, i.e. one point every period will be kept.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from giotto.time_series import Resampler
    >>> # Create a noisy signal sampled
    >>> signal = np.asarray([np.sin(x /40) + np.random.random()
    ... for x in range(0, 300)])
    >>> plt.plot(signal)
    >>> plt.show()
    >>> # Set up the Resampler
    >>> period = 10
    >>> periodic_sampler = Resampler(period=period)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(signal)
    >>> signal_resampled = periodic_sampler.transform(signal)
    >>> plt.plot(signal_resampled)

    """
    _hyperparameters = {'period': [int, (1, np.inf)]}

    def __init__(self, period=2):
        self.period = period

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        # check_array(X, ensure_2d=False)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform/resample X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples_new, n_features)
            The transformed/resampled input array. ``n_samples_new =
            n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        # Xt = check_array(X, ensure_2d=False)
        Xt = X

        return np.concatenate([np.expand_dims(Xt[i::self.period], axis=0) for i in range(self.period)])

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_features)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``n_samples_new = n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)

        return np.concatenate([np.expand_dims(y[i::self.period], axis=0) for i in range(self.period)])


class Grouper(BaseEstimator, TransformerResamplerMixin):

    _hyperparameters = {'period': [int, (1, np.inf)]}

    def __init__(self, period=2):
        self.period = period

    def fit(self, X, y=None):
        validate_params(self.get_params(), self._hyperparameters)
        # check_array(X, ensure_2d=False)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['_is_fitted'])
        # Xt = check_array(X, ensure_2d=False)
        Xt = X

        return np.concatenate([np.expand_dims(Xt[i:i+self.period], axis=0) for i in range(0, Xt.shape[0], self.period)])

    def resample(self, y, X=None):
        check_is_fitted(self, ['_is_fitted'])
        print(y.shape)
        y = column_or_1d(y)

        return np.concatenate([np.expand_dims(y[i:i+self.period], axis=0) for i in range(0, len(y), self.period)])


class Resampler(BaseEstimator, TransformerResamplerMixin):
    """Data sampling transformer that returns a sampled numpy.ndarray. Note that tha sampling is done on the axis=1!!!

    Parameters
    ----------
    period : int, default: 2
        The sampling period, i.e. one point every period will be kept.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from giotto.time_series import Resampler
    >>> # Create a noisy signal sampled
    >>> signal = np.asarray([np.sin(x /40) + np.random.random()
    ... for x in range(0, 300)])
    >>> plt.plot(signal)
    >>> plt.show()
    >>> # Set up the Resampler
    >>> period = 10
    >>> periodic_sampler = Resampler(period=period)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(signal)
    >>> signal_resampled = periodic_sampler.transform(signal)
    >>> plt.plot(signal_resampled)

    """
    _hyperparameters = {'period': [int, (1, np.inf)]}

    def __init__(self, period=2):
        self.period = period

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        # check_array(X, ensure_2d=False)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform/resample X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples_new, n_features)
            The transformed/resampled input array. ``n_samples_new =
            n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        # Xt = check_array(X, ensure_2d=False)
        Xt = X

        return Xt[:, :self.period]

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_features)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``n_samples_new = n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)

        return y
