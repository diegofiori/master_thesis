from giotto.utils import validate_params
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn.base import BaseEstimator
from giotto.base import TransformerResamplerMixin

import numpy as np

# @TODO: the comments must be modified


class ShiftResampler(BaseEstimator, TransformerResamplerMixin):
    """Data sampling transformer that returns a shifted and sampled numpy.ndarray. It means that from a 1d array
    for instance the class produce as output a matrix where each raw is a sample from the original array. Each raw is
    shifted from the previous one of 1. Clearly the number of raw will be equal to the period.

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
    >>> periodic_sampler = ShiftResampler(period=period)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(signal)
    >>> signal_resampled = periodic_sampler.transform(signal)
    >>> print(f'the signal sampled from index 0 with period {period}')
    >>> plt.plot(signal_resampled[0])
    >>> print(f'the signal sampled from index 5 with period {period}')
    >>> plt.plot(signal_resampled[5])

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
        X : array-like
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : array-like,
            The transformed/resampled input array. ``new_shape = (period, n_samples // period,  *X.shape[1:])``.

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
        y : ndarray, shape (n_samples, )
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``new_shape = (period, n_samples // period)``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)

        return np.concatenate([np.expand_dims(y[i::self.period], axis=0) for i in range(self.period)])


class Grouper(BaseEstimator, TransformerResamplerMixin):
    """ Data grouping transformer. It groups tha data in batch of length period.

    Parameters
    ----------
    period: int, (default 2)

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.arange(10)
    >>> grouper = Grouper(period=5)
    >>> fake_y = np.zeros(len(vector))
    >>> vector_grouped = grouper.fit_transform_resample(vector, fake_y)[0]
    >>> vector_grouped
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
        X : ndarray
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
        """Transform/group X.
        Note that the method gives an error if n_samples // period != 0

        Parameters
        ----------
        X : array-like
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : array-like,
            The transformed/grouped input array. ``new_shape = (n_samples / period, period, *X.shape[1:])``.

        """
        check_is_fitted(self, ['_is_fitted'])
        # Xt = check_array(X, ensure_2d=False)
        Xt = X

        return np.concatenate([np.expand_dims(Xt[i:i+self.period], axis=0) for i in range(0, Xt.shape[0], self.period)])

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, )
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``new_shape = (n_samples / period, period)``.

        """
        check_is_fitted(self, ['_is_fitted'])
        print(y.shape)
        y = column_or_1d(y)

        return np.concatenate([np.expand_dims(y[i:i+self.period], axis=0) for i in range(0, len(y), self.period)])


class Degrouper(BaseEstimator, TransformerResamplerMixin):
    """ The inverse of the Data grouping transformer. It groups tha data in batch of length period.

    Parameters
    ----------
    period: int, (default 2)

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.arange(10,)
    >>> grouper = Grouper(period=5)
    >>> fake_y = np.zeros(len(vector))
    >>> vector_grouped = grouper.fit_transform_resample(vector, fake_y)[0]
    >>> degrouper = Degrouper()
    >>> old_vector = degrouper.fit_transform(vector_grouped)
    >>> assert vector == old_vector
    """
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform/group X.
        Note that the method gives an error if n_samples // period != 0

        Parameters
        ----------
        X : array-like
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : array-like,
            The transformed/grouped input array. ``new_shape = (n_samples / period, period, *X.shape[1:])``.

        """
        check_is_fitted(self, ['_is_fitted'])
        # Xt = check_array(X, ensure_2d=False)
        Xt = X

        return np.concatenate([Xt[i] for i in range(len(Xt))])

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, )
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``new_shape = (n_samples / period, period)``.

        """
        check_is_fitted(self, ['_is_fitted'])
        # print(y.shape)
        # y = column_or_1d(y)

        return y.reshape((-1, ))


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
    ... for x in range(0, 300)]).reshape((1, -1))
    >>> plt.plot(signal.reshape((-1, )))
    >>> plt.show()
    >>> # Set up the Resampler
    >>> period = 10
    >>> periodic_sampler = Resampler(period=period)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(signal)
    >>> signal_resampled = periodic_sampler.transform(signal)
    >>> plt.plot(signal_resampled.reshape((-1, )))

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
        X : ndarray, shape (n_group, n_samples, n_features)
            Input data. It can be generalized at further dimensions.

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
        X : ndarray, shape (n_group, n_samples, n_features)
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_group, n_samples_new, n_features)
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
        y : ndarray, shape (n_group, n_samples)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_group, n_samples_new, 1)
            The resampled target. ``n_samples_new = n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)

        return y
