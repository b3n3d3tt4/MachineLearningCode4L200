"""
Time Series distances (numpy)
Adapted from Florent Forest (FlorentF9), DeepTemporalClustering
https://github.com/FlorentF9/DeepTemporalClustering
referring to the papaer https://arxiv.org/abs/1802.01059

Note: x, y are single multivariate time series, shape (input_length, channels)
-- not batches. Used by init_cluster_weights for hierarchical cluster
initialization, not inside the TSClusteringLayer forward pass (which
recomputes eucl/cid/cor as differentiable ops directly on tensors).
"""

import numpy as np
from statsmodels.tsa import stattools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def eucl(x, y):
    """
    Euclidean distance between two multivariate time series.
    For each channel: euclidean distance along time; then sum over channels.
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series.
    Corrects the euclidean distance by a complexity factor based on how
    "jagged" each series is (estimated via the sum of squared consecutive
    differences), so that two series of different complexity aren't
    considered close just because they're both smooth/flat.
    Reference: Batista, Wang & Keogh (2011).
    https://doi.org/10.1137/1.9781611972818.60
    """
    assert len(x.shape) == 2 and x.shape == y.shape, \
        "time series must have the same length and number of channels"

    # Complexity estimate per channel: sqrt(sum of squared consecutive diffs)
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)

    # Complexity factor: ratio of the more complex to the less complex series
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(
        np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y)
    )

    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series.
    Z-normalizes each series (zero mean, unit variance) along time, then
    turns the Pearson correlation coefficient into a distance.
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)

    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficient
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation -> distance

    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series.
    Computes a linearly-weighted euclidean distance between the
    autocorrelation coefficients of the two series (more weight on
    lower lags), rather than between the raw values.
    Reference: Galeano & Pena (2000), Multivariate Analysis in Vector Time Series.

    Note: usable here for hierarchical cluster initialization, but NOT
    implemented in the differentiable clustering layer (raises
    NotImplementedError there) -- same as in the original repo.
    """
    assert len(x.shape) == 2 and x.shape == y.shape, \
        "time series must have the same length and number of channels"

    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)

    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))

    return np.sum(d)