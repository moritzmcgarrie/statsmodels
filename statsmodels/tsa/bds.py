"""
BDS test for IID time series

References
----------

Broock, W. A., J. A. Scheinkman, W. D. Dechert, and B. LeBaron. 1996.
"A Test for Independence Based on the Correlation Dimension."
Econometric Reviews 15 (3): 197-235.

Kanzler, Ludwig. 1999.
"Very Fast and Correctly Sized Estimation of the BDS Statistic".
SSRN Scholarly Paper ID 151669. Rochester, NY: Social Science Research Network.

LeBaron, Blake. 1997.
"A Fast Algorithm for the BDS Statistic."
Studies in Nonlinear Dynamics & Econometrics 2 (2) (January 1).
"""

from __future__ import division
import numpy as np
from scipy import stats


def distance_indicators(x, epsilon=None, distance=1.5):
    """
    Calculate all pairwise threshold distance indicators for a time series

    Parameters
    ----------
    x : 1d array
        observations of time series for which bds statistics is calculated
    epsilon : scalar, optional
        the threshold distance to use in calculating the heaviside indicators
    distance : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it

    Returns
    -------
    indicators : 2d array
        matrix of distance threshold indicators

    Notes
    -----

    Since this can be a very large matrix, use np.int8 to save some space.
    """
    nobs_full = len(x)

    if epsilon is not None and epsilon <= 0:
        raise ValueError("Threshold distance must be positive if specified."
                         " Got epsilon of %f" % epsilon)
    if distance <= 0:
        raise ValueError("Threshold distance must be positive."
                         " Got distance multiplier %f" % distance)

    #TODO: add functionality to select epsilon optimally
    #TODO: and/or compute for a range of epsilons in [0.5*s, 2.0*s]?
    #      or [1.5*s, 2.0*s]?
    if epsilon is None:
        epsilon = distance * x.std(ddof=1)

    indicators = np.zeros((nobs_full, nobs_full), dtype=np.int8)
    for i in range(nobs_full):           # i is a "row" of matrix I
        indicators[i, i] = 1             # |x_i-x_i| < epsilon always True
        for j in range(i+1, nobs_full):  # j is a "column" of matrix I
            indicators[j, i] = indicators[i, j] = np.abs(x[i] - x[j]) < epsilon
    return indicators


def correlation_sum(indicators, embedding_dim):
    """
    Calculate a correlation sum

    Useful as an estimator of a correlation integral

    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    embedding_dim : integer
        embedding dimension

    Returns
    -------
    corrsum : float
        Correlation sum
    indicators_joint
        matrix of joint-distance-threshold indicators

    """
    if not indicators.ndim == 2:
        raise ValueError('Indicators must be a matrix')
    if not indicators.shape[0] == indicators.shape[1]:
        raise ValueError('Indicator matrix must be symmetric (square)')

    if embedding_dim == 1:
        indicators_joint = indicators
    else:
        corrsum, indicators = correlation_sum(indicators, embedding_dim - 1)
        indicators_joint = indicators[1:,1:]*indicators[:-1,:-1]
    
    nobs = len(indicators_joint)
    corrsum = np.mean(indicators_joint[np.triu_indices(nobs, 1)])
    return corrsum, indicators_joint

#TODO rework this
def _k(indicators):
    """
    Calculate k

    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators

    Returns
    -------
    k : float
        k
    """
    nobs_full = len(indicators)

    val = 0
    for t in range(0, nobs_full):
        for s in range(t+1, nobs_full):
            for r in range(s+1, nobs_full):
                val += (1/3)*(
                    indicators[t, s]*indicators[s, r] +
                    indicators[t, r]*indicators[r, s] +
                    indicators[s, t]*indicators[t, r]
                )

    return 6 * val / (nobs_full * (nobs_full - 1) * (nobs_full - 2))


def _var(indicators, embedding_dim):
    """
    Calculate the variance of a BDS effect

    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    embedding_dim : integer
        embedding dimension

    Returns
    -------
    variance : float
        Variance of BDS effect

    Notes
    -----

    """
    corrsum_1dim, _ = correlation_sum(indicators, 1)
    k = _k(indicators)

    tmp = 0
    for j in range(1, embedding_dim):
        tmp += (k**(embedding_dim - j))*(corrsum_1dim**(2 * j))

    return 4 * (
        k**embedding_dim +
        2 * tmp +
        ((embedding_dim - 1)**2) * (corrsum_1dim**(2 * embedding_dim)) -
        (embedding_dim**2) * k * (corrsum_1dim**(2 * embedding_dim - 2))
    )


def bds(x, embedding_dim=2, epsilon=None, distance=1.5):
    """
    Calculate the BDS test statistic for a time series

    Parameters
    ----------
    x : 1d array
        observations of time series for which bds statistics is calculated
    embedding_dim : integer
        embedding dimension
    epsilon : scalar, optional
        the threshold distance to use in calculating the correlation sum
    distance : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it

    Returns
    -------
    bds_stat : float
        The BDS statistic
    pvalue : float
        The p-values associated with the BDS statistic

    Notes
    -----

    Implementation conditions on the first m-1 initial values, which are
    required to calculate the m-histories:
    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})
    """
    nobs_full = len(x)
    ninitial = (embedding_dim - 1)
    nobs = nobs_full - ninitial

    if embedding_dim < 2 or embedding_dim >= nobs_full:
        raise ValueError("Embedding dimension must be in the range"
                         " [2,len(x)-1]. Got %d." % embedding_dim)

    # Cache the indicators
    indicators = distance_indicators(x, epsilon, distance)

    # Get the estimates of the correlation integrals
    # (see Kanzler footnote 10 for why indicators are truncated in 1dim case)
    corrsum_1dim, _ = correlation_sum(indicators[ninitial:, ninitial:], 1)
    corrsum_mdim, _ = correlation_sum(indicators, embedding_dim)

    # Get the intermediate values for the statistic
    effect = corrsum_mdim - (corrsum_1dim**embedding_dim)
    sd = np.sqrt(_var(indicators, embedding_dim))

    # Calculate the statistic: bds_stat ~ N(0,1)
    bds_stat = np.sqrt(nobs) * effect / sd

    # Calculate the p-value (two-tailed test)
    pvalue = 2*stats.norm.sf(np.abs(bds_stat))

    return bds_stat, pvalue
