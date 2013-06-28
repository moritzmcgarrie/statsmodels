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
    Parameters
    ----------
    x : 1d array
        observations of time series for which bds statistics is calculated
    epsilon : scalar, optional
        the threshold distance to use in calculating the correlation sum
    distance : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it

    Notes
    -----

    Since this can be a very large matrix, use np.int8 to save some space.
    """
    nobs_full = len(x)

    if epsilon is not None and epsilon <= 0:
        raise ValueError("Threshold distance must be positive if specified." \
                         " Got epsilon of %f" % epsilon)
    if distance <= 0:
        raise ValueError("Threshold distance must be positive." \
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
    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    embedding_dim : integer
        embedding dimension
    """
    nobs_full = len(indicators)
    # We need to condition on m initial values to practically implement this
    nobs = nobs_full - (embedding_dim - 1)

    if not indicators.ndim == 2:
        raise ValueError('Indicators must be a matrix')
    if not indicators.shape[0] == indicators.shape[1]:
        raise ValueError('Indicator matrix must be symmetric (square)')

    val = 0
    for s in range(embedding_dim, nobs_full+1):
        for t in range(s+1, nobs_full+1):
            val += np.product(indicators.diagonal(t - s)[s - embedding_dim:s])
    return 2 * val / (nobs * (nobs - 1))


def _k(indicators):
    """
    Parameters
    ----------
    I : 2d array
        matrix of distance threshold indicators
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
    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    embedding_dim : integer
        embedding dimension

    Notes
    -----

    """
    corrsum_1dim = correlation_sum(indicators, 1)
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

    Notes
    -----

    Implementation conditions on the first m-1 initial values, which are
    required to calculate the m-histories:
    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})
    """
    nobs_full = len(x)
    nobs = nobs_full - (embedding_dim - 1)

    if embedding_dim < 2 or embedding_dim >= nobs_full:
        raise ValueError("Embedding dimension must be in the range" \
                         " [2,len(x)-1]. Got %d." % embedding_dim)

    # Cache the indicators
    indicators = distance_indicators(x, epsilon, distance)

    # Get the estimates of the correlation integrals
    # (see Kanzler footnote 10 for why indicators are truncated in 1dim case)
    corrsum_1dim = correlation_sum(indicators[:-1, :-1], 1)
    corrsum_mdim = correlation_sum(indicators, embedding_dim)

    # Get the intermediate values for the statistic
    effect = corrsum_mdim - (corrsum_1dim**embedding_dim)
    sd = np.sqrt(_var(indicators, embedding_dim))

    # Calculate the statistic: bds_stat ~ N(0,1)
    bds_stat = np.sqrt(nobs) * effect / sd

    # Calculate the p-value (two-tailed test)
    pvalue = 2*stats.norm.sf(np.abs(bds_stat))

    return bds_stat, pvalue
