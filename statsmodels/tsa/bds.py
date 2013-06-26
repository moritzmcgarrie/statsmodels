"""
BDS test for IID time series

References
----------

Broock, W. A., J. A. Scheinkman, W. D. Dechert, and B. LeBaron. 1996.
"A Test for Independence Based on the Correlation Dimension."
Econometric Reviews 15 (3): 197–235.

Kanzler, Ludwig. 1999.
"Very Fast and Correctly Sized Estimation of the BDS Statistic".
SSRN Scholarly Paper ID 151669. Rochester, NY: Social Science Research Network.

LeBaron, Blake. 1997.
"A Fast Algorithm for the BDS Statistic."
Studies in Nonlinear Dynamics & Econometrics 2 (2) (January 1).
"""

import numpy as np
from scipy import stats


def indicators(x, eps=None, d=1.5):
    """
    Parameters
    ----------
    x : 1d array
        observations of time series for which bds statistics is calculated
    eps : scalar, optional
        epsilon
    d : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it
    """
    # Cache the indicators: I(x_i,x_j) = 1 if |x_i - x_j| < eps
    #                                  = 0 otherwise
    T = len(x)

    #TODO: add functionality to select epsilon optimally
    #TODO: and/or compute for a range of epsilons in [0.5*s, 2.0*s]?
    #      or [1.5*s, 2.0*s]?
    if eps is None:
        eps = d*x.std(ddof=1)

    I = np.zeros((T, T), dtype=np.int8)     # Use bool for better storage
    for i in range(T):                      # i is a "row" of matrix I
        I[i, i] = True                      # |x_i - x_i| < eps is always True
        for j in range(i+1, T):             # j is a "column" of matrix I
            I[j, i] = I[i, j] = np.abs(x[i] - x[j]) < eps
    return I


def CI(I, m, Tm=None):
    """
    Parameters
    ----------
    I : 2d array
        upper triangular matrix of indicators
    m : integer
        embedding dimension
    """
    T = len(I)
    # We need to condition on m initial values to practically implement this
    Tm = T - (m-1)

    val = 0
    for s in range(m, T+1):
        for t in range(s+1, T+1):
            val += np.product(I.diagonal(t-s)[s-m:s])
    return 2*val / (Tm*(Tm-1))


def K(I):
    """
    This is correct!

    Parameters
    ----------
    I : 2d array
        upper triangular matrix of indicators
    m : integer
        embedding dimension
    """
    T = len(I)
    val = 0
    for t in range(0, T):
        for s in range(t+1, T):
            for r in range(s+1, T):
                val += (I[t, s]*I[s, r] + I[t, r]*I[r, s] + I[s, t]*I[t, r])/3
    return 6*val/(T*(T-1)*(T-2))


def V2(I, m):
    """
    Parameters
    ----------
    I : 2d array
        upper triangular matrix of indicators
    m : integer
        embedding dimension
    """
    T = len(I)

    c = CI(I, 1)
    k = K(I)

    tmp = 0
    for j in range(1, m):
        tmp += (k**(m-j))*(c**(2*j))

    return 4*(k**m + 2*tmp + ((m-1)**2)*(c**(2*m)) - (m**2)*k*(c**(2*m-2)))


def bds(x, m=2, eps=None, d=1.5):
    """
    Parameters
    ----------
    x : 1d array
        observations of time series for which bds statistics is calculated
    m : integer
        embedding dimension
    eps : scalar, optional
        epsilon
    d : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it
    """
    T = len(x)
    # We need to condition on m initial values to practically implement this
    Tm = T - (m-1)

    # Cache the indicators
    I = indicators(x, eps, d)

    # Get the intermediate values
    C_1T = CI(I[:-1, :-1], 1)  # See Kanzler footnote 10 for why I is truncated
    C_mT = CI(I, m)
    T_mT = C_mT - (C_1T**m)
    V_mT = np.sqrt(V2(I, m))

    # Calculate the statistic: W_mT ~ N(0,1)
    W_mT = np.sqrt(Tm) * T_mT / V_mT

    # Calculate the p-value (two-tailed test)
    pvalue = 2*stats.norm.sf(W_mT)

    return W_mT, pvalue
