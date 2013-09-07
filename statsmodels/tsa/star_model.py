"""
Smooth Transition Autoregression

References
----------

Dijk, Dick van, Timo Terasvirta, and Philip Hans Franses. 2002.
"Smooth Transition Autoregressive Models - a Survey of Recent Developments."
Econometric Reviews 21 (1): 1-47.

"""

from __future__ import division
import numpy as np
import statsmodels.base.model as base
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.miscmodels.nonlinls as nls
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tsa.tsatools import add_constant, lagmat
from statsmodels.regression.linear_model import OLS
from scipy import optimize


class LSTAR1(base.Model):
    """
    Logistic Smooth Transition Autoregression Model

    Estimates the model

    :math:`y_t = \phi_1'x_t + (\phi_2 - \phi_1)'x_t G(s_t;\gamma,c) + \varepsilon_t`

    where :math:`\gamma` is the smoothness parameter and :math:`c` is the
    threshold parameter.

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    ar_order : integer
        The order of the autoregressive parameters.
    threshold_var : array-like or integer, optional
        The threshold variable. Either an array of the same length as `endog`
        or an integer from 1 - `ar_order` as the delay parameter for a
        self-exiting model.
    trend : str {'c','nc'}
        Whether to include a constant or not
        'c' includes constant
        'nc' no constant
    smoothness : float, optional
        The smoothness parameter. If not specified, will be estimated via
        non-linear least squares.
    theshold : float, optional
        The threshold parameter. If not specified, will be estimated via
        non-linear least squares.
    smoothness_grid_size : integer, optional
        The size of the grid in the smoothness dimension. Used to select start
        parameters for non-linear least squares.
    smoothness_grid_int : iterable, optional
        The interval on which the smoothness dimension of the grid is
        constructed.
    threshold_grid_size : integer, optional
        The approximate number of elements in the threshold grid if a grid
        search is used.
    threshold_grid_int : iterable, optional
        The trim parameters of the threshold variable on which the threshold
        dimension of the grid is constructed. For example, (0.1, 0.9) will
        construct `threshold_grid_size` evenly spaced values between the 10th
        and 90th percentiles of the threshold variable values.
    """

    def __init__(self, endog, ar_order, threshold_var=1, trend='c',
                 smoothness=None, threshold=None,
                 smoothness_grid_size=40, smoothness_grid_int=(1,100),
                threshold_grid_size=100, threshold_grid_int=(0.1, 0.9),
                 dates=None, freq=None, missing='none'):

        if ((smoothness is None and threshold is not None) or
                (threshold is None and smoothness is not None)):
            raise ValueError('Must specify both or neither of `smoothness` and'
                             ' `threshold` arguments.')        

        self.nobs_initial = ar_order
        self.nobs = endog.shape[0] - ar_order
        
        self.ar_order = ar_order
        self.k_trend = int(trend == 'c')
        self.k = self.ar_order + self.k_trend

        self.smoothness = smoothness
        self.threshold = threshold
        self.cov_hyperparams = None

        self.threshold_grid_size = threshold_grid_size
        self.threshold_grid_int = threshold_grid_int
        self.smoothness_grid_size = smoothness_grid_size
        self.smoothness_grid_int = smoothness_grid_int

        orig_endog = endog
        orig_exog = lagmat(orig_endog, ar_order)
        if self.k_trend:
            orig_exog = add_constant(orig_exog)

        # Create datasets / complete initialization
        endog = orig_endog[self.nobs_initial:]
        exog = orig_exog[self.nobs_initial:]
        super(LSTAR1, self).__init__(endog, exog,
                                    hasconst=self.k_trend, missing=missing)

        # Overwrite originals
        self.data.orig_endog = orig_endog
        self.data.orig_exog = orig_exog

        # Create the threshold variable
        if isinstance(threshold_var, int):
            if threshold_var < 1 or threshold_var > ar_order:
                raise ValueError('Delay parameter (from `threshold_var`) must'
                                 ' be an integer in the range [1, %d].'
                                 ' Got %d.' % (ar_order, threshold_var))
            self.threshold_var = self.exog[:,-threshold_var]
        else:
            threshold_var = np.asanyarray(threshold_var)
            if not threshold_var.shape[0] == endog.shape[0]:
                raise ValueError('Threshold variable must be of the same'
                                 ' length as the endogenous variable.'
                                 ' Expected length %d, got %d.' %
                                 (endog.shape[0], threshold_var.shape[0]))
            self.threshold_var = threshold_var

    def _fit(self, params):
        smoothness, threshold = params
        transition = 1/(1 + np.exp(-smoothness*(self.threshold_var - threshold)))
        
        exog = np.c_[
            self.exog,
            np.multiply(self.exog.T, transition).T
        ]

        return OLS(self.endog, exog).fit()

    def _ssr(self, params):
        return self._fit(params).ssr

    def score(self, params):
        '''
        Gradient of log-likelihood evaluated at params
        '''
        kwds = {}
        kwds.setdefault('centered', True)
        return approx_fprime(params, self._ssr, **kwds).ravel()

    def jac(self, params, **kwds):
        '''
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        '''
        #kwds.setdefault('epsilon', 1e-4)
        kwds.setdefault('centered', True)
        return approx_fprime(params, self._ssr, **kwds)

    def hessian(self, params):
        '''
        Hessian of log-likelihood evaluated at params
        '''
        from statsmodels.tools.numdiff import approx_hess
        # need options for hess (epsilon)
        return approx_hess(params, self._ssr)

    def start_params(self):
        t_start, t_stop = self.threshold_grid_int
        t_step = (t_stop - t_start) / self.threshold_grid_size
        threshold_grid = (t_start, t_stop, t_step)

        s_start, s_stop = self.smoothness_grid_int
        s_step = (s_stop - s_start) / self.smoothness_grid_size
        smoothness_grid = (s_start, s_stop, s_step)

        f = lambda params, *args: self._ssr(params, *args)

        return optimize.brute(f, (threshold_grid, smoothness_grid),
                              finish=None)
            
    def fit(self, start_params=None, method='bfgs', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None,
            retall=False, **kwargs):
        
        if not self.smoothness or not self.threshold:
            # Extract kwargs specific to fit_regularized calling fit
            cov_params_func = kwargs.setdefault('cov_params_func', None)

            Hinv = None  # JP error if full_output=0, Hinv not defined
            methods = ['bfgs']
            if start_params is None:
                start_params = self.start_params()

            if method.lower() not in methods:
                message = "Unknown fit method %s" % method
                raise ValueError(message)
            method = method.lower()
            
            nobs = self.endog.shape[0]
            f = lambda params, *args: self._ssr(params, *args)
            score = lambda params: -self.score(params) / nobs
            try:
                hess = lambda params: -self.hessian(params) / nobs
            except:
                hess = None
            
            fit_funcs = {
                'bfgs': _fit_bfgs
            }

            func = fit_funcs[method]
            xopt, retvals = func(f, score, start_params, fargs, kwargs,
                                 disp=disp, maxiter=maxiter, callback=callback,
                                 retall=retall, full_output=full_output,
                                 hess=hess)

            if not full_output: # xopt should be None and retvals is argmin
                xopt = retvals
            elif cov_params_func:
                Hinv = cov_params_func(self, xopt, retvals)
            else:
                try:
                    Hinv = np.linalg.inv(-1 * self.hessian(xopt))
                except:
                    #might want custom warning ResultsWarning? NumericalWarning?
                    from warnings import warn
                    warndoc = ('Inverting hessian failed, no bse or '
                               'cov_params available')
                    warn(warndoc, Warning)
                    Hinv = None

            self.smoothness, self.threshold = xopt
            self.cov_hyperparams = Hinv


        res = self._fit((self.smoothness, self.threshold))

        return res.params, res.ssr
        
def _fit_bfgs(f, score, start_params, fargs, kwargs, disp=True,
                  maxiter=100, callback=None, retall=False,
                  full_output=True, hess=None):
    gtol = kwargs.setdefault('gtol', 1.0000000000000001e-05)
    norm = kwargs.setdefault('norm', np.Inf)
    epsilon = kwargs.setdefault('epsilon', 1.4901161193847656e-08)
    retvals = optimize.fmin_bfgs(f, start_params, score, args=fargs,
                                 gtol=gtol, norm=norm, epsilon=epsilon,
                                 maxiter=maxiter, full_output=full_output,
                                 disp=disp, retall=retall, callback=callback)
    if full_output:
        if not retall:
            xopt, fopt, gopt, Hinv, fcalls, gcalls, warnflag = retvals
        else:
            (xopt, fopt, gopt, Hinv, fcalls,
             gcalls, warnflag, allvecs) = retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'gopt': gopt, 'Hinv': Hinv,
                'fcalls': fcalls, 'gcalls': gcalls, 'warnflag':
                warnflag, 'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = None

    return xopt, retvals


if __name__=='__main__':
    import statsmodels.api as sm
    import pandas as pd
    from numpy.testing import assert_almost_equal
    
    # Load the sunspots dataset
    dta = sm.datasets.sunspots.load_pandas().data
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    dta = dta[dta.YEAR <= 1988]
    del dta["YEAR"]
    dta.SUNACTIVITY = 2*(np.sqrt(1 + dta.SUNACTIVITY) - 1)
    # Adjustments to match R sunspots dataset
    dta.SUNACTIVITY.iloc[[262, 280, 281, 287]] = [
        10.40967365,
        22.95596121,
        21.79075451,
        8.99090533
    ]
    
    # Test against tsDyn output, specifying the smoothness and threshold
    mod = LSTAR1(dta.SUNACTIVITY, 2, smoothness=0.4737064, threshold=7.67034)
    params, ssr = mod.fit()
    assert_almost_equal(ssr, 1352.506, 3)
    assert_almost_equal(params, (2.1401445, 1.5967946, -0.3190053, -3.94325109, -0.30183168, 0.01058212))
    
    # Test against tsDyn smoothness and threshold selection
    # Note: the optimizer is selecting slightly different smoothness
    #       and threshold parameters, which is why I can only assert
    #       almost equal to 2 decimal places. But the objective (ssr)
    #       is the same, so I think it's not a problem.
    mod = LSTAR1(dta.SUNACTIVITY, 2)
    params, ssr = mod.fit()
    assert_almost_equal(ssr, 1352.506, 3)
    assert_almost_equal(mod.smoothness, 0.4737064, 2)
    assert_almost_equal(mod.threshold, 7.67034, 2)
    assert_almost_equal(params, (2.1401445, 1.5967946, -0.3190053, -3.94325109, -0.30183168, 0.01058212), 3)
