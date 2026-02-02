from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import hist
import numpy as np


def fit(
    histogram: hist.Hist,
    kernel,
    blind_range = None,
    empty_bin_variance = None,
    **kwargs
) -> GaussianProcessRegressor:
    """fit the input histogram with a GP using the input kernel

    optionally, blind the fit to a range of the histogram

    Parameters
    ----------
    histogram: hist.Hist
        histogram to fit a GP to
    kernel:
        kernel to use in GP
    blind_range: 2-tuple, optional, default None
        range of histogram to "blind" the fit to
        (i.e. do /not/ use this range of values in fit)
    empty_bin_variance: float|None
        specify how we should handle empty bins in the original histogram.
        If None, drop the empty bins and only fit to non-empty bins.
        If float, fit to empty bins using 0 for their value and the passed
        number to be their variance.
    kwargs: dict[str,Any]
        all the rest of the keyword arguments are passed to GaussianProcessRegressor


    Note
    ----
    The running of this fit can take a while. The GP is manipulating N-dimensional
    matrices where N corresponds to the number of training points. Since there are
    6000 bins in the input invariant mass histogram, N=6000 which are really large
    matrices and can take a long time to run.
    The manipulation of large matrices is a problem _built_ for GPUs and hence
    we may want to switch to a GPU-possible implementation of GPs like GPyTorch[1].
    In the mean time, I would highly recommend pickling[2] the resulting fitted GP object
    so that the fit doesn't need to be re-run if you just want to make different plots
    with the predictions the GP makes.

        import pickle
        with open('helpful-name-for-this-gp-fit.pkl','wb') as f:
            pickle.dump(gp, f)

    and then somewhere else (e.g. in a notebook where you are playing with plots) you can

        import pickle
        with open('helpful-name-for-this-gp-fit.pkl','rb') as f:
            gp = pickle.load(f)


    [1]: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    [2]: https://docs.python.org/3/library/pickle.html
    """

    x     = histogram.axes[0].centers
    value = histogram.values()
    variance = histogram.variances()

    fit_selection = np.full(value.shape, True)
    if empty_bin_variance is None:
        fit_selection = (value > 0.0)
    else:
        # set the variance on empty bins to be the 95% upper limit of a Poisson mean
        # when observing 0
        variance[value < 1.0] = 3.6889 
    if blind_range is not None:
        if isinstance(blind_range, (tuple,list)) and len(blind_range)==2:
            fit_selection = fit_selection&((x < blind_range[0])|(x > blind_range[1]))
        else:
            raise ValueError('blind_range is not a length-2 tuple defining the range of coordinates to blind')

    x_train = x[fit_selection]
    y_train = value[fit_selection]
    variance = variance[fit_selection]

    if 'alpha' in kwargs:
        raise KeyError('alpha cannot be manually set. It is determined to be the variance of the histogram to fit')

    # update n_restarts_optimizer default to be 9 as is used in example
    kwargs.setdefault('n_restarts_optimizer', 9)
    _gp = GaussianProcessRegressor(
        kernel = kernel,
    #    alpha = variance,
        **kwargs
    )
    # fit expects a _column_ for x and a _row_ for y so we need to reshape x
    _gp.fit(x_train.reshape(-1,1), y_train)
    return _gp
