import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import uproot
import hist
import scipy

from ._fit import fit, kernels
from . import _hist
from ._plot import plt, label
from ._mass_resolution import mass_resolution_2016_bump_hunt as mass_resolution
from . import _limit_setting as limit_setting


class GaussianProcessModel:
    """Model the IMD with a Gaussian Process fit (GP)

    Parameters
    ----------
    h : hist.Hist|str
        specification of histogram
        can provide a hist.Hist or a string naming a specific one
    kernel :
        GP kernel to use for model
    blind_range: 2-tuple, optional, default None
        range of histogram to blind model to during fit
    modify_histogram: Callable
        function to modify histogram after it is loaded but before it is fitted
        could (for example) rebin, inject a signal bump, or limit the fitting range
    kwargs: dict[str,Any]
        rest of keyword arguments passed to the function fit
    """

    def __init__(
        self,
        h: Union[hist.Hist, str],
        kernel, *,
        blind_range = None, #= input_mass-sigma, imput_mass+sigma
        modify_histogram = None,
        **kwargs
    ):
        self.histogram = _hist.io._deduce_histogram(h)
        if modify_histogram is not None:
            if isinstance(modify_histogram, (list,tuple)):
                for func in modify_histogram:
                    self.histogram = func(self.histogram)
            elif callable(modify_histogram):
                self.histogram = modify_histogram(self.histogram)
            else:
                raise TypeError('modify_histogram is not a Callable or list/tuple of Callables.')
        self.blind_range = blind_range # store for plotting purposes
        self.model = fit(
            self.histogram,
            kernel,
            blind_range = self.blind_range,
            **kwargs
        )
        self.mean_pred, self.std_pred = self.predict(self.histogram.axes[0].centers)
        self.chi2_statistic, self.p_value = scipy.stats.chisquare(
            self.histogram.values(),
            np.sum(self.histogram.values())/np.sum(self.mean_pred) * self.mean_pred
        )


    def predict(self, mass):
        """Evaluate the model and the input mass, returning its prediction and uncertainty
        for the event yield

        Parameters
        ----------
        mass: np.array
            array of masses to evaluate in GeV

        Return
        ------
        2-tuple of np.array
            the mean prediction and the uncertainty on that prediction for the event yield
            at the input masses
        """
        return self.model.predict(mass.reshape(-1,1), return_std=True)


    @property
    def blind_range_indices(self):
        return self.histogram.axes[0].index(self.blind_range)


    @property
    def blind_single_bin_prediction(self):
        if self.blind_range is None:
            raise ValueError('Cannot predict into a blind region when no bliding was done!')
        pred_val = np.sum(self.mean_pred[slice(*self.blind_range_indices)])
        pred_unc = np.sqrt(np.sum(self.std_pred[slice(*self.blind_range_indices)]**2))
        return pred_val, pred_unc
    

    @property
    def blind_single_bin_observation(self):
        if self.blind_range is None:
            raise ValueError('Cannot observe what is in the blind region when no blinding was done!')
        return np.sum(self.histogram.values()[slice(*self.blind_range_indices)])


    def search_in_blind_region(self):
        """calculate the test statistic and the resulting p-value looking into the blind region

        This is a very simplified statistical treatment where we basically treat the entire
        blind region as a single bin. Inside that single bin, we have a prediction (and its
        uncertainty) from our background model and the observed data yield. The pull value
        (Data - Pred)/Err is then used as a test statistic where the null distribution is
        approximated by a simple, standard normal.

        Assumptions
        -----------
        - The yield is high enough so a normal approximation is appropriate. We satisfy
          this assumption since the yield within our bins is at minimum several thousand.

        We get this test statistic (the single-bin pull value) by simply summing over the pull values
        that fall within the blinded region.
        The p-value of this test statistic is then calculated using the normal
        distributions survival function (scipy.stats.norm.sf) which is equivalent to integrating
        from the input value up to positive infinity (i.e. the probability that the observed test
        statistic or anything more extreme is observed).
        """
        test_statistic = np.sum(self.pull[slice(*self.blind_range_indices)])
        p_value = scipy.stats.norm.sf(test_statistic)
        return test_statistic, p_value

    
    def upper_limit_in_blind_region(
        self,
        calculator = limit_setting.single_bin_cls()
    ):
        """estimate the upper limit within the blinded region in a single-bin
        (no templating) model

        Various statistical calculations are possible and some options
        exist in the limit_setting submodule. The default is a technique
        equivalent to CLs.
        """
        exp_val = np.sum(self.mean_pred[slice(*self.blind_range_indices)])
        exp_err = np.sqrt(np.sum(self.std_pred[slice(*self.blind_range_indices)]**2))
        obs = np.sum(self.histogram.values()[slice(*self.blind_range_indices)])
        return calculator(exp_val, exp_err, obs)


    @property
    def kernel(self):
        """Get the _fit_ kernel and not the kernel passed as a parameter"""
        return self.model.kernel_
    
    @property
    def combined_variance(self):
        return self.histogram.variances()+self.std_pred**2

    @property
    def positive_prediction(self):
        return (self.mean_pred > 0)&(self.histogram.values() > 0)

    @property
    def ratio(self):
        vals = (
            self.histogram.values()[self.positive_prediction]
            /self.mean_pred[self.positive_prediction]
        )
        err = vals*np.sqrt(
            self.std_pred[self.positive_prediction]**2/self.mean_pred[self.positive_prediction]**2
            +self.histogram.variances()[self.positive_prediction]/self.histogram.values()[self.positive_prediction]**2
        )
        return vals, err


    @property
    def pull(self):
        return (
            (self.histogram.values()-self.mean_pred)[self.positive_prediction]
            /np.sqrt(self.combined_variance[self.positive_prediction])
        )


    def plot_histogram(self, **kwargs):
        kwargs.setdefault('flow', None)
        kwargs.setdefault('label', 'Observed Data')
        art = self.histogram.plot(
            **kwargs
        )
        kwargs.get('ax', plt.gca()).set_ylabel('Event Yield')
        return art


    def plot_prediction(self, ax = None, **kwargs):
        kwargs.setdefault(
            'label',
            '\n'.join([
                'GP with 95% Confidence Interval',
                rf'$\chi^2 = ${self.chi2_statistic:.3g}',
                r'$P_{\chi^2} = $'+f'{self.p_value:.3g}'
            ])
        )
        x = self.histogram.axes[0].centers
        if ax is None:
            ax = plt.gca()
        line_art, = ax.plot(x, self.mean_pred, **kwargs)
        fill_art = ax.fill_between(
             x, self.mean_pred - 1.96*self.std_pred, self.mean_pred + 1.96*self.std_pred,
             alpha = 0.5, color = line_art.get_color()
        )
        return line_art, fill_art


    def plot_pull_histogram(self, **kwargs):
        kwargs.setdefault('yerr', False)
        art = (
            hist.Hist.new
            .Reg(41,-10,10,label=r'Pull $(\mathrm{Data}-\mathrm{Fit})/\sigma$')
            .Double()
            .fill(self.pull)
        ).plot(**kwargs)
        kwargs.get('ax', plt.gca()).set_ylabel('Bin Count')
        return art
    

    def plot_comparison(self):
        """Plot a comparison between the histogram and the Fit"""
    
        fig, axes = plt.subplots(
            nrows = 3,
            height_ratios = [2, 1, 1],
            sharex = 'col',
            gridspec_kw = dict(
                hspace = 0.05
            ),
            figsize = (10,12)
        )

        raw, ratio, pull = axes

        # RAW
        x = self.histogram.axes[0].centers[self.positive_prediction]
        # hist has plotting methods already
        #   add label (for legend) and don't show the flow bins
        #   (default is to draw a little arrow hinting that something exists out there)
        self.plot_histogram(ax=raw)
        self.plot_prediction(ax=raw)
        raw.legend(
            title = f'Kernel: {repr(self.model.kernel_)}',
            title_fontsize = 'xx-small'
        )
        # mpl default is to add some horizontal padding which I don't like
        raw.set_xlim(self.histogram.axes[0].edges[[0,-1]])
        label(ax=raw)

        # RATIO
        ratio_vals, ratio_err = self.ratio
        ratio.plot(x, ratio_vals)
        ratio.fill_between(x, ratio_vals - 1.96*ratio_err, ratio_vals + 1.96*ratio_err, alpha=0.5)
        ratio.axhline(1, color='gray', ls=':')
        ratio.set_ylabel(r'Data / Fit')
        #ratio.set_ylim(0, 2)

        # PULL
        pull.plot(x, self.pull)
        pull.fill_between(
            x, np.full(x.shape, -2), np.full(x.shape, +2),
            color='gray', alpha=0.5
        )
        pull.set_ylabel(r'$(\mathrm{Data} - \mathrm{Fit})/\sigma$')
        pull.set_ylim(-5.5,5.5)

        # FINAL CLEANUP

        for ax in axes[:-1]:
            ax.set_xlabel(None) # undo labeling to avoid ugliness
        axes[-1].set_xlabel(self.histogram.axes[0].label)
        
        if self.blind_range is not None:
            for e in self.blind_range:
                for ax in axes:
                    ax.axvline(e, color='tab:red')

        return fig, axes
