"""various methods for setting limits given the observed and predicted counts"""

from dataclasses import dataclass


import numpy as np
from scipy.stats import chi2

def _single_bin_cls(expected_bkgd, obs_n_events):
    alpha = 0.05
    degrees_of_freedom = 2*(obs_n_events + 1)
    p = 1 - alpha*(1 - chi2.cdf(2*expected_bkgd, degrees_of_freedom))
    return 0.5*chi2.ppf(p, degrees_of_freedom) - expected_bkgd

@dataclass
class single_bin_cls:
    """Calculate the maximum number of expected signal events
    given the expected bkgd events and the number of events observed.

    From 'Statistics for Searches at the LHC' by Glen Cowan Section 10.
      https://arxiv.org/abs/1307.2487v1

    Using the Bayesian upper limit with the assumption of a flat prior,
    we can generate an equation for the upper limit on the expected
    number of (Poisson-distributed) signal events s_up. This equation
    can be shown to be equivalent to the CLs technique.

      s_up = (1/2)Finv(p) - b

    where
      
      b = expected number of (Poisson-distributed) bkgd events
      p = 1 - alpha(1-F(2b))
      alpha = 0.05 (for 95% confidence)
      Finv = inverse of chi-squared cumulative distribution function for 2(n+1) degrees of freedom
           = chi-squared percent point function (ppf)
      F = chi-squared cumulative distribution function (cdf) for 2(n+1) degrees of freedom
      n = number of events actually observed

    Parameters
    ----------
    expected_bkgd : float
        Number of bkgd events expected
    obs_n_events : float
        Number of events observed
    """
    num_toys: int = 500
    seed: int = 1

    def __post_init__(self):
        self.rng = np.random.default_rng(seed = self.seed)
    
    def __call__(self, expected_bkgd, bkgd_uncertainty, obs_n_events) :
        return np.median(
            _single_bin_cls(
                self.rng.normal(loc=expected_bkgd, scale=bkgd_uncertainty, size=self.num_toys),
                obs_n_events
            )
        )


@dataclass
class single_bin_toys:
    """Construct a null distribution using the expected background and its uncertainty
    and then set the upper limit to its 95% quantil subtracted by the observed
    number of events

    Generally found to be more conservative than the CLs technique.
    """

    num_toys: int = 10_000
    seed: int = 1

    def __post_init__(self):
        self.rng = np.random.default_rng(seed = self.seed)

    def __call__(self, expected_bkgd, bkgd_uncertainty, obs_n_events):
        toy_experiments = self.rng.poisson(
            lam = self.rng.normal(
                loc = expected_bkgd,
                scale = bkgd_uncertainty,
                size = self.num_toys
            )
        )
        return np.quantile(toy_experiments, [0.95])[0] - obs_n_events
