"""Evaluation functions for single and combined dataset analyses."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from .io import BlindPrediction, estimate_background_for_dataset
from .template import (
    build_template,
    cls_limit_for_amplitude,
    cls_amplitude_asymptotic,
    cls_amplitude_toys,
    _log_lr,
    _safe_mvn_draw,
)
from .statistics import p0_from_blind_vectors, p0_lognormal_poisson, fit_A_profiled_gaussian
from .conversion import epsilon2_from_A, A_from_epsilon2

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


@dataclass
class SingleResult:
    """Results from single-dataset evaluation."""

    dataset: str
    mass: float
    A_up: float
    eps2_up: float
    p0_analytic: float
    Z_analytic: float
    A_hat: float
    sigma_A: float
    extract_success: bool


@dataclass
class CombinedResult:
    """Results from combined dataset evaluation."""

    mass: float
    eps2_up: float
    p0_analytic: float
    Z_analytic: float


def evaluate_single_dataset(
    ds: "DatasetConfig",
    mass: float,
    config: "Config",
    do_extraction: bool = True,
) -> Tuple[SingleResult, BlindPrediction]:
    """Evaluate a single dataset at one mass hypothesis.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis (GeV)
        config: Global configuration
        do_extraction: Whether to perform signal extraction fit

    Returns:
        Tuple of (SingleResult, BlindPrediction)
    """
    pred = estimate_background_for_dataset(ds, mass, config)

    A_up, _ = cls_limit_for_amplitude(
        n_obs=pred.obs,
        b_mean=pred.mu,
        b_cov=pred.cov,
        edges=pred.edges,
        mass=mass,
        sigma_val=pred.sigma_val,
        config=config,
        seed=1,
    )

    eps2_up = epsilon2_from_A(ds, mass, A_up, pred.integral_density)

    p0, Z = p0_from_blind_vectors(pred.obs, pred.mu, pred.cov)

    if do_extraction:
        tmpl = build_template(pred.edges, mass, pred.sigma_val)
        fit = fit_A_profiled_gaussian(
            pred.obs,
            pred.mu,
            pred.cov,
            tmpl,
            allow_negative=config.extract_allow_negative,
        )
        A_hat = float(fit["A_hat"])
        sigma_A = float(fit["sigma_A"])
        ok = bool(fit["success"])
    else:
        A_hat, sigma_A, ok = float("nan"), float("nan"), False

    return (
        SingleResult(
            ds.key,
            float(mass),
            float(A_up),
            float(eps2_up),
            float(p0),
            float(Z),
            float(A_hat),
            float(sigma_A),
            bool(ok),
        ),
        pred,
    )


def _concat_block_diag(covs: List[np.ndarray]) -> np.ndarray:
    """Concatenate covariance matrices into block diagonal."""
    sizes = [c.shape[0] for c in covs]
    out = np.zeros((sum(sizes), sum(sizes)), float)
    i = 0
    for C in covs:
        n = C.shape[0]
        out[i : i + n, i : i + n] = C
        i += n
    return out


def combined_cls_limit_epsilon2(
    mass: float,
    ds_list: List["DatasetConfig"],
    preds: List[BlindPrediction],
    config: "Config",
    seed: int = 1,
) -> float:
    """Compute combined CLs limit on epsilon^2.

    Args:
        mass: Signal mass hypothesis (GeV)
        ds_list: List of dataset configurations
        preds: List of background predictions
        config: Global configuration
        seed: Random seed

    Returns:
        Upper limit on epsilon^2
    """
    obs = np.concatenate([p.obs for p in preds]).astype(int)
    b = np.concatenate([p.mu for p in preds]).astype(float)
    cov = _concat_block_diag([p.cov for p in preds]).astype(float)

    templates = [build_template(p.edges, mass, p.sigma_val) for p in preds]
    Ks = [
        A_from_epsilon2(ds, mass, 1.0, p.integral_density)
        for ds, p in zip(ds_list, preds)
    ]
    s_unit = np.concatenate([K * t for K, t in zip(Ks, templates)]).astype(float)

    rng = np.random.default_rng(seed)
    alpha = config.cls_alpha
    mode = config.cls_mode
    num_toys = config.cls_num_toys

    def cls_at_eps2(eps2: float) -> float:
        eps2 = float(max(eps2, 0.0))
        s = eps2 * s_unit

        if mode == "asymptotic":
            return cls_amplitude_asymptotic(1.0, obs, b, cov, s)[0]

        lnQ_obs = float(_log_lr(obs, b, s))
        b_toys = _safe_mvn_draw(b, cov, size=int(num_toys), rng=rng)

        n_b = rng.poisson(b_toys)
        lnQ_b = _log_lr(n_b, b_toys, s)

        sb_means = b_toys + s
        n_sb = rng.poisson(sb_means)
        lnQ_sb = _log_lr(n_sb, b_toys, s)

        CL_b = float(np.mean(lnQ_b <= lnQ_obs))
        CL_sb = float(np.mean(lnQ_sb <= lnQ_obs))
        CL_b = max(CL_b, 1e-9)
        return CL_sb / CL_b

    # Bracket + bisect
    eps_lo = 0.0
    eps_hi = 1e-10
    cls_hi = cls_at_eps2(eps_hi)
    it = 0

    while cls_hi > alpha and eps_hi < 1.0 and it < 80:
        eps_hi *= 2.0
        cls_hi = cls_at_eps2(eps_hi)
        it += 1

    for _ in range(80):
        mid = 0.5 * (eps_lo + eps_hi)
        c = cls_at_eps2(mid)
        if abs(c - alpha) < 1e-2:
            eps_lo = eps_hi = mid
            break
        if c > alpha:
            eps_lo = mid
        else:
            eps_hi = mid
        if abs(eps_hi - eps_lo) <= max(1e-16, 1e-3 * eps_hi):
            break

    return float(0.5 * (eps_lo + eps_hi))


def evaluate_combined(
    mass: float,
    ds_list: List["DatasetConfig"],
    preds: List[BlindPrediction],
    config: "Config",
) -> CombinedResult:
    """Evaluate combined datasets at one mass hypothesis.

    Args:
        mass: Signal mass hypothesis (GeV)
        ds_list: List of dataset configurations
        preds: List of background predictions
        config: Global configuration

    Returns:
        CombinedResult with combined upper limit and p-value
    """
    eps2_up = combined_cls_limit_epsilon2(mass, ds_list, preds, config)

    obs_sum = int(sum(int(np.sum(p.obs)) for p in preds))
    mu_sum = float(sum(float(np.sum(p.mu)) for p in preds))
    cov_sum = float(sum(float(np.sum(p.cov)) for p in preds))

    p0 = p0_lognormal_poisson(obs_sum, mu_sum, math.sqrt(max(cov_sum, 1e-12)))
    p0 = min(max(float(p0), 0.0), 1.0)

    if 0.0 < p0 < 1.0:
        Z = float(norm.isf(p0))
    elif p0 == 0.0:
        Z = math.inf
    else:
        Z = 0.0

    return CombinedResult(float(mass), float(eps2_up), float(p0), float(Z))
