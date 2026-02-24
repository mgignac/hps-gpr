"""Histogram loading and per-dataset background estimation."""

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from .gpr import fit_gpr, predict_counts_from_log_gpr

if TYPE_CHECKING:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from .config import Config
    from .dataset import DatasetConfig


@dataclass
class BlindPrediction:
    """Background prediction results for a blind window."""

    mu: np.ndarray  # Background mean in blind window
    cov: np.ndarray  # Background covariance in blind window
    obs: np.ndarray  # Observed counts in blind window
    edges: np.ndarray  # Bin edges in blind window
    sigma_val: float  # Mass resolution
    blind: Tuple[float, float]  # Blind window bounds

    x_full: np.ndarray  # All bin centers
    y_full: np.ndarray  # All observed counts
    mu_full: np.ndarray  # Background prediction for all bins
    edges_full: np.ndarray  # All bin edges

    integral_density: float  # Counts per GeV in signal region


def _gp_model(h, kernel, **kwargs):
    """Compatibility wrapper for gp.GaussianProcessModel.

    Some versions of the local gp package require kernel as a mandatory
    (possibly positional) argument.
    """
    import gp

    # Try keyword kernel + keyword h
    try:
        return gp.GaussianProcessModel(h=h, kernel=kernel, **kwargs)
    except TypeError as e1:
        # Try positional kernel, keyword h
        try:
            return gp.GaussianProcessModel(kernel, h=h, **kwargs)
        except TypeError:
            # Try positional kernel, positional h
            try:
                return gp.GaussianProcessModel(kernel, h, **kwargs)
            except TypeError:
                raise e1


def _build_model(
    ds: "DatasetConfig",
    blind: Tuple[float, float],
    rebin: int,
    config: "Config",
):
    """Build the gp model for a dataset."""
    import gp

    kernel = config.get_kernel()

    # Probe histogram edges
    probe = _gp_model((ds.root_path, ds.hist_name), kernel)
    edges_all = np.asarray(probe.histogram.axes[0].edges, float)
    first_edge = float(edges_all[0])
    last_edge = float(edges_all[-1])

    # Clamp to dataset analysis range
    lower = max(first_edge, ds.m_low)
    upper = min(last_edge, ds.m_high)

    manip = gp._hist.manipulation.rebin_and_limit(int(rebin), lower, upper)

    model = _gp_model(
        (ds.root_path, ds.hist_name),
        kernel,
        n_restarts_optimizer=0,
        blind_range=blind,
        modify_histogram=manip,
    )
    return model


def _blind_pred_detail(
    model,
    gpr: "GaussianProcessRegressor",
    blind: Tuple[float, float],
    config: "Config",
):
    """Extract prediction details for the blind window."""
    Xc = model.histogram.axes[0].centers
    vals = model.histogram.values().astype(int)
    edges = np.asarray(model.histogram.axes[0].edges, dtype=float)

    msk = (Xc >= blind[0]) & (Xc <= blind[1])
    idx = np.where(msk)[0]
    if idx.size == 0:
        raise RuntimeError("Blind window has no bins")

    Xb = Xc[msk]
    e_slice = edges[idx[0] : idx[-1] + 2]

    mu, cov = predict_counts_from_log_gpr(gpr, Xb, config)
    obs = vals[msk]

    return (
        np.asarray(mu, float),
        np.asarray(cov, float),
        np.asarray(obs, int),
        np.asarray(e_slice, float),
    )


def _compute_integral_density(model, mass: float, sigma_val: float) -> float:
    """Compute counts per GeV in +/- 2*sigma region."""
    ax = model.histogram.axes[0]
    vals = model.histogram.values().astype(float)
    nb = ax.size

    lo, hi = mass - 2.0 * sigma_val, mass + 2.0 * sigma_val
    i0 = max(0, int(ax.index(lo)))
    i1 = min(int(nb), int(ax.index(hi)) + 1)

    if i1 <= i0:
        i0 = max(0, min(i0, nb - 1))
        i1 = i0 + 1

    integral_counts = float(np.sum(vals[i0:i1]))
    widths = np.asarray(ax.widths, dtype=float)

    if widths.ndim == 0:
        integral_size = (i1 - i0) * float(widths)
    else:
        integral_size = float(np.sum(widths[i0:i1]))

    if not np.isfinite(integral_size) or integral_size <= 0:
        raise ValueError("Non-positive integral_size")

    return float(integral_counts / integral_size)


def estimate_background_for_dataset(
    ds: "DatasetConfig",
    mass: float,
    config: "Config",
    rebin: int = None,
    restarts: int = None,
    train_exclude_nsigma: Optional[float] = None,
) -> BlindPrediction:
    """Estimate background for a dataset at a given mass.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis (GeV)
        config: Global configuration
        rebin: Rebinning factor (defaults to config.neighborhood_rebin)
        restarts: Number of GPR restarts (defaults to config.n_restarts)
        train_exclude_nsigma: Half-width of GP training exclusion in sigma units.
            Defaults to config.gp_train_exclude_nsigma (or config.blind_nsigma if
            gp_train_exclude_nsigma is None). The extraction blind window always uses
            config.blind_nsigma; only the GP training mask is affected.

    Returns:
        BlindPrediction with background estimates
    """
    if rebin is None:
        rebin = config.neighborhood_rebin
    if restarts is None:
        restarts = config.n_restarts

    sigma_val = ds.sigma(mass)
    blind = (
        mass - config.blind_nsigma * sigma_val,
        mass + config.blind_nsigma * sigma_val,
    )

    # GP training exclusion window (may differ from extraction blind window)
    if train_exclude_nsigma is None:
        train_exclude_nsigma = float(
            getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma
        )
    blind_train = (
        mass - float(train_exclude_nsigma) * sigma_val,
        mass + float(train_exclude_nsigma) * sigma_val,
    )

    model = _build_model(ds, blind, rebin=rebin, config=config)

    X = model.histogram.axes[0].centers
    y = model.histogram.values().astype(float)

    mask_train = (X < blind_train[0]) | (X > blind_train[1])
    X_train = X[mask_train]
    y_train = y[mask_train]

    gpr = fit_gpr(X_train, y_train, config, restarts=int(restarts))

    mu_blind, cov_blind, obs_blind, edges_blind = _blind_pred_detail(
        model, gpr, blind, config
    )

    mu_full, _ = predict_counts_from_log_gpr(gpr, X, config)

    integral_density = _compute_integral_density(model, mass, sigma_val)

    return BlindPrediction(
        mu=mu_blind,
        cov=cov_blind,
        obs=obs_blind,
        edges=edges_blind,
        sigma_val=sigma_val,
        blind=blind,
        x_full=np.asarray(X, float),
        y_full=np.asarray(y, float),
        mu_full=np.asarray(mu_full, float),
        edges_full=np.asarray(model.histogram.axes[0].edges, float),
        integral_density=integral_density,
    )
