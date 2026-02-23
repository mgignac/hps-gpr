"""Gaussian Process Regression preprocessing, kernel policy, and fitting."""

import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.base import clone as sk_clone
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as skgp

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


# ---------------------------------------------------------------------------
# Noise-variance / preprocessing
# ---------------------------------------------------------------------------

def alpha_var_log_from_counts(
    y: np.ndarray, config: "Config"
) -> np.ndarray:
    """Compute alpha (noise variance) for log-space GPR."""
    y = np.asarray(y, dtype=float)
    alpha = np.full_like(y, config.pre_zero_alpha, dtype=float)
    pos = y > 0.0

    if config.alpha_model == "1/y":
        alpha[pos] = 1.0 / np.clip(y[pos], 1e-12, None)
    else:
        alpha[pos] = 1.0 / np.clip(y[pos], 1.0, None)

    if config.pre_alpha_first_n > 0:
        k = min(config.pre_alpha_first_n, alpha.size)
        alpha[:k] *= config.pre_alpha_first_factor

    return alpha


def preprocess_xy_for_gpr(
    X: np.ndarray, y: np.ndarray, config: "Config"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess X, y arrays for GPR fitting.

    Optionally transforms to log space and computes alpha values.

    Returns:
        Tuple of (X_in, y_in, alpha)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_in = np.log(np.clip(X, 1e-12, None)) if config.pre_log else X.copy()
    y_in = np.zeros_like(y, dtype=float)

    pos = y > 0.0
    if config.pre_log:
        y_in[pos] = np.log(y[pos])
        alpha = alpha_var_log_from_counts(y, config)
    else:
        y_in = y.copy()
        alpha = np.clip(y, 1.0, None)

    return X_in, y_in, alpha


# ---------------------------------------------------------------------------
# Resolution-scaled kernel policy
# ---------------------------------------------------------------------------

_SIGMA_X_STAT_CACHE: Dict[tuple, float] = {}


def _sigma_x_from_sigma(
    m: np.ndarray, sigma: np.ndarray, pre_log: bool
) -> np.ndarray:
    """Convert detector mass resolution σ(m) [GeV] to x-space scale σ_x(m).

    The GP is trained in x = ln(m) if pre_log=True, else x = m.
    We define σ_x(m) = x(m + σ(m)) - x(m).

    For pre_log=True and σ << m:  σ_x ≈ σ/m.
    For pre_log=False:            σ_x = σ (GeV).
    """
    m = np.clip(np.asarray(m, float), 1e-12, None)
    sigma = np.clip(np.asarray(sigma, float), 0.0, None)
    if pre_log:
        return np.log(m + sigma) - np.log(m)
    return sigma


def length_scale_x_to_mass_delta(ls_x: float, mass: float, pre_log: bool) -> float:
    """Convert a GP x-space length-scale to an intuitive GeV-equivalent Δm.

    pre_log=True:  x = ln(m)  →  Δm ≈ m·(exp(ℓ) - 1)
    pre_log=False: x = m      →  Δm = ℓ
    """
    m = float(mass)
    lx = float(ls_x)
    if not np.isfinite(m) or m <= 0.0 or not np.isfinite(lx):
        return float("nan")
    if pre_log:
        return float(m * (np.exp(lx) - 1.0))
    return float(lx)


def _kernel_bounds_from_resolution_global(
    ds: "DatasetConfig",
    *,
    upper_factor: float,
    lower_factor: float,
    stat: str,
    npts: int,
    pre_log: bool,
) -> Tuple[float, float, float]:
    """Compute (ls_lower, ls_upper, base_sigma_x) in x-units using a dataset-wide statistic.

    Samples σ(m) across the dataset mass range and picks a representative σ_x value.
    Bounds do not vary with the mass hypothesis (legacy v11/v12 behavior).
    """
    stat = str(stat).lower().strip()
    npts = int(max(10, npts))
    mgrid = np.linspace(float(ds.m_low), float(ds.m_high), npts)
    sgrid = np.array([float(ds.sigma(float(m))) for m in mgrid], dtype=float)
    sx = _sigma_x_from_sigma(mgrid, sgrid, pre_log)
    sx = sx[np.isfinite(sx)]

    if sx.size == 0:
        base = 1.0
    elif stat == "mean":
        base = float(np.mean(sx))
    elif stat == "min":
        base = float(np.min(sx))
    elif stat == "max":
        base = float(np.max(sx))
    else:  # default: "median"
        base = float(np.median(sx))

    base = max(base, 1e-12)
    ls_lo = max(1e-12, float(lower_factor) * base)
    ls_hi = max(ls_lo * 1.001, float(upper_factor) * base)
    return ls_lo, ls_hi, base


def _kernel_bounds_from_resolution_local(
    ds: "DatasetConfig",
    mass: float,
    *,
    upper_factor: float,
    lower_factor: float,
    pre_log: bool,
) -> Tuple[float, float, float]:
    """Compute (ls_lower, ls_upper, base_sigma_x) in x-units using σ(mass).

    Bounds are tied to the local mass resolution at the scan hypothesis.
    """
    m = float(mass)
    s = float(ds.sigma(m))
    base = float(_sigma_x_from_sigma(np.array([m], float), np.array([s], float), pre_log)[0])
    base = max(base, 1e-12)
    ls_lo = max(1e-12, float(lower_factor) * base)
    ls_hi = max(ls_lo * 1.001, float(upper_factor) * base)
    return ls_lo, ls_hi, base


def _sigma_x_stat_cached(
    ds: "DatasetConfig", *, stat: str, npts: int, pre_log: bool
) -> float:
    """Dataset-wide typical σ_x statistic (cached)."""
    key = (str(getattr(ds, "key", "")), bool(pre_log), str(stat).lower().strip(), int(npts))
    if key in _SIGMA_X_STAT_CACHE:
        return float(_SIGMA_X_STAT_CACHE[key])
    _, _, base = _kernel_bounds_from_resolution_global(
        ds, upper_factor=1.0, lower_factor=1.0, stat=stat, npts=npts, pre_log=pre_log
    )
    _SIGMA_X_STAT_CACHE[key] = float(base)
    return float(base)


def compute_kernel_ls_bounds(
    ds: "DatasetConfig",
    config: "Config",
    *,
    mass: Optional[float] = None,
    policy: Optional[str] = None,
    upper_factor: Optional[float] = None,
    lower_factor: Optional[float] = None,
    stat: Optional[str] = None,
    npts: Optional[int] = None,
) -> Dict[str, float]:
    """Compute RBF length-scale bounds/init for a dataset (and optionally a mass hypothesis).

    Returns dict with: ls_lo, ls_hi, ls_init, sigma_x, policy_used.
    """
    pol = str(policy if policy is not None else config.kernel_ls_policy).lower().strip()
    if pol == "resolution_scaled":
        pol = "resolution_scaled_local"

    ds_key = str(getattr(ds, "key", ""))
    pre_log = bool(config.pre_log)

    uf_by = dict(config.kernel_ls_res_upper_factor_by_dataset or {})
    lf_by = dict(config.kernel_ls_res_lower_factor_by_dataset or {})
    st_by = dict(getattr(config, "kernel_ls_res_stat_by_dataset", {}) or {})
    np_by = dict(getattr(config, "kernel_ls_res_npts_by_dataset", {}) or {})

    uf = float(upper_factor if upper_factor is not None else uf_by.get(ds_key, config.kernel_ls_res_upper_factor))
    lf = float(lower_factor if lower_factor is not None else lf_by.get(ds_key, config.kernel_ls_res_lower_factor))
    st = str(stat if stat is not None else st_by.get(ds_key, config.kernel_ls_res_stat))
    np_ = int(npts if npts is not None else np_by.get(ds_key, config.kernel_ls_res_npts))

    if pol == "manual":
        ls_lo, ls_hi = float(config.kernel_ls_bounds[0]), float(config.kernel_ls_bounds[1])
        if ls_hi <= ls_lo:
            raise ValueError(f"kernel_ls_bounds must satisfy hi>lo; got {config.kernel_ls_bounds}")
        ls_init = float(min(max(config.kernel_ls_init, ls_lo), ls_hi))
        return dict(ls_lo=ls_lo, ls_hi=ls_hi, ls_init=ls_init, sigma_x=float("nan"), policy_used=pol)

    if pol == "resolution_scaled_local":
        if mass is None:
            # No mass provided — fall back to global
            ls_lo, ls_hi, base = _kernel_bounds_from_resolution_global(
                ds, upper_factor=uf, lower_factor=lf, stat=st, npts=np_, pre_log=pre_log
            )
            ls_init = float(np.sqrt(ls_lo * ls_hi))
            return dict(ls_lo=float(ls_lo), ls_hi=float(ls_hi), ls_init=float(ls_init),
                        sigma_x=float(base), policy_used="resolution_scaled_global")

        ls_lo, ls_hi, base = _kernel_bounds_from_resolution_local(
            ds, float(mass), upper_factor=uf, lower_factor=lf, pre_log=pre_log
        )

        # Optional floor/cap on ℓ_hi
        floor_mode = str(getattr(config, "kernel_ls_local_hi_floor_mode", "none")).lower().strip()
        if floor_mode == "dataset_stat":
            try:
                st_val = getattr(config, "kernel_ls_res_stat", "median")
                np_val = int(getattr(config, "kernel_ls_res_npts", 200))
                ff = float(getattr(config, "kernel_ls_local_hi_floor_factor", 1.0))
                base_stat = _sigma_x_stat_cached(ds, stat=st_val, npts=np_val, pre_log=pre_log)
                hi_floor = uf * base_stat * ff
                if np.isfinite(hi_floor):
                    ls_hi = max(float(ls_hi), float(hi_floor))
            except Exception:
                pass

        cap_frac = getattr(config, "kernel_ls_local_hi_cap_xrange_frac", None)
        if cap_frac is not None:
            try:
                cap_frac = float(cap_frac)
                mlo = float(getattr(ds, "data_low", None) or ds.m_low)
                mhi = float(getattr(ds, "data_high", None) or ds.m_high)
                xlo = float(np.log(max(mlo, 1e-12)) if pre_log else mlo)
                xhi = float(np.log(max(mhi, 1e-12)) if pre_log else mhi)
                cap = cap_frac * abs(xhi - xlo)
                if np.isfinite(cap) and cap > 0:
                    ls_hi = min(float(ls_hi), float(cap))
            except Exception:
                pass

        ls_hi = max(float(ls_hi), float(ls_lo) * 1.001)
        ls_init = float(np.sqrt(ls_lo * ls_hi))
        return dict(ls_lo=float(ls_lo), ls_hi=float(ls_hi), ls_init=float(ls_init),
                    sigma_x=float(base), policy_used=pol)

    if pol == "resolution_scaled_global":
        ls_lo, ls_hi, base = _kernel_bounds_from_resolution_global(
            ds, upper_factor=uf, lower_factor=lf, stat=st, npts=np_, pre_log=pre_log
        )
        ls_init = float(np.sqrt(ls_lo * ls_hi))
        return dict(ls_lo=float(ls_lo), ls_hi=float(ls_hi), ls_init=float(ls_init),
                    sigma_x=float(base), policy_used=pol)

    raise ValueError(
        f"Unknown kernel_ls_policy='{pol}' "
        f"(expected 'manual', 'resolution_scaled_local', or 'resolution_scaled_global')"
    )


def make_kernel_for_dataset(
    ds: "DatasetConfig",
    config: "Config",
    *,
    mass: Optional[float] = None,
    policy: Optional[str] = None,
    upper_factor: Optional[float] = None,
    lower_factor: Optional[float] = None,
    stat: Optional[str] = None,
    npts: Optional[int] = None,
):
    """Create a fresh ConstantKernel * RBF kernel for a given DatasetConfig.

    Priority order:
      1. Explicit per-dataset numeric bounds (kernel_ls_bounds_by_dataset)
      2. Policy-based computation via compute_kernel_ls_bounds()
    """
    ds_key = str(getattr(ds, "key", ""))
    bounds_by_ds = dict(getattr(config, "kernel_ls_bounds_by_dataset", {}) or {})
    init_by_ds = dict(getattr(config, "kernel_ls_init_by_dataset", {}) or {})

    if ds_key in bounds_by_ds:
        b = bounds_by_ds[ds_key]
        if b is not None and len(b) == 2:
            ls_lo, ls_hi = float(b[0]), float(b[1])
            if ls_hi <= ls_lo:
                raise ValueError(f"kernel_ls_bounds_by_dataset[{ds_key}] must satisfy hi>lo")
            ls_init = float(init_by_ds.get(ds_key, math.sqrt(ls_lo * ls_hi)))
            ls_init = float(min(max(ls_init, ls_lo), ls_hi))
            return (
                skgp.kernels.ConstantKernel(config.kernel_constant_init, config.kernel_constant_bounds)
                * skgp.kernels.RBF(length_scale=ls_init, length_scale_bounds=(ls_lo, ls_hi))
            )

    info = compute_kernel_ls_bounds(
        ds, config, mass=mass, policy=policy,
        upper_factor=upper_factor, lower_factor=lower_factor, stat=stat, npts=npts,
    )
    ls_lo, ls_hi, ls_init = float(info["ls_lo"]), float(info["ls_hi"]), float(info["ls_init"])
    return (
        skgp.kernels.ConstantKernel(config.kernel_constant_init, config.kernel_constant_bounds)
        * skgp.kernels.RBF(length_scale=ls_init, length_scale_bounds=(ls_lo, ls_hi))
    )


def _extract_rbf_bounds_and_scale(kernel) -> Tuple[float, float, float]:
    """Return (ls_lo, ls_hi, ls_init) from a Constant*RBF kernel (best-effort)."""
    try:
        rbf = kernel.k2 if hasattr(kernel, "k2") else kernel
        b = getattr(rbf, "length_scale_bounds", None)
        ls = getattr(rbf, "length_scale", None)
        if b is None or ls is None:
            return float("nan"), float("nan"), float("nan")
        return float(b[0]), float(b[1]), float(np.atleast_1d(ls)[0])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _effective_gp_range(
    ds: "DatasetConfig", lo_e: float, hi_e: float
) -> Tuple[float, float]:
    """Apply optional data_low/data_high limits, keeping the scan range covered."""
    lo = lo_e if getattr(ds, "data_low", None) is None else float(ds.data_low)
    hi = hi_e if getattr(ds, "data_high", None) is None else float(ds.data_high)
    lo = max(lo_e, lo)
    hi = min(hi_e, hi)
    lo = min(lo, float(ds.m_low))
    hi = max(hi, float(ds.m_high))
    return lo, hi


# ---------------------------------------------------------------------------
# GP fitting and prediction
# ---------------------------------------------------------------------------

def fit_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: "Config",
    restarts: int = None,
    kernel=None,
    *,
    optimize: bool = True,
) -> GaussianProcessRegressor:
    """Fit a Gaussian Process Regressor.

    Args:
        X_train: Training mass values
        y_train: Training count values
        config: Global configuration
        restarts: Number of optimizer restarts (defaults to config.n_restarts)
        kernel: Kernel to use; if None, config.get_kernel() is used
        optimize: If False, freeze hyperparameters (no optimizer)

    Returns:
        Fitted GaussianProcessRegressor
    """
    if restarts is None:
        restarts = config.n_restarts

    X_in, y_in, alpha = preprocess_xy_for_gpr(X_train, y_train, config)

    ker = config.get_kernel() if kernel is None else kernel
    try:
        ker = sk_clone(ker)
    except Exception:
        pass

    n_restarts = int(restarts) if optimize else 0
    opt = "fmin_l_bfgs_b" if optimize else None

    try:
        gpr = GaussianProcessRegressor(
            kernel=ker,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            optimizer=opt,
            normalize_y=False,
        )
    except TypeError:
        # Older sklearn without optimizer kwarg
        gpr = GaussianProcessRegressor(
            kernel=ker,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
        )

    return gpr.fit(X_in.reshape(-1, 1), y_in)


def predict_counts_from_log_gpr(
    gpr: GaussianProcessRegressor,
    X_query: np.ndarray,
    config: "Config",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict counts from a fitted GPR model (full covariance matrix).

    Applies lognormal moment transform if config.pre_log is enabled.

    Returns:
        Tuple of (mean counts, covariance matrix)
    """
    Xq = np.asarray(X_query, dtype=float)
    Xq_in = np.log(np.clip(Xq, 1e-12, None)) if config.pre_log else Xq.copy()

    mu, cov = gpr.predict(Xq_in.reshape(-1, 1), return_cov=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if not config.pre_log:
        return mu, cov

    # Lognormal moment transform
    diag = np.clip(np.diag(cov), 0.0, None)
    E = np.exp(mu + 0.5 * diag)
    EyEj = np.outer(E, E)
    cov_y = EyEj * (np.exp(np.clip(cov, -40.0, 40.0)) - 1.0)

    return E, cov_y


def predict_counts_mean_var_from_log_gpr(
    gpr: GaussianProcessRegressor,
    X_query: np.ndarray,
    config: "Config",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast diagonal-only prediction: returns (mean, variance) in count-space.

    Uses gpr.predict(return_std=True) to avoid constructing the full NxN covariance.
    Applies lognormal moment transform if config.pre_log is enabled.
    """
    Xq = np.asarray(X_query, dtype=float).reshape(-1)
    Xq_in = np.log(np.clip(Xq, 1e-12, None)) if config.pre_log else Xq.copy()

    mu, std = gpr.predict(Xq_in.reshape(-1, 1), return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)

    if not config.pre_log:
        return mu, np.clip(std ** 2, 0.0, None)

    var_z = np.clip(std ** 2, 0.0, 40.0)
    mean_y = np.exp(mu + 0.5 * var_z)
    var_y = (np.exp(var_z) - 1.0) * np.exp(2.0 * mu + var_z)
    return np.asarray(mean_y, float), np.asarray(var_y, float)
