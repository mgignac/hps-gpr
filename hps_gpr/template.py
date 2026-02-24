"""Signal template and CLs calculations."""

import math
from math import erf, sqrt
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def gaussian_bin_integrals(
    edges: np.ndarray, m0: float, sigma: float
) -> np.ndarray:
    """Compute Gaussian CDF integrals over bins.

    Args:
        edges: Bin edges
        m0: Gaussian mean
        sigma: Gaussian width

    Returns:
        Array of integrals for each bin
    """
    e = np.asarray(edges, dtype=float)
    z = (e - m0) / (sqrt(2.0) * float(sigma))
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z))
    integ = np.diff(cdf)
    return np.clip(integ, 0.0, None)


def normalize_template(w: np.ndarray) -> np.ndarray:
    """Normalize a template to sum to 1."""
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(w, 1.0 / max(1, w.size))
    return w / s


def build_template(
    edges: np.ndarray, mass: float, sigma_val: float
) -> np.ndarray:
    """Build a normalized signal template.

    Args:
        edges: Bin edges
        mass: Signal mass hypothesis
        sigma_val: Mass resolution

    Returns:
        Normalized template array
    """
    return normalize_template(gaussian_bin_integrals(edges, mass, sigma_val))


def _safe_mvn_draw(
    mean: np.ndarray,
    cov: Optional[np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Safely draw from multivariate normal, with fallbacks."""
    m = np.asarray(mean, dtype=float)

    if cov is None:
        draws = np.tile(m, (size, 1))
    else:
        C = np.asarray(cov, dtype=float)
        try:
            draws = rng.multivariate_normal(
                m, C, size=size, check_valid="ignore", tol=1e-8
            )
        except Exception:
            diag = np.clip(np.diag(C), 0.0, None)
            draws = rng.normal(loc=m, scale=np.sqrt(diag), size=(size, m.size))

    return np.clip(draws, 0.0, None)


def _log_lr(
    n: np.ndarray, b: np.ndarray, s: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Compute log likelihood ratio."""
    n = np.asarray(n, dtype=float)
    b = np.asarray(b, dtype=float)
    s = np.asarray(s, dtype=float)

    if b.ndim == 1 and n.ndim > 1:
        b = np.broadcast_to(b, n.shape)

    b_eff = np.clip(b, eps, None)
    sb_eff = np.clip(b + s, eps, None)
    term = -s + n * (np.log(sb_eff) - np.log(b_eff))
    return np.sum(term, axis=-1)


def cls_amplitude_asymptotic(
    A: float,
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    cov: Optional[np.ndarray],
    template: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute CLs using asymptotic approximation.

    Args:
        A: Signal amplitude
        n_obs: Observed counts
        b_mean: Background mean prediction
        cov: Background covariance matrix
        template: Normalized signal template
        eps: Small value to prevent division by zero

    Returns:
        Tuple of (CLs, CL_sb, CL_b)
    """
    A = float(max(A, 0.0))
    b = np.asarray(b_mean, dtype=float)
    s = A * np.asarray(template, dtype=float)
    n = np.asarray(n_obs, dtype=float)

    b_eff = np.clip(b, eps, None)
    c = np.log1p(s / b_eff)
    S = float(np.sum(s))
    lnQ_obs = float(-S + np.dot(c, n))

    cov_term = 0.0
    if cov is not None:
        try:
            cov_term = float(c @ np.asarray(cov, dtype=float) @ c)
        except Exception:
            cov_term = 0.0

    mu_b = float(-S + np.dot(c, b))
    var_b = float(np.dot(c * c, b) + cov_term)
    sd_b = math.sqrt(max(var_b, eps))

    sb = b + s
    mu_sb = float(-S + np.dot(c, sb))
    var_sb = float(np.dot(c * c, sb) + cov_term)
    sd_sb = math.sqrt(max(var_sb, eps))

    def Phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    z_b = (lnQ_obs - mu_b) / sd_b
    z_sb = (lnQ_obs - mu_sb) / sd_sb

    CL_b = float(Phi(z_b))
    CL_sb = float(Phi(z_sb))
    CL_b = max(CL_b, 1e-9)

    return CL_sb / CL_b, CL_sb, CL_b


def cls_amplitude_toys(
    A: float,
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    cov: Optional[np.ndarray],
    template: np.ndarray,
    rng: np.random.Generator,
    num_toys: int,
    floor: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute CLs using toy Monte Carlo.

    Args:
        A: Signal amplitude
        n_obs: Observed counts
        b_mean: Background mean prediction
        cov: Background covariance matrix
        template: Normalized signal template
        rng: Random number generator
        num_toys: Number of toys to generate
        floor: Small value to prevent division by zero

    Returns:
        Tuple of (CLs, CL_sb, CL_b)
    """
    A = float(max(A, 0.0))
    s = A * template
    lnQ_obs = float(_log_lr(n_obs, b_mean, s, eps=floor))

    b_toys = _safe_mvn_draw(b_mean, cov, size=num_toys, rng=rng)

    n_b = rng.poisson(b_toys)
    lnQ_b = _log_lr(n_b, b_toys, s, eps=floor)

    sb_means = b_toys + s
    n_sb = rng.poisson(sb_means)
    lnQ_sb = _log_lr(n_sb, b_toys, s, eps=floor)

    CL_b = float(np.mean(lnQ_b <= lnQ_obs))
    CL_sb = float(np.mean(lnQ_sb <= lnQ_obs))
    CL_b = max(CL_b, 1e-9)

    return CL_sb / CL_b, CL_sb, CL_b


def cls_limit_for_amplitude(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: Optional[np.ndarray],
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    config: "Config",
    seed: int = 1,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Find the CLs upper limit on signal amplitude.

    Uses bisection to find the amplitude A such that CLs = alpha.

    Args:
        n_obs: Observed counts
        b_mean: Background mean prediction
        b_cov: Background covariance matrix
        edges: Bin edges
        mass: Signal mass hypothesis
        sigma_val: Mass resolution
        config: Global configuration
        seed: Random seed

    Returns:
        Tuple of (A_upper_limit, debug_dict)
    """
    template = build_template(edges, mass, sigma_val)
    rng = np.random.default_rng(seed)
    alpha = config.cls_alpha
    mode = config.cls_mode
    num_toys = config.cls_num_toys

    def cls_at(A):
        if mode == "asymptotic":
            return cls_amplitude_asymptotic(A, n_obs, b_mean, b_cov, template)[0]
        return cls_amplitude_toys(
            A, n_obs, b_mean, b_cov, template, rng, max(1, int(num_toys))
        )[0]

    b_sum = float(np.sum(b_mean))
    A_lo = 0.0
    A_hi = max(1.0, 3.0 * math.sqrt(max(b_sum, 1.0)))

    cls_hi = cls_at(A_hi)
    it = 0
    while cls_hi > alpha and A_hi < 1e7 and it < 40:
        A_hi *= 2.0
        cls_hi = cls_at(A_hi)
        it += 1

    gridA = [A_lo, A_hi]
    gridC = [cls_at(A_lo), cls_hi]

    for _ in range(40):
        Amid = 0.5 * (A_lo + A_hi)
        cls_mid = cls_at(Amid)
        gridA.append(Amid)
        gridC.append(cls_mid)

        if abs(cls_mid - alpha) < 1e-2:
            A_lo = A_hi = Amid
            break
        if cls_mid > alpha:
            A_lo = Amid
        else:
            A_hi = Amid
        if abs(A_hi - A_lo) <= max(1e-3, 1e-3 * (1.0 + A_hi)):
            break

    return 0.5 * (A_lo + A_hi), {
        "A_grid": np.array(gridA),
        "CLs_grid": np.array(gridC),
    }


def cls_limit_for_template(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: Optional[np.ndarray],
    template: np.ndarray,
    *,
    ds: Optional["DatasetConfig"] = None,
    mass: Optional[float] = None,
    integral_density: Optional[float] = None,
    alpha: float = 0.05,
    mode: str = "asymptotic",
    use_eps2: bool = False,
    num_toys: int = 100,
    seed: int = 1,
    A_hi0: Optional[float] = None,
) -> Tuple[float, float]:
    """CLs upper limit for a pre-built signal template.

    More flexible than cls_limit_for_amplitude: accepts a pre-built template
    and can optionally convert the amplitude limit to epsilon^2.

    Args:
        n_obs: Observed counts
        b_mean: Background mean prediction
        b_cov: Background covariance matrix
        template: Pre-built normalized signal template
        ds: Dataset config (required when use_eps2=True)
        mass: Signal mass in GeV (required when use_eps2=True)
        integral_density: Counts per GeV (required when use_eps2=True)
        alpha: CL level (default 0.05 → 95% UL)
        mode: "asymptotic" or "toys"
        use_eps2: If True, convert A_up to eps2_up and return (eps2_up, A_up)
        num_toys: Number of CLs toys (only used when mode="toys")
        seed: Random seed
        A_hi0: Initial upper bracket for bisection (auto-set if None)

    Returns:
        (limit, A_up) where limit = eps2_up if use_eps2=True else A_up
    """
    template = np.asarray(template, float)
    rng = np.random.default_rng(int(seed))

    def cls_at(A: float) -> float:
        if mode == "asymptotic":
            return cls_amplitude_asymptotic(float(A), n_obs, b_mean, b_cov, template)[0]
        return cls_amplitude_toys(
            float(A), n_obs, b_mean, b_cov, template, rng, max(1, int(num_toys))
        )[0]

    b_sum = float(np.sum(b_mean))
    A_lo = 0.0
    if A_hi0 is None:
        A_hi = max(1.0, 3.0 * math.sqrt(max(b_sum, 1.0)))
    else:
        A_hi = float(A_hi0)

    cls_hi = cls_at(A_hi)
    it = 0
    while cls_hi > alpha and A_hi < 1e7 and it < 40:
        A_hi *= 2.0
        cls_hi = cls_at(A_hi)
        it += 1

    for _ in range(60):
        Amid = 0.5 * (A_lo + A_hi)
        cls_mid = cls_at(Amid)
        if abs(cls_mid - alpha) < 1e-3:
            A_lo = A_hi = Amid
            break
        if cls_mid > alpha:
            A_lo = Amid
        else:
            A_hi = Amid
        if abs(A_hi - A_lo) <= max(1e-3, 1e-3 * (1.0 + A_hi)):
            break

    A_up = float(0.5 * (A_lo + A_hi))

    if not bool(use_eps2):
        return A_up, A_up

    if ds is None or mass is None or integral_density is None:
        raise ValueError(
            "cls_limit_for_template(use_eps2=True) requires ds, mass, and integral_density."
        )

    from .conversion import epsilon2_from_A

    eps2_up = float(epsilon2_from_A(ds, float(mass), A_up, float(integral_density)))
    return eps2_up, A_up
