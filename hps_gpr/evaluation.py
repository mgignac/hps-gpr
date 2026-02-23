"""Evaluation functions for single and combined dataset analyses."""

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
from .statistics import (
    p0_from_blind_vectors,
    p0_lognormal_poisson,
    fit_A_profiled_gaussian,
    fit_A_profiled_gaussian_details,
    p0_profiled_gaussian_LRT,
)
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


# ---------------------------------------------------------------------------
# Visibility and edge-guard helpers
# ---------------------------------------------------------------------------

def _stable_seed_from_tag(tag: str, *, base: int = 0) -> int:
    """Deterministic 32-bit seed from a string tag (stable across sessions/processes)."""
    h = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
    return int((base + int(h, 16)) % (2 ** 31 - 1))


def _dataset_visibility(ds: "DatasetConfig", config: "Config") -> str:
    """Return the visibility policy ('observed' or 'expected_only') for a dataset."""
    vis_map = dict(getattr(config, "data_visibility", {}) or {})
    vis = str(vis_map.get(ds.key, "observed")).lower().strip()
    return vis if vis in ("observed", "expected_only") else "observed"


def _all_observed(ds_list: List["DatasetConfig"], config: "Config") -> bool:
    """Return True only if every dataset in the list is in 'observed' mode."""
    return all(_dataset_visibility(ds, config) == "observed" for ds in ds_list)


def active_datasets_for_mass(
    mass: float,
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
) -> List["DatasetConfig"]:
    """Return datasets active at this mass point, applying optional edge guards.

    If config.scan_require_two_sidebands=True, require the full training exclusion
    window (±guard_nsigma·σ(m)) to lie within the dataset's data range.
    This suppresses unstable fits near dataset boundaries.
    """
    out: List["DatasetConfig"] = []
    m = float(mass)
    require_two = bool(getattr(config, "scan_require_two_sidebands", False))
    guard_nsig = float(
        getattr(config, "scan_edge_guard_nsigma", None)
        or getattr(config, "gp_train_exclude_nsigma", None)
        or config.blind_nsigma
    )

    for ds in datasets.values():
        if not bool(getattr(ds, "enabled", True)):
            continue
        if m < float(ds.m_low) or m > float(ds.m_high):
            continue
        if require_two:
            try:
                s = float(ds.sigma(m))
                lo = float(getattr(ds, "data_low", None) or ds.m_low)
                hi = float(getattr(ds, "data_high", None) or ds.m_high)
                if (m - guard_nsig * s) <= lo:
                    continue
                if (m + guard_nsig * s) >= hi:
                    continue
            except Exception:
                pass  # fail-open on guard errors
        out.append(ds)

    return out


# ---------------------------------------------------------------------------
# Single dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_single_dataset(
    ds: "DatasetConfig",
    mass: float,
    config: "Config",
    do_extraction: bool = True,
    *,
    compute_observed: Optional[bool] = None,
    return_fit_details: bool = False,
) -> Tuple[SingleResult, BlindPrediction, Optional[Dict]]:
    """Evaluate a single dataset at one mass hypothesis.

    If data_visibility for this dataset is 'expected_only', observed quantities
    (UL, p0, extraction) are suppressed and returned as NaN.

    Returns:
        Tuple of (SingleResult, BlindPrediction, fit_details or None)
    """
    pred = estimate_background_for_dataset(ds, mass, config)

    vis = _dataset_visibility(ds, config)
    if compute_observed is None:
        compute_observed = (vis == "observed")

    if not compute_observed:
        return (
            SingleResult(ds.key, float(mass),
                         float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), False),
            pred,
            None,
        )

    # --- CLs upper limit on A ---
    seed = None
    if config.cls_mode == "toys":
        seed = _stable_seed_from_tag(
            f"CLS:{ds.key}:{float(mass):.6f}", base=config.cls_seed_base
        )

    A_up, _ = cls_limit_for_amplitude(
        n_obs=pred.obs,
        b_mean=pred.mu,
        b_cov=pred.cov,
        edges=pred.edges,
        mass=mass,
        sigma_val=pred.sigma_val,
        config=config,
        seed=seed,
    )
    eps2_up = epsilon2_from_A(ds, mass, A_up, pred.integral_density)

    # --- p0/Z via profiled LRT (v15) ---
    tmpl = build_template(pred.edges, mass, pred.sigma_val)
    p0, Z, _, _ = p0_profiled_gaussian_LRT(pred.obs, pred.mu, pred.cov, tmpl)

    # --- Signal extraction ---
    fit_details = None
    if do_extraction:
        allow_neg = bool(getattr(config, "extract_allow_negative", True))
        if return_fit_details:
            fit_details = fit_A_profiled_gaussian_details(
                pred.obs, pred.mu, pred.cov, tmpl, allow_negative=allow_neg
            )
            A_hat = float(fit_details.get("A_hat", float("nan")))
            sigma_A = float(fit_details.get("sigma_A", float("nan")))
            ok = bool(fit_details.get("success", False))
        else:
            fit = fit_A_profiled_gaussian(
                pred.obs, pred.mu, pred.cov, tmpl, allow_negative=allow_neg
            )
            A_hat = float(fit["A_hat"])
            sigma_A = float(fit["sigma_A"])
            ok = bool(fit["success"])
    else:
        A_hat, sigma_A, ok = float("nan"), float("nan"), False

    return (
        SingleResult(
            ds.key, float(mass), float(A_up), float(eps2_up),
            float(p0), float(Z), float(A_hat), float(sigma_A), bool(ok),
        ),
        pred,
        fit_details if return_fit_details else None,
    )


# ---------------------------------------------------------------------------
# Combined dataset evaluation
# ---------------------------------------------------------------------------

def _concat_block_diag(covs: List[np.ndarray]) -> np.ndarray:
    """Concatenate covariance matrices into block diagonal."""
    sizes = [c.shape[0] for c in covs]
    out = np.zeros((sum(sizes), sum(sizes)), float)
    i = 0
    for C in covs:
        n = C.shape[0]
        out[i: i + n, i: i + n] = C
        i += n
    return out


def build_combined_components(
    mass: float,
    ds_list: List["DatasetConfig"],
    preds: List[BlindPrediction],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build concatenated (obs, b, cov, s_unit) for a shared mass hypothesis.

    s_unit is the expected counts per unit ε² in each concatenated bin.
    """
    obs = np.concatenate([p.obs for p in preds]).astype(int)
    b = np.concatenate([p.mu for p in preds]).astype(float)
    cov = _concat_block_diag([p.cov for p in preds]).astype(float)
    templates = [build_template(p.edges, mass, p.sigma_val) for p in preds]
    Ks = [A_from_epsilon2(ds, mass, 1.0, p.integral_density) for ds, p in zip(ds_list, preds)]
    s_unit = np.concatenate([K * t for K, t in zip(Ks, templates)]).astype(float)
    return obs, b, cov, s_unit


def combined_cls_limit_epsilon2(
    mass: float,
    ds_list: List["DatasetConfig"],
    preds: List[BlindPrediction],
    config: "Config",
    seed: int = 1,
) -> float:
    """Compute combined CLs limit on epsilon^2."""
    obs, b, cov, s_unit = build_combined_components(float(mass), ds_list, preds)
    rng = np.random.default_rng(seed)
    alpha = config.cls_alpha
    mode = config.cls_mode
    num_toys = config.cls_num_toys

    def cls_at_eps2(eps2: float) -> float:
        eps2 = float(max(eps2, 0.0))
        if mode == "asymptotic":
            return cls_amplitude_asymptotic(1.0, obs, b, cov, eps2 * s_unit)[0]
        s = eps2 * s_unit
        lnQ_obs = float(_log_lr(obs, b, s))
        b_toys = _safe_mvn_draw(b, cov, size=int(num_toys), rng=rng)
        n_b = rng.poisson(b_toys)
        lnQ_b = _log_lr(n_b, b_toys, s)
        sb_means = b_toys + s
        n_sb = rng.poisson(sb_means)
        lnQ_sb = _log_lr(n_sb, b_toys, s)
        CL_b = max(float(np.mean(lnQ_b <= lnQ_obs)), 1e-9)
        return float(np.mean(lnQ_sb <= lnQ_obs)) / CL_b

    eps_lo, eps_hi = 0.0, 1e-10
    it = 0
    while cls_at_eps2(eps_hi) > alpha and eps_hi < 1.0 and it < 80:
        eps_hi *= 2.0
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
    """Evaluate combined datasets at one mass hypothesis."""
    obs, b, cov, s_unit = build_combined_components(float(mass), ds_list, preds)
    eps2_up = combined_cls_limit_epsilon2(mass, ds_list, preds, config)

    # Profiled LRT p0 on ε² scale
    eps2_scale = float(getattr(config, "eps2_lrt_scale", 1e10))
    p0, Z, _, _ = p0_profiled_gaussian_LRT(obs, b, cov, s_unit / eps2_scale)

    return CombinedResult(float(mass), float(eps2_up), float(p0), float(Z))
