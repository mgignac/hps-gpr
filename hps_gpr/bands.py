"""Expected upper limit bands calculation."""

from types import SimpleNamespace
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import joblib
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    import contextlib
    _threadpool_limits = contextlib.nullcontext  # type: ignore[assignment]

from .io import estimate_background_for_dataset
from .template import build_template, cls_limit_for_template
from .statistics import draw_bkg_mvn_nonneg, p0_profiled_gaussian_LRT
from .gpr import make_kernel_for_dataset, fit_gpr, predict_counts_from_log_gpr
from .evaluation import (
    build_combined_components,
    combined_cls_limit_epsilon2_from_vectors,
    active_datasets_for_mass,
    _dataset_visibility,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


# ---------------------------------------------------------------------------
# Single-dataset expected UL bands
# ---------------------------------------------------------------------------

def expected_ul_bands_for_dataset(
    ds: "DatasetConfig",
    masses: List[float],
    config: "Config",
    *,
    n_toys: Optional[int] = None,
    seed: Optional[int] = None,
    use_eps2: bool = True,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    train_exclude_nsigma: Optional[float] = None,
) -> pd.DataFrame:
    """Compute expected upper limit bands for a dataset vs mass.

    Toy modes
    ---------
    (A) Conditional-GP toys (fast, default):
        Draw background from the fixed GP posterior MVN (μ, Σ) from real-data sidebands.
        Poisson-sample blind-window counts and compute UL with the same (μ, Σ).

    (B) Full procedural toys (slow; refit_gp_on_toy=True):
        Generate a full pseudo-dataset, refit GP on toy sidebands (excluding
        ±train_exclude_nsigma·σ), predict toy-specific (μ, Σ), compute UL.

    Args:
        ds: Dataset configuration
        masses: Mass values to evaluate (GeV)
        config: Global configuration
        n_toys: Number of toys per mass (defaults to config.ul_bands_toys)
        seed: Random seed (defaults to config.ul_bands_seed)
        use_eps2: Compute and return epsilon^2 bands in addition to amplitude bands
        refit_gp_on_toy: Override config.ul_bands_refit_gp_on_toy
        refit_restarts: Override config.ul_bands_refit_gp_restarts
        refit_optimize: Override config.ul_bands_refit_gp_optimize
        train_exclude_nsigma: Override config.ul_bands_train_exclude_nsigma

    Returns:
        DataFrame with per-mass band columns (A_med, eps2_med, etc.)
    """
    masses = [float(m) for m in masses]

    # Resolve parameters from config
    if n_toys is None:
        n_toys = int(config.ul_bands_toys)
    if seed is None:
        seed = int(getattr(config, "ul_bands_seed", 12345))
    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "ul_bands_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "ul_bands_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "ul_bands_refit_gp_optimize", True))
    if train_exclude_nsigma is None:
        train_exclude_nsigma = float(
            getattr(config, "ul_bands_train_exclude_nsigma", None)
            or getattr(config, "gp_train_exclude_nsigma", None)
            or config.blind_nsigma
        )

    n_workers = int(getattr(config, "ul_bands_n_workers", 1))
    backend = str(getattr(config, "ul_bands_parallel_backend", "loky"))
    threads_per_worker = int(getattr(config, "ul_bands_threads_per_worker", 1))
    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))

    compute_obs = (_dataset_visibility(ds, config) == "observed")
    child_ss = np.random.SeedSequence(int(seed)).spawn(len(masses))

    def _quantiles(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        return tuple(float(x) for x in np.quantile(a, [0.025, 0.16, 0.50, 0.84, 0.975]))

    def _nanmean(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else np.nan

    def _one_mass(i: int, m: float) -> dict:
        rng = np.random.default_rng(child_ss[i])

        with _threadpool_limits(limits=int(threads_per_worker)):
            pred = estimate_background_for_dataset(
                ds, float(m), config,
                train_exclude_nsigma=float(train_exclude_nsigma),
            )

        tmpl = build_template(pred.edges, m, pred.sigma_val)
        obs = np.asarray(pred.obs, int)
        mu = np.asarray(pred.mu, float)
        cov = np.asarray(pred.cov, float)

        alpha = float(config.cls_alpha)
        mode = str(config.cls_mode)
        num_toys_cls = int(config.cls_num_toys)

        # Observed UL
        if compute_obs:
            if bool(use_eps2):
                eps2_obs, A_obs = cls_limit_for_template(
                    obs, mu, cov, tmpl,
                    alpha=alpha, mode=mode, use_eps2=True,
                    ds=ds, mass=float(m), integral_density=float(pred.integral_density),
                    num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                )
            else:
                A_obs, _ = cls_limit_for_template(
                    obs, mu, cov, tmpl,
                    alpha=alpha, mode=mode, use_eps2=False,
                    num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                )
                eps2_obs = np.nan
        else:
            eps2_obs, A_obs = np.nan, np.nan

        toy_A: List[float] = []
        toy_eps2: List[float] = []

        if not refit_gp_on_toy:
            # --- Mode (A): conditional-GP toys ---
            lam_draws = draw_bkg_mvn_nonneg(
                mu, cov, int(n_toys), rng,
                method=mvn_method, max_tries=mvn_max_tries,
            )
            n_draws = rng.poisson(lam_draws).astype(int)

            for n_t in n_draws:
                if bool(use_eps2):
                    eps2_t, A_t = cls_limit_for_template(
                        n_t, mu, cov, tmpl,
                        alpha=alpha, mode=mode, use_eps2=True,
                        ds=ds, mass=float(m), integral_density=float(pred.integral_density),
                        num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_eps2.append(float(eps2_t))
                    toy_A.append(float(A_t))
                else:
                    A_t, _ = cls_limit_for_template(
                        n_t, mu, cov, tmpl,
                        alpha=alpha, mode=mode, use_eps2=False,
                        num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_A.append(float(A_t))

        else:
            # --- Mode (B): full procedural toys with GP refit ---
            x_full = np.asarray(pred.x_full, float).reshape(-1)
            mu_full = np.asarray(pred.mu_full, float).reshape(-1)
            blind = tuple(pred.blind)
            msk_blind = (x_full >= blind[0]) & (x_full <= blind[1])
            if int(np.sum(msk_blind)) <= 0:
                raise RuntimeError(f"[bands][{ds.key}] m={m:.6g}: blind window has no bins")
            x_win = x_full[msk_blind]
            blind_train = (
                float(m) - float(train_exclude_nsigma) * float(pred.sigma_val),
                float(m) + float(train_exclude_nsigma) * float(pred.sigma_val),
            )
            msk_train = (x_full < blind_train[0]) | (x_full > blind_train[1])
            ker = make_kernel_for_dataset(ds, config, mass=m)

            for _ in range(int(n_toys)):
                y_full_toy = rng.poisson(np.clip(mu_full, 0.0, None)).astype(int)
                obs_toy = y_full_toy[msk_blind].astype(int)
                mu_fit, cov_fit = mu, cov
                try:
                    X_tr = x_full[msk_train]
                    y_tr = y_full_toy[msk_train].astype(float)
                    gpr = fit_gpr(
                        X_tr, y_tr, config,
                        restarts=int(refit_restarts),
                        kernel=ker,
                        optimize=bool(refit_optimize),
                    )
                    mu_fit, cov_fit = predict_counts_from_log_gpr(gpr, x_win, config)
                except Exception:
                    pass

                if bool(use_eps2):
                    eps2_t, A_t = cls_limit_for_template(
                        obs_toy,
                        np.asarray(mu_fit, float), np.asarray(cov_fit, float),
                        tmpl,
                        alpha=alpha, mode=mode, use_eps2=True,
                        ds=ds, mass=float(m), integral_density=float(pred.integral_density),
                        num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_eps2.append(float(eps2_t))
                    toy_A.append(float(A_t))
                else:
                    A_t, _ = cls_limit_for_template(
                        obs_toy,
                        np.asarray(mu_fit, float), np.asarray(cov_fit, float),
                        tmpl,
                        alpha=alpha, mode=mode, use_eps2=False,
                        num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_A.append(float(A_t))

        toy_A = np.asarray(toy_A, float)
        toy_eps2 = np.asarray(toy_eps2, float)

        qA = _quantiles(toy_A)
        mA = _nanmean(toy_A)
        if bool(use_eps2):
            qE = _quantiles(toy_eps2)
            mE = _nanmean(toy_eps2)
        else:
            qE = (np.nan, np.nan, np.nan, np.nan, np.nan)
            mE = np.nan

        # p-values comparing obs UL to toy UL distribution (eps2 scale)
        if compute_obs and bool(use_eps2) and np.isfinite(eps2_obs):
            te = toy_eps2[np.isfinite(toy_eps2)]
            if te.size > 0:
                p_strong = float(np.mean(te <= float(eps2_obs)))
                p_weak = float(np.mean(te >= float(eps2_obs)))
                p_two = float(2.0 * min(p_strong, p_weak))
            else:
                p_strong = p_weak = p_two = np.nan
        else:
            p_strong = p_weak = p_two = np.nan

        # Analytic p0 / Z on observed data (only when unblinded)
        if compute_obs:
            try:
                p0_obs = float(p0_profiled_gaussian_LRT(obs, mu, cov, tmpl))
                from scipy.stats import norm as _norm
                Z_obs = float(_norm.ppf(1.0 - p0_obs)) if np.isfinite(p0_obs) and p0_obs < 1.0 else np.nan
            except Exception:
                p0_obs = np.nan
                Z_obs = np.nan
        else:
            p0_obs = np.nan
            Z_obs = np.nan

        return dict(
            dataset=ds.key,
            mass_GeV=float(m),
            # Publication-facing columns
            eps2_obs=float(eps2_obs) if np.isfinite(eps2_obs) else np.nan,
            A_obs=float(A_obs) if np.isfinite(A_obs) else np.nan,
            p0_analytic=float(p0_obs), Z_analytic=float(Z_obs),
            eps2_lo2=float(qE[0]), eps2_lo1=float(qE[1]), eps2_med=float(qE[2]),
            eps2_hi1=float(qE[3]), eps2_hi2=float(qE[4]), eps2_mean=float(mE),
            A_lo2=float(qA[0]), A_lo1=float(qA[1]), A_med=float(qA[2]),
            A_hi1=float(qA[3]), A_hi2=float(qA[4]), A_mean=float(mA),
            # Backward-compatible aliases
            ul_eps2_obs=float(eps2_obs) if np.isfinite(eps2_obs) else np.nan,
            ul_A_obs=float(A_obs) if np.isfinite(A_obs) else np.nan,
            toy_eps2_uls_q02=float(qE[0]), toy_eps2_uls_q16=float(qE[1]),
            toy_eps2_uls_q50=float(qE[2]), toy_eps2_uls_q84=float(qE[3]),
            toy_eps2_uls_q97=float(qE[4]), toy_eps2_uls_mean=float(mE),
            toy_A_uls_q02=float(qA[0]), toy_A_uls_q16=float(qA[1]),
            toy_A_uls_q50=float(qA[2]), toy_A_uls_q84=float(qA[3]),
            toy_A_uls_q97=float(qA[4]), toy_A_uls_mean=float(mA),
            p_strong=float(p_strong), p_weak=float(p_weak), p_two=float(p_two),
            # Provenance
            bands_refit_gp_on_toy=bool(refit_gp_on_toy),
            bands_train_exclude_nsigma=float(train_exclude_nsigma),
            bands_refit_restarts=int(refit_restarts),
            bands_refit_optimize=bool(refit_optimize),
        )

    if n_workers <= 1 or not _HAVE_JOBLIB:
        rows = [_one_mass(i, m) for i, m in enumerate(masses)]
    else:
        rows = joblib.Parallel(n_jobs=int(n_workers), backend=str(backend))(
            joblib.delayed(_one_mass)(i, m) for i, m in enumerate(masses)
        )

    return pd.DataFrame(rows).sort_values("mass_GeV").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined expected UL bands
# ---------------------------------------------------------------------------

def expected_ul_bands_for_combination(
    ds_keys: List[str],
    datasets: Dict[str, "DatasetConfig"],
    masses: List[float],
    config: "Config",
    *,
    n_toys: Optional[int] = None,
    seed: Optional[int] = None,
    only_overlap: bool = False,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    train_exclude_nsigma: Optional[float] = None,
) -> pd.DataFrame:
    """Compute expected epsilon^2 UL bands for a dataset combination vs mass.

    Args:
        ds_keys: Dataset keys to combine
        datasets: Full datasets dictionary (from make_datasets)
        masses: Mass values to evaluate (GeV)
        config: Global configuration
        n_toys: Number of toys per mass (defaults to config.combined_bands_n_toys)
        seed: Random seed (defaults to config.combined_bands_seed)
        only_overlap: If True, restrict to masses where >=2 datasets are simultaneously active
        refit_gp_on_toy: Override config.ul_bands_refit_gp_on_toy
        refit_restarts: Override config.ul_bands_refit_gp_restarts
        refit_optimize: Override config.ul_bands_refit_gp_optimize
        train_exclude_nsigma: Override config.ul_bands_train_exclude_nsigma

    Returns:
        DataFrame with per-mass combined epsilon^2 band columns
    """
    masses = [float(m) for m in masses]
    # Accept either string keys or DatasetConfig objects
    ds_keys = [getattr(d, "key", str(d)) for d in ds_keys]

    if n_toys is None:
        n_toys = int(getattr(config, "combined_bands_n_toys", config.ul_bands_toys))
    if seed is None:
        seed = int(getattr(config, "combined_bands_seed", 24680))
    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "ul_bands_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "ul_bands_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "ul_bands_refit_gp_optimize", True))
    if train_exclude_nsigma is None:
        train_exclude_nsigma = float(
            getattr(config, "ul_bands_train_exclude_nsigma", None)
            or getattr(config, "gp_train_exclude_nsigma", None)
            or config.blind_nsigma
        )

    n_workers = int(getattr(config, "ul_bands_n_workers", 1))
    backend = str(getattr(config, "ul_bands_parallel_backend", "loky"))
    threads_per_worker = int(getattr(config, "ul_bands_threads_per_worker", 1))
    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))

    def _active_set(m: float) -> List[str]:
        return [d.key for d in active_datasets_for_mass(float(m), datasets, config)]

    # Filter masses to those with at least one (or two if only_overlap) active datasets
    min_active = 2 if only_overlap else 1
    masses = [m for m in masses if len(set(_active_set(m)).intersection(ds_keys)) >= min_active]

    child_ss = np.random.SeedSequence(int(seed)).spawn(len(masses))

    def _quantiles(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        return tuple(float(x) for x in np.quantile(a, [0.025, 0.16, 0.50, 0.84, 0.975]))

    def _nanmean(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else np.nan

    nan = float("nan")

    def _empty_row(tag: str, m: float) -> dict:
        return dict(
            dataset_set=str(tag), mass_GeV=float(m),
            eps2_obs=nan, p0_analytic=nan, Z_analytic=nan,
            eps2_lo2=nan, eps2_lo1=nan, eps2_med=nan, eps2_hi1=nan, eps2_hi2=nan, eps2_mean=nan,
            ul_eps2_obs=nan,
            toy_eps2_uls_q02=nan, toy_eps2_uls_q16=nan, toy_eps2_uls_q50=nan,
            toy_eps2_uls_q84=nan, toy_eps2_uls_q97=nan, toy_eps2_uls_mean=nan,
            p_strong=nan, p_weak=nan, p_two=nan, meta=str([]),
            bands_refit_gp_on_toy=bool(refit_gp_on_toy),
            bands_train_exclude_nsigma=float(train_exclude_nsigma),
            bands_refit_restarts=int(refit_restarts),
            bands_refit_optimize=bool(refit_optimize),
        )

    def _one_mass(i: int, m: float) -> dict:
        rng = np.random.default_rng(child_ss[i])
        ds_here = [k for k in ds_keys if k in _active_set(m)]
        if len(ds_here) == 0:
            return _empty_row("EMPTY", float(m))

        ds_tag = "+".join(ds_here)

        with _threadpool_limits(limits=int(threads_per_worker)):
            preds = {
                k: estimate_background_for_dataset(
                    datasets[k], float(m), config,
                    train_exclude_nsigma=float(train_exclude_nsigma),
                )
                for k in ds_here
            }

        preds_list = [preds[k] for k in ds_here]
        ds_list = [datasets[k] for k in ds_here]
        meta = [
            dict(key=k, sigma=float(preds[k].sigma_val), dens=float(preds[k].integral_density))
            for k in ds_here
        ]

        obs_vec0, b_vec, cov_mat, s_unit = build_combined_components(float(m), ds_list, preds_list)

        alpha = float(config.cls_alpha)
        mode = str(config.cls_mode)
        num_toys_cls = int(config.cls_num_toys)

        # Blinding: compute obs only if all datasets are "observed"
        compute_obs = all(
            _dataset_visibility(datasets[k], config) == "observed" for k in ds_here
        )

        if compute_obs:
            eps2_obs = combined_cls_limit_epsilon2_from_vectors(
                obs_vec0, b_vec, cov_mat, s_unit, config,
                seed=int(rng.integers(1, 2**31 - 1)),
            )
        else:
            eps2_obs = nan

        toy_eps2: List[float] = []

        if not refit_gp_on_toy:
            # --- Mode (A): conditional MVN toys ---
            lam_draws_list = [
                draw_bkg_mvn_nonneg(
                    preds[k].mu, preds[k].cov, int(n_toys), rng,
                    method=mvn_method, max_tries=mvn_max_tries,
                )
                for k in ds_here
            ]
            n_draws_list = [rng.poisson(lam).astype(int) for lam in lam_draws_list]

            if len(ds_here) == 1:
                # Single-dataset fallback: use single-dataset CLs
                k = ds_here[0]
                tmpl = build_template(preds[k].edges, m, preds[k].sigma_val)
                for t in range(int(n_toys)):
                    eps2_t, _ = cls_limit_for_template(
                        n_draws_list[0][t],
                        np.asarray(preds[k].mu, float),
                        np.asarray(preds[k].cov, float),
                        tmpl,
                        alpha=alpha, mode=mode, use_eps2=True,
                        ds=datasets[k], mass=float(m),
                        integral_density=float(preds[k].integral_density),
                        num_toys=num_toys_cls, seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_eps2.append(float(eps2_t))
            else:
                # Multi-dataset: concatenate toy obs and run combined CLs
                for t in range(int(n_toys)):
                    obs_toy = np.concatenate([n_draws_list[j][t] for j in range(len(ds_here))])
                    eps2_t = combined_cls_limit_epsilon2_from_vectors(
                        obs_toy, b_vec, cov_mat, s_unit, config,
                        seed=int(rng.integers(1, 2**31 - 1)),
                    )
                    toy_eps2.append(float(eps2_t))

        else:
            # --- Mode (B): full procedural toys with per-dataset GP refit ---
            per: dict = {}
            for k in ds_here:
                x_full = np.asarray(preds[k].x_full, float).reshape(-1)
                mu_full = np.asarray(preds[k].mu_full, float).reshape(-1)
                blind = tuple(preds[k].blind)
                msk_blind = (x_full >= blind[0]) & (x_full <= blind[1])
                if int(np.sum(msk_blind)) <= 0:
                    raise RuntimeError(
                        f"[bands][combo][{ds_tag}] m={m:.6g}: blind window empty for {k}"
                    )
                x_win = x_full[msk_blind]
                blind_train = (
                    float(m) - float(train_exclude_nsigma) * float(preds[k].sigma_val),
                    float(m) + float(train_exclude_nsigma) * float(preds[k].sigma_val),
                )
                msk_train = (x_full < blind_train[0]) | (x_full > blind_train[1])
                ker = make_kernel_for_dataset(datasets[k], config, mass=m)
                per[k] = dict(
                    x_full=x_full, mu_full=mu_full,
                    msk_blind=msk_blind, x_win=x_win,
                    msk_train=msk_train, ker=ker,
                )

            for _ in range(int(n_toys)):
                preds_toy_list = []
                for k in ds_here:
                    y_full_toy = rng.poisson(np.clip(per[k]["mu_full"], 0.0, None)).astype(int)
                    obs_toy_k = y_full_toy[per[k]["msk_blind"]].astype(int)
                    mu_fit = np.asarray(preds[k].mu, float)
                    cov_fit = np.asarray(preds[k].cov, float)
                    try:
                        X_tr = per[k]["x_full"][per[k]["msk_train"]]
                        y_tr = y_full_toy[per[k]["msk_train"]].astype(float)
                        gpr = fit_gpr(
                            X_tr, y_tr, config,
                            restarts=int(refit_restarts),
                            kernel=per[k]["ker"],
                            optimize=bool(refit_optimize),
                        )
                        mu_fit, cov_fit = predict_counts_from_log_gpr(
                            gpr, per[k]["x_win"], config
                        )
                    except Exception:
                        pass
                    preds_toy_list.append(SimpleNamespace(
                        obs=np.asarray(obs_toy_k, int),
                        mu=np.asarray(mu_fit, float),
                        cov=np.asarray(cov_fit, float),
                        edges=np.asarray(preds[k].edges, float),
                        sigma_val=float(preds[k].sigma_val),
                        integral_density=float(preds[k].integral_density),
                    ))

                obs_v, b_v, cov_v, s_v = build_combined_components(float(m), ds_list, preds_toy_list)
                eps2_t = combined_cls_limit_epsilon2_from_vectors(
                    obs_v, b_v, cov_v, s_v, config,
                    seed=int(rng.integers(1, 2**31 - 1)),
                )
                toy_eps2.append(float(eps2_t))

        toy_eps2 = np.asarray(toy_eps2, float)
        q02, q16, q50, q84, q97 = _quantiles(toy_eps2)
        mean_e = _nanmean(toy_eps2)

        if compute_obs and np.isfinite(eps2_obs):
            te = toy_eps2[np.isfinite(toy_eps2)]
            if te.size > 0:
                p_strong = float(np.mean(te <= float(eps2_obs)))
                p_weak = float(np.mean(te >= float(eps2_obs)))
                p_two = float(2.0 * min(p_strong, p_weak))
            else:
                p_strong = p_weak = p_two = nan
        else:
            p_strong = p_weak = p_two = nan

        # Analytic p0 / Z for combined observed data (null signal, A=0 template)
        if compute_obs:
            try:
                # Build a unit template at eps2=1 to get a combined signal shape
                from .template import build_template as _bt
                tmpl_c = np.concatenate([
                    _bt(preds[k].edges, float(m), float(preds[k].sigma_val))
                    for k in ds_here
                ])
                p0_obs_c = float(p0_profiled_gaussian_LRT(obs_vec0, b_vec, cov_mat, tmpl_c))
                from scipy.stats import norm as _norm
                Z_obs_c = float(_norm.ppf(1.0 - p0_obs_c)) if np.isfinite(p0_obs_c) and p0_obs_c < 1.0 else nan
            except Exception:
                p0_obs_c = nan
                Z_obs_c = nan
        else:
            p0_obs_c = nan
            Z_obs_c = nan

        return dict(
            dataset_set=str(ds_tag),
            mass_GeV=float(m),
            # Publication-facing columns
            eps2_obs=float(eps2_obs) if np.isfinite(eps2_obs) else nan,
            p0_analytic=float(p0_obs_c), Z_analytic=float(Z_obs_c),
            eps2_lo2=float(q02), eps2_lo1=float(q16), eps2_med=float(q50),
            eps2_hi1=float(q84), eps2_hi2=float(q97), eps2_mean=float(mean_e),
            # Backward-compatible aliases
            ul_eps2_obs=float(eps2_obs) if np.isfinite(eps2_obs) else nan,
            toy_eps2_uls_q02=float(q02), toy_eps2_uls_q16=float(q16),
            toy_eps2_uls_q50=float(q50), toy_eps2_uls_q84=float(q84),
            toy_eps2_uls_q97=float(q97), toy_eps2_uls_mean=float(mean_e),
            p_strong=float(p_strong), p_weak=float(p_weak), p_two=float(p_two),
            meta=str(meta),
            # Provenance
            bands_refit_gp_on_toy=bool(refit_gp_on_toy),
            bands_train_exclude_nsigma=float(train_exclude_nsigma),
            bands_refit_restarts=int(refit_restarts),
            bands_refit_optimize=bool(refit_optimize),
        )

    if n_workers <= 1 or not _HAVE_JOBLIB:
        rows = [_one_mass(i, m) for i, m in enumerate(masses)]
    else:
        rows = joblib.Parallel(n_jobs=int(n_workers), backend=str(backend))(
            joblib.delayed(_one_mass)(i, m) for i, m in enumerate(masses)
        )

    return pd.DataFrame(rows).sort_values("mass_GeV").reset_index(drop=True)
