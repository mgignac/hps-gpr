"""Signal injection and extraction closure tests."""

import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .io import estimate_background_for_dataset
from .template import build_template
from .statistics import (
    fit_A_profiled_gaussian,
    fit_A_profiled_gaussian_details,
    draw_bkg_mvn_nonneg,
)
from .gpr import make_kernel_for_dataset, fit_gpr, predict_counts_from_log_gpr
from .plotting import ensure_dir

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig
    from .io import BlindPrediction


# ---------------------------------------------------------------------------
# Low-level injection helpers
# ---------------------------------------------------------------------------

def _inject_counts_weighted(
    weights: np.ndarray, total: int, rng: np.random.Generator
) -> np.ndarray:
    """Multinomial draw across bins (weights must sum to ~1)."""
    total = int(total)
    if total <= 0:
        return np.zeros_like(weights, dtype=int)
    w = np.asarray(weights, float)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.zeros_like(weights, dtype=int)
    return rng.multinomial(total, w / wsum)


def _inject_counts_from_template(
    template: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    mode: str = "multinomial",
) -> Tuple[np.ndarray, int, float]:
    """Inject a signal distributed like `template` at amplitude `strength`.

    Returns:
        (sig_counts, Nsig_realized, f_sumw)
    where f_sumw = Σw (the sum of template weights, useful for leakage diagnostics).
    """
    w = np.asarray(template, float)
    f = float(np.sum(w))
    if not np.isfinite(f) or f <= 0 or not np.isfinite(strength) or strength <= 0:
        return np.zeros_like(w, dtype=int), 0, f

    if mode == "poisson":
        sig = rng.poisson(float(strength) * w)
        return sig.astype(int), int(np.sum(sig)), f

    # multinomial: interpret strength as total yield parameter
    Ntot = int(np.round(float(strength)))
    if Ntot <= 0:
        return np.zeros_like(w, dtype=int), 0, f

    if f < 1.0:
        Nwin = int(rng.binomial(Ntot, f))
        if Nwin <= 0:
            return np.zeros_like(w, dtype=int), 0, f
        sig = _inject_counts_weighted(w / f, Nwin, rng)
    else:
        sig = _inject_counts_weighted(w / f, Ntot, rng)

    return sig.astype(int), int(np.sum(sig)), f


def inject_counts(
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    strength: int,
    rng: np.random.Generator,
    mode: str = "multinomial",
) -> np.ndarray:
    """Inject signal counts into bins (legacy interface, kept for back-compat)."""
    if strength <= 0:
        return np.zeros(len(edges) - 1, dtype=int)
    w = build_template(edges, mass, sigma_val)
    if mode == "multinomial":
        return rng.multinomial(int(strength), w).astype(int)
    return rng.poisson(float(strength) * w).astype(int)


# ---------------------------------------------------------------------------
# Reference sigma_A for strength scaling
# ---------------------------------------------------------------------------

def _sigmaA_reference(
    pred: "BlindPrediction",
    mass: float,
    *,
    source: str = "asimov",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate σ_A(m) under B-only for 'sigma-level' injection scaling."""
    tmpl = build_template(pred.edges, mass, pred.sigma_val)
    b = np.asarray(pred.mu, float)
    if source == "poisson":
        if rng is None:
            rng = np.random.default_rng(12345)
        n_ref = rng.poisson(b)
    else:
        n_ref = b  # Asimov
    d = fit_A_profiled_gaussian_details(n_ref, b, pred.cov, tmpl, allow_negative=True)
    return float(d.get("sigma_A", np.nan))


# ---------------------------------------------------------------------------
# Main injection/extraction study
# ---------------------------------------------------------------------------

def run_injection_extraction_toys(
    ds: "DatasetConfig",
    config: "Config",
    *,
    masses: List[float],
    strengths: Optional[List[float]] = None,
    n_toys: int = 200,
    outdir: str = None,
    seed: int = 314159,
    inj_mode: Optional[str] = None,
    strengths_mode: Optional[str] = None,
    sigma_multipliers: Optional[List[float]] = None,
    sigma_source: Optional[str] = None,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    inj_shape_mode: Optional[str] = None,
    train_exclude_nsigma: Optional[float] = None,
) -> pd.DataFrame:
    """Run injection+extraction closure tests for one dataset.

    Modes
    -----
    (1) Conditional blind-window toys (fast, default):
        Draw background from the GP posterior MVN, Poisson-sample obs, inject in window only.

    (2) Full procedural toys with GP refit (slow):
        Full-range pseudo-dataset, GP refit on toy sidebands, extract in blind window.
        Controlled by refit_gp_on_toy=True.

    strengths_mode:
        'absolute' — inject `strengths` counts (or use config.inj_strengths)
        'sigmaA'   — inject multiples of σ_A (use sigma_multipliers)
    """
    if outdir is None:
        outdir = os.path.join(config.output_dir, "injection_extraction")
    ensure_dir(outdir)

    rng = np.random.default_rng(int(seed))

    # Resolve mode toggles from config or arguments
    strengths_mode = (strengths_mode or getattr(config, "inj_strength_mode", "absolute")).lower().strip()
    sigma_source = (sigma_source or getattr(config, "inj_sigma_a_source", "asimov")).lower().strip()
    inj_mode = (inj_mode or config.inj_mode).lower().strip()

    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "inj_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "inj_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "inj_refit_gp_optimize", True))
    inj_shape_mode = (inj_shape_mode or getattr(config, "inj_shape_mode", "full")).lower().strip()
    if inj_shape_mode not in ("full", "window"):
        inj_shape_mode = "full"
    if train_exclude_nsigma is None:
        train_exclude_nsigma = getattr(config, "inj_train_exclude_nsigma", None)

    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))

    # Strength grid
    if strengths_mode == "sigmaa":
        mults = sigma_multipliers or getattr(config, "inj_sigma_multipliers", [0.0, 1.0, 2.0, 3.0])
        strength_tags = [float(x) for x in mults]
    else:
        strength_tags = [float(x) for x in (strengths or config.inj_strengths)]

    train_nsig_default = float(
        getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma
    )

    rows: List[Dict] = []

    for m in masses:
        m = float(m)
        pred = estimate_background_for_dataset(ds, m, config)

        tmpl_win = build_template(pred.edges, m, pred.sigma_val)
        f_win = float(np.sum(tmpl_win))

        # Full-range template for leakage diagnostics
        edges_full = np.asarray(pred.edges_full, float)
        x_full = np.asarray(pred.x_full, float).reshape(-1)
        tmpl_full = build_template(edges_full, m, pred.sigma_val)
        f_full = float(np.sum(tmpl_full))

        sigmaA_ref = _sigmaA_reference(pred, m, source=sigma_source, rng=rng)

        if strengths_mode == "sigmaa":
            A_inj_list = [float(t) * float(sigmaA_ref) for t in strength_tags]
            inj_nsigma_list = list(strength_tags)
        else:
            A_inj_list = list(strength_tags)
            inj_nsigma_list = [
                float(A) / float(sigmaA_ref) if np.isfinite(sigmaA_ref) and sigmaA_ref > 0 else np.nan
                for A in A_inj_list
            ]

        blind = tuple(pred.blind)
        msk_blind = (x_full >= blind[0]) & (x_full <= blind[1])
        if int(np.sum(msk_blind)) == 0:
            raise RuntimeError(f"[inj][{ds.key}] m={m:.6g}: blind window has no bins")

        if train_exclude_nsigma is not None:
            tn = float(train_exclude_nsigma)
        else:
            tn = float(getattr(pred, "train_exclude_nsigma", train_nsig_default))
        blind_train = (m - tn * float(pred.sigma_val), m + tn * float(pred.sigma_val))
        msk_train = (x_full < blind_train[0]) | (x_full > blind_train[1])

        f_train = float(np.sum(tmpl_full[msk_train])) if tmpl_full.shape[0] == x_full.shape[0] else float("nan")
        f_train_frac = float(f_train / f_full) if np.isfinite(f_train) and f_full > 0 else float("nan")

        toy_mode = "full_refit" if refit_gp_on_toy else "conditional_gp"

        if refit_gp_on_toy:
            mu_full_arr = np.asarray(pred.mu_full, float).reshape(-1)
            ker = make_kernel_for_dataset(ds, config, mass=m)
            x_win = x_full[msk_blind]
        else:
            b_draws = draw_bkg_mvn_nonneg(
                pred.mu, pred.cov, int(n_toys) * len(strength_tags),
                rng, method=mvn_method, max_tries=mvn_max_tries,
            )
            draw_idx = 0

        for A_inj, inj_nsigma in zip(A_inj_list, inj_nsigma_list):
            A_inj = float(A_inj)

            for i in range(int(n_toys)):
                refit_ok = float("nan")

                if refit_gp_on_toy:
                    bkg_full = rng.poisson(np.clip(mu_full_arr, 0.0, None)).astype(int)
                    if inj_shape_mode == "window":
                        sig_full = np.zeros_like(bkg_full, dtype=int)
                        s_win, Nsig_win, _ = _inject_counts_from_template(tmpl_win, A_inj, rng, inj_mode)
                        idx_blind = np.where(msk_blind)[0]
                        n = min(len(s_win), len(idx_blind))
                        sig_full[idx_blind[:n]] = s_win[:n]
                        Nsig_full = int(np.sum(sig_full))
                    else:
                        s_full, Nsig_full, _ = _inject_counts_from_template(tmpl_full, A_inj, rng, inj_mode)
                        sig_full = np.asarray(s_full, dtype=int)
                        Nsig_win = int(np.sum(sig_full[msk_blind]))

                    y_toy = (bkg_full + sig_full).astype(int)
                    obs = y_toy[msk_blind].astype(int)
                    Nsig_train = int(np.sum(sig_full[msk_train]))

                    mu_fit = np.asarray(pred.mu, float)
                    cov_fit = np.asarray(pred.cov, float)
                    try:
                        X_tr = x_full[msk_train]
                        y_tr = y_toy[msk_train].astype(float)
                        gpr = fit_gpr(X_tr, y_tr, config, restarts=refit_restarts, kernel=ker, optimize=refit_optimize)
                        mu_fit, cov_fit = predict_counts_from_log_gpr(gpr, x_win, config)
                        refit_ok = 1.0
                    except Exception:
                        refit_ok = 0.0

                else:
                    b = b_draws[draw_idx % b_draws.shape[0]]
                    draw_idx += 1
                    sig, Nsig_win, _ = _inject_counts_from_template(tmpl_win, A_inj, rng, inj_mode)
                    lam = np.clip(b, 0.0, None) + np.clip(sig.astype(float), 0.0, None)
                    obs = rng.poisson(lam).astype(int)
                    Nsig_train = 0
                    mu_fit, cov_fit = pred.mu, pred.cov

                fit = fit_A_profiled_gaussian(
                    obs, mu_fit, cov_fit, tmpl_win,
                    allow_negative=bool(getattr(config, "extract_allow_negative", True)),
                )
                A_hat = float(fit["A_hat"])
                sigma_A = float(fit["sigma_A"])
                pull = (A_hat - A_inj) / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")
                Zhat = A_hat / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")

                rows.append(dict(
                    dataset=ds.key, mass_GeV=m, toy=int(i),
                    strength=float(A_inj), inj_nsigma=float(inj_nsigma),
                    sigmaA_ref=float(sigmaA_ref), sigma_val=float(pred.sigma_val),
                    sigma_x=float(getattr(pred, "sigma_x", float("nan"))),
                    f_win=float(f_win), f_full=float(f_full),
                    f_train=float(f_train), f_train_frac=float(f_train_frac),
                    blind_nsigma=float(config.blind_nsigma), train_exclude_nsigma=float(tn),
                    inj_shape_mode=inj_shape_mode,
                    A_hat=float(A_hat), sigma_A=float(sigma_A),
                    Zhat=float(Zhat), pull_param=float(pull),
                    Nsig_win=int(Nsig_win), Nsig_train=int(Nsig_train),
                    success=bool(fit["success"]), nll=float(fit.get("nll", float("nan"))),
                    toy_mode=toy_mode, refit_gp_on_toy=bool(refit_gp_on_toy),
                    refit_ok=float(refit_ok), refit_restarts=int(refit_restarts),
                    refit_optimize=bool(refit_optimize),
                ))

        print(f"[inj] {ds.key} m={m:.6g} GeV done ({toy_mode}; {len(strength_tags)} strengths × {n_toys} toys)")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, f"inj_extract_toys_{ds.key}.csv")
    df.to_csv(out_csv, index=False)
    print("[inj] wrote", out_csv)
    return df


def run_injection_extraction(
    ds: "DatasetConfig",
    masses: List[float],
    strengths: List[int],
    config: "Config",
    outdir: str = None,
    seed: int = 314159,
) -> pd.DataFrame:
    """Legacy single-toy injection/extraction (kept for back-compat)."""
    return run_injection_extraction_toys(
        ds, config, masses=masses, strengths=[float(s) for s in strengths],
        n_toys=1, outdir=outdir, seed=seed, strengths_mode="absolute",
    )


def summarize_injection_grid(df_toys: pd.DataFrame) -> pd.DataFrame:
    """Summarize injection toys by (dataset, mass, strength)."""
    if df_toys.empty:
        return pd.DataFrame()

    def q(x, p):
        v = np.asarray(x, float)
        return float(np.nanquantile(v, p)) if np.any(np.isfinite(v)) else float("nan")

    rows = []
    for (ds, m, A), sub in df_toys.groupby(["dataset", "mass_GeV", "strength"], dropna=False):
        Ahat = sub["A_hat"].to_numpy(float)
        sigA = sub["sigma_A"].to_numpy(float)
        pull = sub["pull_param"].to_numpy(float)
        Zhat = sub["Zhat"].to_numpy(float)

        rows.append(dict(
            dataset=ds, mass_GeV=float(m), strength=float(A),
            inj_nsigma=float(np.nanmean(sub["inj_nsigma"].to_numpy(float))),
            sigmaA_ref=float(np.nanmean(sub["sigmaA_ref"].to_numpy(float))),
            f_win=float(np.nanmean(sub["f_win"].to_numpy(float))),
            f_train_frac=float(np.nanmean(sub["f_train_frac"].to_numpy(float))) if "f_train_frac" in sub.columns else float("nan"),
            A_hat_mean=float(np.nanmean(Ahat)),
            A_hat_std=float(np.nanstd(Ahat, ddof=1)),
            sigma_A_mean=float(np.nanmean(sigA)),
            pull_mean=float(np.nanmean(pull)),
            pull_std=float(np.nanstd(pull, ddof=1)),
            pull_q16=q(pull, 0.16), pull_q84=q(pull, 0.84),
            pull_q02=q(pull, 0.025), pull_q97=q(pull, 0.975),
            cov_1sigma=float(np.nanmean(np.abs(pull) < 1.0)),
            cov_2sigma=float(np.nanmean(np.abs(pull) < 2.0)),
            Zhat_mean=float(np.nanmean(Zhat)),
            Zhat_q16=q(Zhat, 0.16), Zhat_q84=q(Zhat, 0.84),
            success_rate=float(np.nanmean(sub["success"].to_numpy(float))),
        ))

    return pd.DataFrame(rows).sort_values(["dataset", "mass_GeV", "strength"]).reset_index(drop=True)
