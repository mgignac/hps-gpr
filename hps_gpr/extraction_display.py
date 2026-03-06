"""Reviewer-facing extraction display plots for representative pseudoexperiments."""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .conversion import A_from_epsilon2, epsilon2_from_A
from .dataset import DatasetConfig, make_datasets
from .evaluation import combined_cls_limit_epsilon2
from .gpr import fit_gpr, make_kernel_for_dataset, predict_counts_from_log_gpr
from .injection import _inject_counts_from_template, _sigmaA_reference
from .io import BlindPrediction, estimate_background_for_dataset
from .plotting import (
    _grid,
    _save_plot_outputs,
    _set_title_above,
    _shade_blind_window,
    ensure_dir,
    set_injection_plot_style,
)
from .statistics import fit_A_profiled_gaussian_details
from .template import build_template, cls_limit_for_amplitude, gaussian_bin_integrals


@dataclass
class _SingleDisplayContext:
    ds: DatasetConfig
    mass: float
    pred: BlindPrediction
    sigmaA_ref: float
    A_per_eps2_unit: float
    A_up_obs: float
    eps2_up_obs: float
    blind_mask: np.ndarray
    blind_train: tuple
    raw_signal_full: np.ndarray
    signal_per_blind_amp_full: np.ndarray
    blind_fraction: float


@dataclass
class SingleExtractionDisplay:
    ctx: _SingleDisplayContext
    inj_nsigma: float
    A_inj_window: float
    eps2_inj: float
    seed: int
    refit_gp_on_toy: bool
    refit_ok: bool
    y_full_toy: np.ndarray
    y_win_toy: np.ndarray
    mu_full_plot: np.ndarray
    mu_win_fit: np.ndarray
    cov_win_fit: np.ndarray
    b_fit: np.ndarray
    lambda_fit: np.ndarray
    A_hat: float
    sigma_A: float
    fit_success: bool
    Nsig_realized_total: int
    Nsig_realized_blind: int
    Nsig_expected_total: float

    @property
    def x_full(self) -> np.ndarray:
        return np.asarray(self.ctx.pred.x_full, float)

    @property
    def x_win(self) -> np.ndarray:
        return self.x_full[self.ctx.blind_mask]

    @property
    def blind(self) -> tuple:
        return tuple(self.ctx.pred.blind)

    @property
    def pull(self) -> float:
        if np.isfinite(self.sigma_A) and self.sigma_A > 0:
            return float((self.A_hat - self.A_inj_window) / self.sigma_A)
        return float("nan")

    @property
    def signal_curve_injected(self) -> np.ndarray:
        return float(self.A_inj_window) * self.ctx.signal_per_blind_amp_full

    @property
    def signal_curve_extracted(self) -> np.ndarray:
        return float(self.A_hat) * self.ctx.signal_per_blind_amp_full

    def to_metadata(self) -> Dict[str, object]:
        return {
            "dataset": str(self.ctx.ds.key),
            "mass_GeV": float(self.ctx.mass),
            "mass_MeV": float(self.ctx.mass * 1e3),
            "inj_nsigma": float(self.inj_nsigma),
            "A_inj_window": float(self.A_inj_window),
            "A_hat": float(self.A_hat),
            "sigma_A": float(self.sigma_A),
            "pull": float(self.pull),
            "eps2_inj": float(self.eps2_inj),
            "eps2_up_obs": float(self.ctx.eps2_up_obs),
            "A_up_obs": float(self.ctx.A_up_obs),
            "sigmaA_ref": float(self.ctx.sigmaA_ref),
            "blind_fraction": float(self.ctx.blind_fraction),
            "Nsig_expected_total": float(self.Nsig_expected_total),
            "Nsig_realized_total": int(self.Nsig_realized_total),
            "Nsig_realized_blind": int(self.Nsig_realized_blind),
            "refit_gp_on_toy": bool(self.refit_gp_on_toy),
            "refit_ok": bool(self.refit_ok),
            "fit_success": bool(self.fit_success),
            "seed": int(self.seed),
        }


@dataclass
class CombinedExtractionDisplay:
    mass: float
    inj_nsigma_combined: float
    eps2_inj: float
    eps2_hat: float
    sigma_eps2: float
    Zhat_combined: float
    eps2_up_obs: float
    displays: List[SingleExtractionDisplay]
    dataset_keys: List[str]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "mass_GeV": float(self.mass),
            "mass_MeV": float(self.mass * 1e3),
            "inj_nsigma_combined": float(self.inj_nsigma_combined),
            "eps2_inj": float(self.eps2_inj),
            "eps2_hat": float(self.eps2_hat),
            "sigma_eps2": float(self.sigma_eps2),
            "Zhat_combined": float(self.Zhat_combined),
            "eps2_up_obs": float(self.eps2_up_obs),
            "datasets": list(self.dataset_keys),
            "per_dataset": [d.to_metadata() for d in self.displays],
        }


def _stable_display_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int((int(base_seed) + int.from_bytes(digest, "little")) & 0xFFFFFFFF)


def _strength_tag(z: float) -> str:
    return str(float(z)).replace("-", "m").replace(".", "p")


def _zoom_mask(display: SingleExtractionDisplay, zoom_half_sigma: float) -> np.ndarray:
    pred = display.ctx.pred
    zlo = float(pred.blind[0] - float(zoom_half_sigma) * float(pred.sigma_val))
    zhi = float(pred.blind[1] + float(zoom_half_sigma) * float(pred.sigma_val))
    x_full = display.x_full
    return (x_full >= zlo) & (x_full <= zhi)


def _build_single_context(
    ds: DatasetConfig,
    config,
    *,
    mass: float,
    sigma_source: str,
    train_exclude_nsigma: Optional[float],
) -> _SingleDisplayContext:
    pred = estimate_background_for_dataset(
        ds,
        float(mass),
        config,
        train_exclude_nsigma=train_exclude_nsigma,
    )
    sigma_rng = np.random.default_rng(
        _stable_display_seed(int(getattr(config, "extraction_display_seed", 271828)), ds.key, float(mass), "sigmaA")
    )
    sigmaA_ref = _sigmaA_reference(pred, float(mass), source=str(sigma_source), rng=sigma_rng)
    A_per_eps2_unit = float(A_from_epsilon2(ds, float(mass), 1.0, pred.integral_density))

    seed_cls = None
    if str(getattr(config, "cls_mode", "asymptotic")).lower().strip() == "toys":
        seed_cls = _stable_display_seed(int(getattr(config, "cls_seed_base", 12345)), ds.key, float(mass), "display-cls")
    A_up_obs, _ = cls_limit_for_amplitude(
        n_obs=pred.obs,
        b_mean=pred.mu,
        b_cov=pred.cov,
        edges=pred.edges,
        mass=float(mass),
        sigma_val=float(pred.sigma_val),
        config=config,
        seed=seed_cls,
    )
    eps2_up_obs = float(epsilon2_from_A(ds, float(mass), float(A_up_obs), pred.integral_density))

    x_full = np.asarray(pred.x_full, float).reshape(-1)
    blind = tuple(pred.blind)
    blind_mask = (x_full >= float(blind[0])) & (x_full <= float(blind[1]))
    raw_signal_full = gaussian_bin_integrals(np.asarray(pred.edges_full, float), float(mass), float(pred.sigma_val))
    blind_fraction = float(np.sum(raw_signal_full[blind_mask]))
    if not np.isfinite(blind_fraction) or blind_fraction <= 0:
        raise RuntimeError(f"{ds.key} m={float(mass):.6g}: blind-window signal fraction is non-positive")
    signal_per_blind_amp_full = raw_signal_full / blind_fraction

    tn = getattr(config, "extraction_display_train_exclude_nsigma", None)
    if train_exclude_nsigma is not None:
        tn = train_exclude_nsigma
    if tn is None:
        blind_train_pred = getattr(pred, "blind_train", None)
        if blind_train_pred is not None:
            tn = float((float(blind_train_pred[1]) - float(mass)) / float(pred.sigma_val))
        else:
            tn = getattr(config, "gp_train_exclude_nsigma", config.blind_nsigma)
    blind_train = (
        float(mass) - float(tn) * float(pred.sigma_val),
        float(mass) + float(tn) * float(pred.sigma_val),
    )

    return _SingleDisplayContext(
        ds=ds,
        mass=float(mass),
        pred=pred,
        sigmaA_ref=float(sigmaA_ref),
        A_per_eps2_unit=float(A_per_eps2_unit),
        A_up_obs=float(A_up_obs),
        eps2_up_obs=float(eps2_up_obs),
        blind_mask=blind_mask,
        blind_train=blind_train,
        raw_signal_full=np.asarray(raw_signal_full, float),
        signal_per_blind_amp_full=np.asarray(signal_per_blind_amp_full, float),
        blind_fraction=float(blind_fraction),
    )


def _simulate_single_display_from_context(
    ctx: _SingleDisplayContext,
    config,
    *,
    inj_nsigma: float,
    A_inj_window: float,
    seed: int,
    inj_mode: str,
    refit_gp_on_toy: bool,
    gp_restarts: int,
    gp_optimize: bool,
) -> SingleExtractionDisplay:
    rng = np.random.default_rng(int(seed))
    pred = ctx.pred
    x_full = np.asarray(pred.x_full, float).reshape(-1)
    mu_full = np.asarray(pred.mu_full, float).reshape(-1)

    bkg_full = rng.poisson(np.clip(mu_full, 0.0, None)).astype(int)
    expected_total_strength = float(A_inj_window) / float(ctx.blind_fraction)
    sig_full, n_sig_realized_total, _ = _inject_counts_from_template(
        ctx.raw_signal_full,
        expected_total_strength,
        rng,
        mode=str(inj_mode),
    )
    y_full_toy = (bkg_full + np.asarray(sig_full, int)).astype(int)
    y_win_toy = y_full_toy[ctx.blind_mask].astype(int)
    n_sig_realized_blind = int(np.sum(np.asarray(sig_full, int)[ctx.blind_mask]))

    mu_win_fit = np.asarray(pred.mu, float)
    cov_win_fit = np.asarray(pred.cov, float)
    mu_full_plot = np.asarray(mu_full, float)
    refit_ok = False
    if bool(refit_gp_on_toy):
        mask_train = (x_full < float(ctx.blind_train[0])) | (x_full > float(ctx.blind_train[1]))
        try:
            gpr = fit_gpr(
                x_full[mask_train],
                y_full_toy[mask_train].astype(float),
                config,
                restarts=int(gp_restarts),
                kernel=make_kernel_for_dataset(ctx.ds, config, mass=float(ctx.mass)),
                optimize=bool(gp_optimize),
            )
            mu_full_plot, _ = predict_counts_from_log_gpr(gpr, x_full, config)
            mu_win_fit, cov_win_fit = predict_counts_from_log_gpr(gpr, x_full[ctx.blind_mask], config)
            mu_full_plot = np.asarray(mu_full_plot, float).reshape(-1)
            mu_win_fit = np.asarray(mu_win_fit, float).reshape(-1)
            cov_win_fit = np.asarray(cov_win_fit, float)
            refit_ok = True
        except Exception:
            refit_ok = False

    fitd = fit_A_profiled_gaussian_details(
        y_win_toy,
        mu_win_fit,
        cov_win_fit,
        build_template(pred.edges, float(ctx.mass), float(pred.sigma_val)),
        allow_negative=bool(getattr(config, "extract_allow_negative", True)),
    )
    return SingleExtractionDisplay(
        ctx=ctx,
        inj_nsigma=float(inj_nsigma),
        A_inj_window=float(A_inj_window),
        eps2_inj=float(epsilon2_from_A(ctx.ds, float(ctx.mass), float(A_inj_window), pred.integral_density)),
        seed=int(seed),
        refit_gp_on_toy=bool(refit_gp_on_toy),
        refit_ok=bool(refit_ok),
        y_full_toy=np.asarray(y_full_toy, int),
        y_win_toy=np.asarray(y_win_toy, int),
        mu_full_plot=np.asarray(mu_full_plot, float),
        mu_win_fit=np.asarray(mu_win_fit, float),
        cov_win_fit=np.asarray(cov_win_fit, float),
        b_fit=np.asarray(fitd.get("b_fit", mu_win_fit), float).reshape(-1),
        lambda_fit=np.asarray(fitd.get("lambda_hat", mu_win_fit), float).reshape(-1),
        A_hat=float(fitd.get("A_hat", float("nan"))),
        sigma_A=float(fitd.get("sigma_A", float("nan"))),
        fit_success=bool(fitd.get("success", False)),
        Nsig_realized_total=int(n_sig_realized_total),
        Nsig_realized_blind=int(n_sig_realized_blind),
        Nsig_expected_total=float(expected_total_strength),
    )


def make_single_extraction_display(
    ds: DatasetConfig,
    config,
    *,
    mass: float,
    inj_nsigma: float,
    seed: int,
) -> SingleExtractionDisplay:
    ctx = _build_single_context(
        ds,
        config,
        mass=float(mass),
        sigma_source=str(getattr(config, "extraction_display_sigma_source", "asimov")),
        train_exclude_nsigma=getattr(config, "extraction_display_train_exclude_nsigma", None),
    )
    return _simulate_single_display_from_context(
        ctx,
        config,
        inj_nsigma=float(inj_nsigma),
        A_inj_window=float(inj_nsigma) * float(ctx.sigmaA_ref),
        seed=int(seed),
        inj_mode=str(getattr(config, "extraction_display_inj_mode", "multinomial")),
        refit_gp_on_toy=bool(getattr(config, "extraction_display_refit_gp_on_toy", True)),
        gp_restarts=int(getattr(config, "extraction_display_gp_restarts", 0)),
        gp_optimize=bool(getattr(config, "extraction_display_gp_optimize", True)),
    )


def make_combined_extraction_display(
    datasets: Sequence[DatasetConfig],
    config,
    *,
    mass: float,
    inj_nsigma: float,
    seed: int,
) -> CombinedExtractionDisplay:
    contexts = [
        _build_single_context(
            ds,
            config,
            mass=float(mass),
            sigma_source=str(getattr(config, "extraction_display_sigma_source", "asimov")),
            train_exclude_nsigma=getattr(config, "extraction_display_train_exclude_nsigma", None),
        )
        for ds in datasets
    ]
    info_sum = float(
        np.sum(
            [
                (ctx.A_per_eps2_unit * ctx.A_per_eps2_unit) / (ctx.sigmaA_ref * ctx.sigmaA_ref)
                for ctx in contexts
                if np.isfinite(ctx.A_per_eps2_unit)
                and np.isfinite(ctx.sigmaA_ref)
                and ctx.sigmaA_ref > 0
            ]
        )
    )
    if not np.isfinite(info_sum) or info_sum <= 0:
        raise RuntimeError(f"m={float(mass):.6g}: could not build combined epsilon^2 information")
    sigma_eps2 = float(1.0 / np.sqrt(info_sum))
    eps2_inj = float(inj_nsigma) * float(sigma_eps2)

    displays = []
    for ctx in contexts:
        A_inj_window = float(ctx.A_per_eps2_unit) * float(eps2_inj)
        ds_seed = _stable_display_seed(int(seed), str(ctx.ds.key), float(mass), float(inj_nsigma))
        displays.append(
            _simulate_single_display_from_context(
                ctx,
                config,
                inj_nsigma=float(A_inj_window / ctx.sigmaA_ref) if ctx.sigmaA_ref > 0 else float("nan"),
                A_inj_window=float(A_inj_window),
                seed=int(ds_seed),
                inj_mode=str(getattr(config, "extraction_display_inj_mode", "multinomial")),
                refit_gp_on_toy=bool(getattr(config, "extraction_display_refit_gp_on_toy", True)),
                gp_restarts=int(getattr(config, "extraction_display_gp_restarts", 0)),
                gp_optimize=bool(getattr(config, "extraction_display_gp_optimize", True)),
            )
        )

    num = 0.0
    denom = 0.0
    for disp in displays:
        ctx = disp.ctx
        if np.isfinite(disp.A_hat) and np.isfinite(disp.sigma_A) and disp.sigma_A > 0 and np.isfinite(ctx.A_per_eps2_unit):
            w = float(ctx.A_per_eps2_unit) / float(disp.sigma_A * disp.sigma_A)
            num += w * float(disp.A_hat)
            denom += float(ctx.A_per_eps2_unit) * w
    eps2_hat = float(num / denom) if denom > 0 else float("nan")
    sigma_eps2_hat = float(1.0 / np.sqrt(denom)) if denom > 0 else float("nan")
    zhat_combined = float(eps2_hat / sigma_eps2_hat) if np.isfinite(sigma_eps2_hat) and sigma_eps2_hat > 0 else float("nan")
    eps2_up_obs = float(combined_cls_limit_epsilon2(float(mass), [ctx.ds for ctx in contexts], [ctx.pred for ctx in contexts], config))

    return CombinedExtractionDisplay(
        mass=float(mass),
        inj_nsigma_combined=float(inj_nsigma),
        eps2_inj=float(eps2_inj),
        eps2_hat=float(eps2_hat),
        sigma_eps2=float(sigma_eps2_hat),
        Zhat_combined=float(zhat_combined),
        eps2_up_obs=float(eps2_up_obs),
        displays=displays,
        dataset_keys=[str(ctx.ds.key) for ctx in contexts],
    )


def plot_single_extraction_display(
    display: SingleExtractionDisplay,
    *,
    outpath: str,
    blind_shade_alpha: float,
    blind_shade_color: str,
    zoom_half_sigma: float,
) -> None:
    set_injection_plot_style("paper")
    ensure_dir(os.path.dirname(outpath) or ".")

    fig = plt.figure(figsize=(13.2, 9.4), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, width_ratios=[4.9, 1.55], height_ratios=[1.1, 1.0, 0.92], wspace=0.16, hspace=0.24)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_fit = fig.add_subplot(gs[1, 0])
    ax_sig = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    pred = display.ctx.pred
    blind_train = display.ctx.blind_train
    x_full = display.x_full
    blind_mask = display.ctx.blind_mask
    x_win = display.x_win

    yerr_full = np.sqrt(np.clip(display.y_full_toy, 1.0, None))
    ax_full.errorbar(x_full, display.y_full_toy, yerr=yerr_full, fmt="o", ms=2.6, lw=0.9, color="black", label="Pseudoexperiment", zorder=4)
    ax_full.plot(x_full, display.mu_full_plot, color="#1f77b4", lw=1.6, label="GP background", zorder=5)
    ax_full.plot(x_full, display.mu_full_plot + display.signal_curve_injected, color="#C44E52", lw=1.5, ls=":", label="Injected mean", zorder=6)
    _shade_blind_window(ax_full, pred.blind, blind_train=blind_train, alpha=float(blind_shade_alpha) * 0.8, color=str(blind_shade_color), zorder=0)
    ax_full.set_ylabel("Counts / bin")
    _set_title_above(
        ax_full,
        f"{display.ctx.ds.label} — Injected {display.inj_nsigma:.0f} sigma [{display.Nsig_realized_total} realized events] at {display.ctx.mass * 1e3:.0f} MeV",
    )
    ax_full.legend(loc="upper left", frameon=True, fontsize=9)
    _grid(ax_full)

    m_zoom = _zoom_mask(display, float(zoom_half_sigma))
    xz = x_full[m_zoom]
    yz = display.y_full_toy[m_zoom]
    muz = display.mu_full_plot[m_zoom]
    ax_fit.errorbar(xz, yz, yerr=np.sqrt(np.clip(yz, 1.0, None)), fmt="o", ms=3.0, lw=0.9, color="black", label="Pseudoexperiment", zorder=4)
    ax_fit.plot(xz, muz, color="#1f77b4", lw=1.5, label="GP background", zorder=5)
    ax_fit.plot(x_win, display.b_fit, color="#9467bd", lw=1.5, label="Profiled background", zorder=6)
    ax_fit.plot(x_win, display.lambda_fit, color="#6A3D9A", lw=1.7, label="Blind-window fit", zorder=7)
    _shade_blind_window(ax_fit, pred.blind, blind_train=blind_train, alpha=float(blind_shade_alpha), color=str(blind_shade_color), zorder=0)
    ax_fit.set_xlim(float(xz[0]), float(xz[-1]))
    ax_fit.set_ylabel("Counts / bin")
    _set_title_above(ax_fit, f"Blind-window fit ({zoom_half_sigma:.1f} sigma sideband padding)")
    ax_fit.legend(loc="upper left", frameon=True, fontsize=8.8)
    _grid(ax_fit)

    sig_obs = display.y_win_toy - display.b_fit
    ax_sig.errorbar(x_win, sig_obs, yerr=np.sqrt(np.clip(display.y_win_toy, 1.0, None)), fmt="o", ms=3.0, lw=0.9, color="black", label="Data - profiled background", zorder=4)
    ax_sig.plot(xz, display.signal_curve_injected[m_zoom], color="#C44E52", lw=1.6, ls=":", label="Injected signal", zorder=5)
    ax_sig.plot(xz, display.signal_curve_extracted[m_zoom], color="#2CA02C", lw=1.8, label="Extracted signal", zorder=6)
    ax_sig.axhline(0.0, color="0.2", lw=0.9, alpha=0.7, zorder=1)
    _shade_blind_window(ax_sig, pred.blind, blind_train=blind_train, alpha=float(blind_shade_alpha), color=str(blind_shade_color), zorder=0)
    ax_sig.set_xlim(float(xz[0]), float(xz[-1]))
    ax_sig.set_xlabel("Mass [GeV]")
    ax_sig.set_ylabel("Signal counts / bin")
    _set_title_above(ax_sig, "Extracted signal component")
    ax_sig.legend(loc="upper left", frameon=True, fontsize=8.8)
    _grid(ax_sig)

    ul_ratio = float(display.eps2_inj / display.ctx.eps2_up_obs) if np.isfinite(display.ctx.eps2_up_obs) and display.ctx.eps2_up_obs > 0 else float("nan")
    info_lines = [
        f"Dataset: {display.ctx.ds.key}",
        f"Mass: {display.ctx.mass * 1e3:.0f} MeV",
        f"Injected level: {display.inj_nsigma:.1f} sigma",
        f"A_inj (blind): {display.A_inj_window:.1f}",
        f"A_hat: {display.A_hat:.1f} ± {display.sigma_A:.1f}",
        f"Pull: {display.pull:.2f}",
        f"eps^2_inj: {display.eps2_inj:.3e}",
        f"obs UL eps^2: {display.ctx.eps2_up_obs:.3e}",
        f"eps^2_inj / UL: {ul_ratio:.2f}" if np.isfinite(ul_ratio) else "eps^2_inj / UL: n/a",
        f"Sigma_A ref: {display.ctx.sigmaA_ref:.1f}",
        f"Blind frac: {display.ctx.blind_fraction:.3f}",
        f"Realized signal: {display.Nsig_realized_total} total, {display.Nsig_realized_blind} in blind",
        f"GP refit: {'yes' if display.refit_ok else 'no'}",
        f"Fit success: {'yes' if display.fit_success else 'no'}",
    ]
    ax_info.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.65", alpha=0.96),
    )

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.97)
    _save_plot_outputs(fig, outpath, png_dpi=320)
    with open(os.path.splitext(outpath)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(display.to_metadata(), fh, indent=2, sort_keys=True)


def plot_combined_extraction_display(
    display: CombinedExtractionDisplay,
    *,
    outpath: str,
    blind_shade_alpha: float,
    blind_shade_color: str,
    zoom_half_sigma: float,
) -> None:
    if len(display.displays) < 2:
        raise ValueError("Combined extraction display requires at least two datasets")

    set_injection_plot_style("paper")
    ensure_dir(os.path.dirname(outpath) or ".")

    fig = plt.figure(figsize=(15.0, 11.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.9], height_ratios=[1.0, 0.92, 0.88], wspace=0.18, hspace=0.26)
    ax_fit = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax_sig = [fig.add_subplot(gs[1, i]) for i in range(2)]
    ax_comb = fig.add_subplot(gs[2, 0:2])
    ax_info = fig.add_subplot(gs[:, 2])
    ax_info.axis("off")

    colors = {"2015": "#0072B2", "2016": "#E69F00", "combined": "#111111"}
    sum_A_inj = 0.0
    sum_A_hat = 0.0

    for idx, disp in enumerate(display.displays[:2]):
        pred = disp.ctx.pred
        x_full = disp.x_full
        x_win = disp.x_win
        blind_train = disp.ctx.blind_train
        m_zoom = _zoom_mask(disp, float(zoom_half_sigma))
        xz = x_full[m_zoom]
        yz = disp.y_full_toy[m_zoom]
        muz = disp.mu_full_plot[m_zoom]
        c = colors.get(str(disp.ctx.ds.key), f"C{idx}")

        ax_fit[idx].errorbar(xz, yz, yerr=np.sqrt(np.clip(yz, 1.0, None)), fmt="o", ms=3.0, lw=0.9, color="black", label="Pseudoexperiment", zorder=4)
        ax_fit[idx].plot(xz, muz, color="#1f77b4", lw=1.5, label="GP background", zorder=5)
        ax_fit[idx].plot(x_win, disp.b_fit, color="#9467bd", lw=1.4, label="Profiled background", zorder=6)
        ax_fit[idx].plot(x_win, disp.lambda_fit, color=c, lw=1.8, label="Blind-window fit", zorder=7)
        _shade_blind_window(ax_fit[idx], pred.blind, blind_train=blind_train, alpha=float(blind_shade_alpha), color=str(blind_shade_color), zorder=0)
        ax_fit[idx].set_xlim(float(xz[0]), float(xz[-1]))
        ax_fit[idx].set_ylabel("Counts / bin")
        _set_title_above(ax_fit[idx], f"{disp.ctx.ds.label} blind-window fit")
        ax_fit[idx].legend(loc="upper left", frameon=True, fontsize=8.6)
        _grid(ax_fit[idx])

        sig_obs = disp.y_win_toy - disp.b_fit
        ax_sig[idx].errorbar(x_win, sig_obs, yerr=np.sqrt(np.clip(disp.y_win_toy, 1.0, None)), fmt="o", ms=3.0, lw=0.9, color="black", label="Data - profiled background", zorder=4)
        ax_sig[idx].plot(xz, disp.signal_curve_injected[m_zoom], color="#C44E52", lw=1.6, ls=":", label="Injected signal", zorder=5)
        ax_sig[idx].plot(xz, disp.signal_curve_extracted[m_zoom], color=c, lw=1.8, label="Extracted signal", zorder=6)
        ax_sig[idx].axhline(0.0, color="0.2", lw=0.9, alpha=0.7, zorder=1)
        _shade_blind_window(ax_sig[idx], pred.blind, blind_train=blind_train, alpha=float(blind_shade_alpha), color=str(blind_shade_color), zorder=0)
        ax_sig[idx].set_xlim(float(xz[0]), float(xz[-1]))
        ax_sig[idx].set_xlabel("Mass [GeV]")
        ax_sig[idx].set_ylabel("Signal counts / bin")
        _set_title_above(ax_sig[idx], f"{disp.ctx.ds.label} extracted signal")
        ax_sig[idx].legend(loc="upper left", frameon=True, fontsize=8.6)
        _grid(ax_sig[idx])

        sum_A_inj += float(disp.A_inj_window)
        sum_A_hat += float(disp.A_hat)

    u = np.linspace(-float(display.displays[0].ctx.pred.blind[1] - display.mass) / float(display.displays[0].ctx.pred.sigma_val) - float(zoom_half_sigma),
                    +float(display.displays[0].ctx.pred.blind[1] - display.mass) / float(display.displays[0].ctx.pred.sigma_val) + float(zoom_half_sigma), 400)
    blind_nsigma = float(display.displays[0].ctx.pred.blind[1] - display.mass) / float(display.displays[0].ctx.pred.sigma_val)
    blind_norm = float(stats.norm.cdf(blind_nsigma) - stats.norm.cdf(-blind_nsigma))
    g = stats.norm.pdf(u) / blind_norm
    ax_comb.plot(u, sum_A_inj * g, color="#C44E52", lw=1.7, ls=":", label=f"Injected total ({sum_A_inj:.1f})", zorder=4)
    ax_comb.plot(u, sum_A_hat * g, color=colors["combined"], lw=1.9, label=f"Extracted total ({sum_A_hat:.1f})", zorder=5)
    ax_comb.axvspan(-blind_nsigma, +blind_nsigma, color=str(blind_shade_color), alpha=float(blind_shade_alpha), lw=0, zorder=0)
    ax_comb.axhline(0.0, color="0.2", lw=0.9, alpha=0.7, zorder=1)
    ax_comb.set_xlabel(r"$(m - m_0)/\sigma_m$")
    ax_comb.set_ylabel(r"Combined signal density [events / $\sigma_m$]")
    _set_title_above(ax_comb, "Combined signal-yield Gaussian")
    ax_comb.legend(loc="upper left", frameon=True, fontsize=8.8)
    _grid(ax_comb)

    ul_ratio = float(display.eps2_inj / display.eps2_up_obs) if np.isfinite(display.eps2_up_obs) and display.eps2_up_obs > 0 else float("nan")
    info_lines = [
        f"Combined model: {' + '.join(display.dataset_keys)}",
        f"Mass: {display.mass * 1e3:.0f} MeV",
        f"Injected level: {display.inj_nsigma_combined:.1f} sigma",
        f"eps^2_inj: {display.eps2_inj:.3e}",
        f"eps^2_hat: {display.eps2_hat:.3e} ± {display.sigma_eps2:.3e}",
        f"Zhat_combined: {display.Zhat_combined:.2f}",
        f"obs UL eps^2: {display.eps2_up_obs:.3e}",
        f"eps^2_inj / UL: {ul_ratio:.2f}" if np.isfinite(ul_ratio) else "eps^2_inj / UL: n/a",
        "",
    ]
    for disp in display.displays:
        info_lines.extend(
            [
                f"{disp.ctx.ds.key}: A_inj={disp.A_inj_window:.1f}, A_hat={disp.A_hat:.1f} ± {disp.sigma_A:.1f}",
                f"{disp.ctx.ds.key}: eps^2_UL={disp.ctx.eps2_up_obs:.3e}, signal={disp.Nsig_realized_total} total / {disp.Nsig_realized_blind} blind",
            ]
        )
    ax_info.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.65", alpha=0.96),
    )

    fig.suptitle(
        f"Injected {display.inj_nsigma_combined:.0f} sigma combined signal [{sum(d.Nsig_realized_total for d in display.displays)} realized events] at {display.mass * 1e3:.0f} MeV",
        y=0.985,
    )
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.97)
    _save_plot_outputs(fig, outpath, png_dpi=320)
    with open(os.path.splitext(outpath)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(display.to_metadata(), fh, indent=2, sort_keys=True)


def run_extraction_display_suite(config, *, dataset_key: Optional[str] = None, output_dir: Optional[str] = None) -> List[str]:
    datasets = make_datasets(config)
    if not datasets:
        raise RuntimeError("No datasets enabled for extraction display")

    masses = [float(m) for m in getattr(config, "extraction_display_masses_gev", [])]
    if not masses:
        raise RuntimeError("Config field extraction_display_masses_gev is empty")
    sigmas = [float(z) for z in getattr(config, "extraction_display_sigma_multipliers", [3.0, 5.0, 7.0])]
    if not sigmas:
        raise RuntimeError("Config field extraction_display_sigma_multipliers is empty")

    target = str(dataset_key or getattr(config, "extraction_display_dataset_key", "")).strip().lower()
    if not target:
        target = list(datasets.keys())[0] if len(datasets) == 1 else "combined"

    out_root = output_dir or os.path.join(config.output_dir, "extraction_display", target)
    ensure_dir(out_root)
    blind_shade_alpha = float(getattr(config, "extraction_display_blind_shade_alpha", 0.18))
    blind_shade_color = str(getattr(config, "extraction_display_blind_shade_color", "0.88"))
    zoom_half_sigma = float(getattr(config, "extraction_display_zoom_half_sigma", 0.5))
    base_seed = int(getattr(config, "extraction_display_seed", 271828))
    written: List[str] = []

    if target == "combined":
        requested = [str(k) for k in getattr(config, "extraction_display_dataset_keys", ["2015", "2016"])]
        ds_list = [datasets[k] for k in requested if k in datasets]
        if len(ds_list) < 2:
            raise RuntimeError("Combined extraction display needs at least two enabled datasets")
        for mass in masses:
            for z in sigmas:
                seed = _stable_display_seed(base_seed, "combined", float(mass), float(z))
                display = make_combined_extraction_display(ds_list, config, mass=float(mass), inj_nsigma=float(z), seed=int(seed))
                stem = os.path.join(out_root, f"extract_display_combined_m{int(round(float(mass) * 1e3)):03d}MeV_z{_strength_tag(z)}")
                plot_combined_extraction_display(
                    display,
                    outpath=stem + ".png",
                    blind_shade_alpha=blind_shade_alpha,
                    blind_shade_color=blind_shade_color,
                    zoom_half_sigma=zoom_half_sigma,
                )
                written.append(stem + ".png")
        return written

    if target not in datasets:
        raise RuntimeError(f"Dataset '{target}' is not enabled; enabled datasets: {sorted(datasets)}")

    ds = datasets[target]
    for mass in masses:
        for z in sigmas:
            seed = _stable_display_seed(base_seed, str(ds.key), float(mass), float(z))
            display = make_single_extraction_display(ds, config, mass=float(mass), inj_nsigma=float(z), seed=int(seed))
            stem = os.path.join(out_root, f"extract_display_{ds.key}_m{int(round(float(mass) * 1e3)):03d}MeV_z{_strength_tag(z)}")
            plot_single_extraction_display(
                display,
                outpath=stem + ".png",
                blind_shade_alpha=blind_shade_alpha,
                blind_shade_color=blind_shade_color,
                zoom_half_sigma=zoom_half_sigma,
            )
            written.append(stem + ".png")
    return written
