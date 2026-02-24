"""Plotting functions for HPS GPR analysis."""

import os
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .template import build_template
from .statistics import (
    _z_from_p_one_sided,
    _p_from_z_one_sided,
    _p_global_from_local,
    _lee_trials_from_grid,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig
    from .io import BlindPrediction


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mass_tag(mass_gev: float) -> str:
    """Create a tag string from mass value."""
    return f"m{int(round(mass_gev * 1000)):03d}MeV"


def make_mass_folder(base_dir: str, mass: float) -> str:
    """Create and return path to mass-specific output folder."""
    d = os.path.join(base_dir, mass_tag(mass))
    os.makedirs(d, exist_ok=True)
    return d


def ensure_dir(p: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(p, exist_ok=True)


def set_plot_style(style: str = "paper") -> None:
    """Set matplotlib rcParams for consistent publication-quality plots.

    Args:
        style: "paper", "talk", or "default"
    """
    style = str(style).lower().strip()
    if style == "talk":
        base_font, lw = 16, 2.0
    elif style == "paper":
        base_font, lw = 13, 1.8
    else:
        base_font, lw = 12, 1.6

    mpl.rcParams.update({
        "figure.figsize": (9.5, 5.5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font + 1,
        "legend.fontsize": base_font - 1,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": lw,
        "lines.markersize": 5,
        "axes.formatter.use_mathtext": True,
        "mathtext.default": "regular",
    })


def _grid(ax, *, which: str = "both") -> None:
    """Apply consistent minor+major grid to an axes."""
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linestyle="-")
    if which == "both":
        ax.grid(True, which="minor", alpha=0.10, linestyle=":")


def _set_title_above(ax, title: str, *, pad: float = 12.0) -> None:
    """Set axis title with extra padding to avoid clipping."""
    ax.set_title(str(title), pad=float(pad))


def _shade_blind_window(
    ax,
    blind: Tuple[float, float],
    *,
    blind_train: Optional[Tuple[float, float]] = None,
    alpha: float = 0.25,
    color: str = "0.90",
    label: Optional[str] = "blind window",
    zorder: int = 0,
) -> None:
    """Shade the extraction blind window (and optional GP training exclusion window)."""
    # Optional broader training exclusion
    if blind_train is not None:
        try:
            bt0, bt1 = float(blind_train[0]), float(blind_train[1])
            b0, b1 = float(blind[0]), float(blind[1])
            if (bt0 < b0 - 1e-15) or (bt1 > b1 + 1e-15):
                ax.axvspan(bt0, bt1, color=color, alpha=alpha * 0.5, lw=0,
                           zorder=zorder, label="GP training excluded")
        except Exception:
            pass

    # Extraction blind window
    ax.axvspan(float(blind[0]), float(blind[1]), color=color, alpha=alpha, lw=0,
               zorder=zorder, label=label)
    ax.axvline(float(blind[0]), color="0.25", lw=0.8, alpha=0.6, zorder=zorder + 1)
    ax.axvline(float(blind[1]), color="0.25", lw=0.8, alpha=0.6, zorder=zorder + 1)


def _add_info_box(
    ax,
    text: str,
    *,
    loc: str = "upper right",
    fontsize: int = 9,
    alpha: float = 0.90,
) -> None:
    """Add a framed text box with fit summary info."""
    try:
        from matplotlib.offsetbox import AnchoredText
        at = AnchoredText(str(text), loc=str(loc),
                          prop=dict(size=int(fontsize)), frameon=True, borderpad=0.45)
        at.patch.set_alpha(float(alpha))
        try:
            at.patch.set_boxstyle("round,pad=0.35")
        except Exception:
            pass
        ax.add_artist(at)
    except Exception:
        ax.text(0.98, 0.98, str(text), transform=ax.transAxes,
                va="top", ha="right", fontsize=int(fontsize),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=float(alpha)))


# ---------------------------------------------------------------------------
# Per-mass fit plots
# ---------------------------------------------------------------------------

def plot_full_range(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    outpath: str,
    title_extra: str = "",
) -> None:
    """Plot full range data vs background fit.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis
        pred: Background prediction results
        outpath: Output file path
        title_extra: Extra text for title
    """
    x = np.asarray(pred.x_full, float)
    y = np.asarray(pred.y_full, float)
    mu = np.asarray(pred.mu_full, float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(x, y, where="mid", label="Data", zorder=3)
    ax.plot(x, mu, label="GPR fit", zorder=4)
    _shade_blind_window(ax, pred.blind)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Counts / bin")
    _set_title_above(ax, f"{ds.label} — full range fit @ {mass:.4f} GeV {title_extra}")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_blind_window(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    A_show: Optional[float],
    outpath: str,
    title_extra: str = "",
) -> None:
    """Plot blind window region with optional signal overlay.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis
        pred: Background prediction results
        A_show: Signal amplitude to overlay (or None)
        outpath: Output file path
        title_extra: Extra text for title
    """
    edges = pred.edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = build_template(edges, mass, pred.sigma_val)
    y = pred.obs.astype(float)
    mu = pred.mu.astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(centers, y, where="mid", label="Data", zorder=3)
    ax.plot(centers, mu, label="Background", zorder=4)
    if A_show is not None and np.isfinite(A_show):
        ax.plot(centers, mu + float(A_show) * w,
                label=f"Bkg + signal (A={A_show:.1f})", zorder=5)
    _shade_blind_window(ax, pred.blind)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Counts / bin")
    _set_title_above(ax, f"{ds.label} — blind window @ {mass:.4f} GeV {title_extra}")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_s_over_b(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    A_show: float,
    outpath: str,
) -> None:
    """Plot signal over background ratio in the blind window.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis
        pred: Background prediction results
        A_show: Signal amplitude
        outpath: Output file path
    """
    edges = pred.edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = build_template(edges, mass, pred.sigma_val)
    s = float(A_show) * w
    b = np.clip(pred.mu.astype(float), 1e-12, None)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, s / b)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("s/b (template)")
    _set_title_above(ax, f"{ds.label} — s/b template @ {mass:.4f} GeV")
    _grid(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_scan_diagnostic_panels(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    outpath: str,
    *,
    A_up: Optional[float] = None,
    A_hat: Optional[float] = None,
    sigma_A: Optional[float] = None,
    zoom_half_sigma: float = 2.0,
) -> None:
    """Multi-panel scan diagnostic plot for a single mass hypothesis.

    Shows full range fit, blind-window zoom, and residual signal in three panels.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis
        pred: Background prediction results
        outpath: Output file path
        A_up: Upper limit on signal amplitude (for overlay)
        A_hat: Extracted signal amplitude (for overlay)
        sigma_A: Uncertainty on extracted amplitude
        zoom_half_sigma: Half-width of zoom region in units of sigma
    """
    x_full = np.asarray(pred.x_full, float)
    y_full = np.asarray(pred.y_full, float)
    mu_full = np.asarray(pred.mu_full, float)
    edges = pred.edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = build_template(edges, mass, pred.sigma_val)
    obs = pred.obs.astype(float)
    mu = pred.mu.astype(float)

    sig = float(pred.sigma_val)
    zoom_lo = mass - float(zoom_half_sigma) * sig
    zoom_hi = mass + float(zoom_half_sigma) * sig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: full range
    ax = axes[0]
    ax.step(x_full, y_full, where="mid", label="Data", zorder=3)
    ax.plot(x_full, mu_full, label="GPR fit", zorder=4)
    if A_up is not None and np.isfinite(A_up):
        mask_full = (x_full >= pred.blind[0]) & (x_full <= pred.blind[1])
        sig_vec_full = np.zeros_like(mu_full)
        sig_vec_full[mask_full] = float(A_up) * w
        ax.plot(x_full, mu_full + sig_vec_full, "--", alpha=0.7, label=f"A_up={A_up:.1f}", zorder=5)
    _shade_blind_window(ax, pred.blind)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Counts / bin")
    _set_title_above(ax, "Full range")
    ax.legend(fontsize=7)
    _grid(ax)

    # Panel 2: blind-window zoom
    ax = axes[1]
    mask = (x_full >= zoom_lo) & (x_full <= zoom_hi)
    ax.step(x_full[mask], y_full[mask], where="mid", label="Data", zorder=3)
    ax.plot(centers, mu, label="Background", zorder=4)
    if A_hat is not None and np.isfinite(A_hat):
        ax.plot(centers, mu + float(A_hat) * w, "-", label=f"Â={A_hat:.1f}", zorder=5)
    if A_up is not None and np.isfinite(A_up):
        ax.plot(centers, mu + float(A_up) * w, "--", alpha=0.7, label=f"A_up={A_up:.1f}", zorder=6)
    _shade_blind_window(ax, pred.blind)
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Counts / bin")
    _set_title_above(ax, f"Blind window zoom (±{zoom_half_sigma}σ)")
    ax.legend(fontsize=7)
    _grid(ax)

    # Panel 3: residuals
    ax = axes[2]
    resid = obs - mu
    ax.bar(centers, resid, width=np.diff(edges), align="center", alpha=0.5, label="data−bkg")
    if A_hat is not None and np.isfinite(A_hat):
        ax.plot(centers, float(A_hat) * w, label=f"Â·w", zorder=4)
    if A_up is not None and np.isfinite(A_up):
        ax.plot(centers, float(A_up) * w, "--", alpha=0.7, label=f"A_up·w", zorder=5)
    ax.axhline(0, color="k", lw=0.8)

    info = f"mass={mass:.4f} GeV\nσ={sig*1000:.1f} MeV"
    if A_hat is not None and np.isfinite(A_hat):
        info += f"\nÂ={A_hat:.1f}"
    if sigma_A is not None and np.isfinite(sigma_A):
        info += f"±{sigma_A:.1f}"
    _add_info_box(ax, info, loc="upper left", fontsize=8)

    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("data − bkg")
    _set_title_above(ax, "Residuals in blind window")
    ax.legend(fontsize=7)
    _grid(ax)

    fig.suptitle(f"{ds.label} @ {mass:.4f} GeV", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UL bands plotting
# ---------------------------------------------------------------------------

def plot_ul_bands(
    df: pd.DataFrame,
    *,
    use_eps2: bool = True,
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Plot observed + expected UL bands (±1σ, ±2σ) vs mass.

    Supports both v15 publication column names (eps2_obs, eps2_lo2, ...) and
    older internal names (ul_eps2_obs, toy_eps2_uls_q02, ...).

    Args:
        df: Bands DataFrame (from expected_ul_bands_for_dataset)
        use_eps2: If True plot epsilon^2 bands; if False plot amplitude A bands
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    if use_eps2:
        prefix, ylabel = "eps2", r"$\epsilon^2$ upper limit"
    else:
        prefix, ylabel = "A", "Amplitude upper limit (A)"

    # Try publication column names first, then legacy aliases
    def _col(key: str) -> Optional[np.ndarray]:
        pub = f"{prefix}_{key}"
        leg = {"obs": f"ul_{prefix}_obs",
               "lo2": f"toy_{prefix}_uls_q02",
               "lo1": f"toy_{prefix}_uls_q16",
               "med": f"toy_{prefix}_uls_q50",
               "hi1": f"toy_{prefix}_uls_q84",
               "hi2": f"toy_{prefix}_uls_q97"}.get(key, pub)
        if pub in df.columns:
            return df[pub].to_numpy(float)
        if leg in df.columns:
            return df[leg].to_numpy(float)
        return None

    masses = df["mass_GeV"].to_numpy(float)
    obs = _col("obs")
    lo2, lo1, med, hi1, hi2 = _col("lo2"), _col("lo1"), _col("med"), _col("hi1"), _col("hi2")

    fig, ax = plt.subplots(figsize=(10, 5))

    if lo2 is not None and hi2 is not None:
        ax.fill_between(masses, lo2, hi2, alpha=0.30, color="gold", label=r"±2σ expected")
    if lo1 is not None and hi1 is not None:
        ax.fill_between(masses, lo1, hi1, alpha=0.55, color="limegreen", label=r"±1σ expected")
    if med is not None:
        ax.plot(masses, med, "k--", lw=1.5, label="Expected median")
    if obs is not None:
        ax.plot(masses, obs, "k-", lw=2.0, label="Observed")

    ax.set_xlabel("m (GeV)")
    ax.set_ylabel(ylabel)

    # Use log y-scale when data is positive
    all_vals = np.concatenate([v for v in [obs, lo2, lo1, med, hi1, hi2] if v is not None])
    finite_pos = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if finite_pos.size > 0:
        ax.set_yscale("log")

    _set_title_above(ax, title or f"Upper limit bands ({prefix})")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_ul_pvalues(
    df: pd.DataFrame,
    *,
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Plot UL p-values vs mass (how consistent is observed with expected bands).

    Args:
        df: Bands DataFrame with p_strong, p_weak, p_two columns
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    masses = df["mass_GeV"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 4))
    for col, label, color in [
        ("p_strong", r"$p_{\rm strong}$ (obs ≤ toy)", "C0"),
        ("p_weak", r"$p_{\rm weak}$ (obs ≥ toy)", "C1"),
        ("p_two", r"$p_{\rm two}$ (2×min)", "C2"),
    ]:
        if col in df.columns:
            v = df[col].to_numpy(float)
            ax.plot(masses, v, label=label, color=color)

    ax.axhline(0.05, color="k", ls="--", lw=0.8, label="5%")
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    _set_title_above(ax, title or "UL p-values vs mass")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Significance / LEE plots
# ---------------------------------------------------------------------------

def plot_analytic_p0(
    df: pd.DataFrame,
    *,
    title: str = "",
    outpath: Optional[str] = None,
    apply_lee: bool = False,
    lee_method: str = "sidak",
    neff: Optional[float] = None,
    sigma_lines: Optional[List[float]] = None,
) -> None:
    """Plot analytic bump-hunt p0 (from profiled LRT) vs mass.

    Args:
        df: Scan results DataFrame with p0_analytic column
        title: Plot title
        outpath: Save path (if None, displays interactively)
        apply_lee: If True, compute and overlay global p0
        lee_method: "sidak" or "bonferroni"
        neff: Effective trials factor (computed from grid if None)
        sigma_lines: Z-score reference lines to draw (default [3, 5])
    """
    if sigma_lines is None:
        sigma_lines = [3, 5]

    masses = df["mass_GeV"].to_numpy(float)
    p0_local = df["p0_analytic"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masses, p0_local, label="Local p0 (analytic LRT)")

    if apply_lee:
        if neff is None:
            neff = float(len(masses))
        p0_global = np.asarray([
            _p_global_from_local(float(p), neff=neff, method=lee_method)
            for p in p0_local
        ], float)
        ax.plot(masses, p0_global, "--", label=f"Global p0 (N_eff={neff:.1f})")

    for z in (sigma_lines or []):
        p_ref = _p_from_z_one_sided(float(z))
        ax.axhline(p_ref, color="k", ls=":", lw=0.8, label=f"{z:.0f}σ")

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("p0 (one-sided)")
    _set_title_above(ax, title or "Local p0 vs mass")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_Z_local_global(
    df: pd.DataFrame,
    *,
    title: str = "",
    outpath: Optional[str] = None,
    apply_lee: bool = False,
    lee_method: str = "sidak",
    neff: Optional[float] = None,
    z_lines: Optional[List[float]] = None,
) -> None:
    """Plot local Z (and optionally global Z with LEE correction) vs mass.

    Args:
        df: Scan results DataFrame with Z_analytic column (or p0_analytic)
        title: Plot title
        outpath: Save path (if None, displays interactively)
        apply_lee: If True, compute and overlay global Z
        lee_method: "sidak" or "bonferroni" for global p-value
        neff: Effective trials factor (number of independent tests)
        z_lines: Reference Z-score lines to draw (default [3, 5])
    """
    if z_lines is None:
        z_lines = [3.0, 5.0]

    masses = df["mass_GeV"].to_numpy(float)

    # Get local Z
    if "Z_analytic" in df.columns:
        Z_local = df["Z_analytic"].to_numpy(float)
    elif "p0_analytic" in df.columns:
        Z_local = np.asarray(
            [_z_from_p_one_sided(float(p)) for p in df["p0_analytic"].to_numpy(float)],
            float,
        )
    else:
        raise ValueError("DataFrame must have 'Z_analytic' or 'p0_analytic' column")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masses, Z_local, label="Local Z")

    if apply_lee:
        if neff is None:
            neff = float(len(masses))
        p_local = np.asarray([_p_from_z_one_sided(float(z)) for z in Z_local], float)
        p_global = np.asarray([
            _p_global_from_local(float(p), neff=neff, method=lee_method)
            for p in p_local
        ], float)
        Z_global = np.asarray([_z_from_p_one_sided(float(p)) for p in p_global], float)
        ax.plot(masses, Z_global, "--", label=f"Global Z (N_eff={neff:.1f})")
        _add_info_box(ax, f"N_eff = {neff:.1f}\nmethod: {lee_method}", loc="upper right")

    for z in (z_lines or []):
        ax.axhline(float(z), color="k", ls=":", lw=0.8, label=f"{z:.0f}σ")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Z (Gaussian significance)")
    _set_title_above(ax, title or "Local/global significance vs mass")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Injection study plots
# ---------------------------------------------------------------------------

def plot_linearity(
    df_sum: pd.DataFrame,
    *,
    xvar: str = "strength",
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Linearity plot: mean extracted Â vs injected signal strength.

    Args:
        df_sum: Summary DataFrame from summarize_injection_grid()
        xvar: x-axis variable ("strength" or "inj_nsigma")
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, sub in df_sum.groupby("dataset"):
        x = sub[xvar].to_numpy(float)
        y = sub["A_hat_mean"].to_numpy(float)
        ax.plot(x, y, "o-", label=str(ds))
    # Identity reference
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, "k--", lw=0.8, label="ideal")
    ax.set_xlim(xlim)
    ax.set_xlabel(xvar)
    ax.set_ylabel(r"$\langle\hat{A}\rangle$")
    _set_title_above(ax, title or "Linearity: mean extracted vs injected")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_bias_vs_injected_strength(
    df_sum: pd.DataFrame,
    *,
    xvar: str = "strength",
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Bias plot: <Â> − A_inj vs injected signal strength.

    Args:
        df_sum: Summary DataFrame from summarize_injection_grid()
        xvar: x-axis variable
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, sub in df_sum.groupby("dataset"):
        x = sub[xvar].to_numpy(float)
        bias = sub["A_hat_mean"].to_numpy(float) - sub["strength"].to_numpy(float)
        ax.plot(x, bias, "o-", label=str(ds))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(xvar)
    ax.set_ylabel(r"$\langle\hat{A}\rangle - A_{\rm inj}$")
    _set_title_above(ax, title or "Extraction bias vs injected strength")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_pull_width(
    df_sum: pd.DataFrame,
    *,
    xvar: str = "strength",
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Pull width: std(pull) vs injected signal strength.

    Args:
        df_sum: Summary DataFrame from summarize_injection_grid()
        xvar: x-axis variable
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, sub in df_sum.groupby("dataset"):
        x = sub[xvar].to_numpy(float)
        y = sub["pull_std"].to_numpy(float)
        ax.plot(x, y, "o-", label=str(ds))
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="ideal (σ=1)")
    ax.set_xlabel(xvar)
    ax.set_ylabel(r"std$((\hat{A}-A_{\rm inj})/\sigma_A)$")
    _set_title_above(ax, title or "Pull width vs injected strength")
    ax.legend(loc="best")
    _grid(ax)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_coverage(
    df_sum: pd.DataFrame,
    *,
    xvar: str = "strength",
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Coverage plot: fraction |pull| < 1σ and |pull| < 2σ vs injected strength.

    Args:
        df_sum: Summary DataFrame from summarize_injection_grid()
        xvar: x-axis variable
        title: Plot title
        outpath: Save path (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, sub in df_sum.groupby("dataset"):
        x = sub[xvar].to_numpy(float)
        ax.plot(x, sub["cov_1sigma"].to_numpy(float), "o-", label=f"{ds} 1σ")
        ax.plot(x, sub["cov_2sigma"].to_numpy(float), "s--", label=f"{ds} 2σ")
    ax.axhline(0.683, color="k", ls=":", lw=0.8, label="68.3%")
    ax.axhline(0.954, color="gray", ls=":", lw=0.8, label="95.4%")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xvar)
    ax.set_ylabel("Coverage fraction")
    _set_title_above(ax, title or "Extraction coverage vs injected strength")
    ax.legend(loc="best", fontsize=8)
    _grid(ax)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary plots (UL curves)
# ---------------------------------------------------------------------------

def plot_eps2_curves(
    df_single: pd.DataFrame,
    df_comb: pd.DataFrame,
    outdir: str,
) -> None:
    """Plot epsilon^2 upper limit curves (individual + combined).

    Args:
        df_single: Single-dataset results DataFrame
        df_comb: Combined results DataFrame
        outdir: Output directory
    """
    ensure_dir(outdir)

    for ds in sorted(df_single["dataset"].unique()):
        sub = df_single[df_single["dataset"] == ds].copy()
        sub = sub[np.isfinite(sub["eps2_up"].to_numpy(float))]
        if len(sub) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(sub["mass_GeV"], sub["eps2_up"])
        ax.set_yscale("log")
        ax.set_xlabel("m (GeV)")
        ax.set_ylabel(r"$\epsilon^2$ upper limit")
        _set_title_above(ax, f"{ds}: epsilon^2 UL (95% CL)")
        _grid(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"eps2_ul_{ds}.png"), dpi=180)
        plt.close(fig)

    subc = df_comb.copy()
    subc = subc[np.isfinite(subc["eps2_up"].to_numpy(float))] if "eps2_up" in subc.columns else subc
    if len(subc) > 0 and "eps2_up" in subc.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        for ds in sorted(df_single["dataset"].unique()):
            sub = df_single[df_single["dataset"] == ds].copy()
            sub = sub[np.isfinite(sub["eps2_up"].to_numpy(float))]
            if len(sub):
                ax.plot(sub["mass_GeV"], sub["eps2_up"], label=ds)
        ax.plot(subc["mass_GeV"], subc["eps2_up"], label="combined", lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("m (GeV)")
        ax.set_ylabel(r"$\epsilon^2$ upper limit")
        _set_title_above(ax, "epsilon^2 UL: individual + combined")
        ax.legend(loc="best")
        _grid(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "eps2_ul_overlay.png"), dpi=180)
        plt.close(fig)

    print("Wrote summary plots to", outdir)


def plot_bands(
    df_bands: pd.DataFrame,
    outpath: str,
    column_prefix: str = "A",
    ylabel: str = "A UL",
    title: str = "Expected UL bands",
) -> None:
    """Plot expected upper limit bands (legacy wrapper for plot_ul_bands).

    Args:
        df_bands: DataFrame with band information
        outpath: Output file path
        column_prefix: Prefix for column names ("A" or "eps2")
        ylabel: Y-axis label
        title: Plot title
    """
    use_eps2 = (column_prefix.lower() == "eps2")
    plot_ul_bands(df_bands, use_eps2=use_eps2, title=title, outpath=outpath)
