"""Plotting functions for HPS GPR analysis."""

import os
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib as mpl

# Force non-interactive backend so plotting works reliably in batch/headless jobs.
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from scipy import stats

from .template import build_template
from .statistics import (
    _z_from_p_one_sided,
    _p_from_z_one_sided,
    _p_global_from_local,
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


def _savefig_png_pdf(fig: plt.Figure, outpath: str, *, dpi: int = 180) -> None:
    """Save a figure as both PNG and PDF using outpath stem."""
    root, ext = os.path.splitext(outpath)
    if ext.lower() in {".png", ".pdf"}:
        base = root
    else:
        base = outpath
    fig.savefig(f"{base}.png", dpi=dpi)
    fig.savefig(f"{base}.pdf", dpi=dpi)


def _mass_style_map(masses: np.ndarray) -> Tuple[dict, dict]:
    """Build stable color/marker maps for a sorted mass list."""
    mvals = np.sort(np.unique(np.asarray(masses, float)))
    cmap = plt.get_cmap("viridis")
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    n = max(1, len(mvals) - 1)
    c_map = {m: cmap(i / n) for i, m in enumerate(mvals)}
    m_map = {m: markers[i % len(markers)] for i, m in enumerate(mvals)}
    return c_map, m_map


def _dataset_axes(df_sum: pd.DataFrame, *, figsize_per_row: float = 3.4) -> Tuple[plt.Figure, np.ndarray, List[str]]:
    """Create one axis per dataset (faceted overlay)."""
    datasets = sorted(df_sum["dataset"].astype(str).unique())
    nrows = max(1, len(datasets))
    fig, axs = plt.subplots(nrows, 1, figsize=(8.8, figsize_per_row * nrows), sharex=True)
    axs = np.atleast_1d(axs)
    return fig, axs, datasets


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
    A_show: Optional[float] = None,
) -> None:
    """Plot full range data vs background fit.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis
        pred: Background prediction results
        outpath: Output file path
        title_extra: Extra text for title
        A_show: Optional signal amplitude to overlay
    """
    x = np.asarray(pred.x_full, float)
    y = np.asarray(pred.y_full, float)
    mu = np.asarray(pred.mu_full, float)

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    yerr = np.sqrt(np.clip(y, 1.0, None))
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=3.0, lw=1.0, color="black", label="Data", zorder=3)
    ax.plot(x, mu, color="C0", label="GPR mean", zorder=4)

    if A_show is not None and np.isfinite(A_show):
        try:
            w_full = build_template(np.asarray(pred.edges_full, float), mass, pred.sigma_val)
            if w_full.size == mu.size:
                ax.plot(x, mu + float(A_show) * w_full, "--", color="C3", lw=1.7,
                        label=rf"GPR + $A w$ ($A={float(A_show):.2g}$)", zorder=5)
        except Exception:
            pass

    _shade_blind_window(ax, pred.blind, blind_train=getattr(pred, "blind_train", None))
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("counts / bin")
    _set_title_above(ax, f"{ds.label} — full range @ m={mass*1000:.1f} MeV {title_extra}".strip())
    ax.legend(loc="best", frameon=True)
    _grid(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_blind_window(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    outpath: str,
    *,
    A_up: Optional[float] = None,
    A_hat: Optional[float] = None,

    A_show: Optional[float] = None,
    title_extra: str = "",
    zoom_half_sigma: float = 0.5,
) -> None:
    """Plot blind-window region with Ahat and UL overlays.

    Args:
        A_show: Backward-compatible alias for A_up.
    """
    if A_up is None and A_show is not None:
        A_up = A_show
    x_full = np.asarray(pred.x_full, float)
    y_full = np.asarray(pred.y_full, float)
    mu_full = np.asarray(pred.mu_full, float)
    edges_full = np.asarray(pred.edges_full, float)


    zlo = float(pred.blind[0] - float(zoom_half_sigma) * pred.sigma_val)
    zhi = float(pred.blind[1] + float(zoom_half_sigma) * pred.sigma_val)
    m_zoom = (x_full >= zlo) & (x_full <= zhi)

    w_full = build_template(edges_full, mass, pred.sigma_val)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    yz = y_full[m_zoom]
    ax.errorbar(
        x_full[m_zoom], yz, yerr=np.sqrt(np.clip(yz, 1.0, None)),
        fmt="o", ms=2.8, lw=0.9, color="black", label="Data", zorder=3,
    )
    ax.plot(x_full[m_zoom], mu_full[m_zoom], color="C0", lw=1.5, label="GPR mean", zorder=4)

    if A_hat is not None and np.isfinite(A_hat):
        ax.plot(
            x_full[m_zoom], (mu_full + float(A_hat) * w_full)[m_zoom],
            color="C2", lw=1.4, label=rf"$\mu + \hat{{A}}w$ ($\hat{{A}}={float(A_hat):.2g}$)", zorder=5,
        )
    if A_up is not None and np.isfinite(A_up):
        ax.plot(
            x_full[m_zoom], (mu_full + float(A_up) * w_full)[m_zoom],
            "--", color="C3", lw=1.4, label=rf"$\mu + A_{{up}}w$ ($A_{{up}}={float(A_up):.2g}$)", zorder=6,
        )

    _shade_blind_window(ax, pred.blind, blind_train=getattr(pred, "blind_train", None))
    ax.set_xlim(zlo, zhi)
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("counts / bin")

    title = f"{ds.label}: blind-window fit @ m={mass*1001:.1f} MeV"
    if title_extra:
        title += f" {title_extra}"
    ax.set_title(title, fontsize=10, pad=6.0, loc="center")
    ax.legend(loc="best", fontsize=7, frameon=True)














































    _grid(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
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

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.step(centers, s / b, where="mid", color="C3", lw=1.8)
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("s/b per bin")
    _set_title_above(ax, f"{ds.label} — s/b in blind window @ m={mass*1000:.1f} MeV")
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
        prefix, ylabel = "A", "Signal-yield upper limit (95% CL)"

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


def plot_observed_ul_only(
    df: pd.DataFrame,
    *,
    y: str = "eps2",
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Plot only observed upper limit vs mass for either signal yield or epsilon^2."""
    y = str(y).strip().lower()
    if y == "eps2":
        col, ylabel = "eps2_obs", r"Observed 95% CL upper limit on $\epsilon^2$"
        legacy = "ul_eps2_obs"
    elif y in {"yield", "signal_yield", "a"}:
        col, ylabel = "A_obs", "Observed 95% CL upper limit on signal yield"
        legacy = "ul_A_obs"
    else:
        raise ValueError("y must be 'eps2' or 'yield'")

    if col not in df.columns and legacy in df.columns:
        col = legacy
    if col not in df.columns:
        return

    masses = df["mass_GeV"].to_numpy(float)
    obs = df[col].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masses, obs, "k-", lw=2.0)
    finite_pos = obs[np.isfinite(obs) & (obs > 0)]
    if finite_pos.size > 0:
        ax.set_yscale("log")
    ax.set_xlabel("Mass hypothesis m (GeV)")
    ax.set_ylabel(ylabel)
    _set_title_above(ax, title or f"{ylabel} vs mass")
    _grid(ax)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def plot_ul_pvalue_components(
    df: pd.DataFrame,
    *,
    title: str = "",
    outpath: Optional[str] = None,
    neff: Optional[float] = None,
    lee_method: str = "sidak",
) -> None:
    """Overlay p_strong, p_weak, p_two with local/global sigma-reference p-values."""
    if not {"p_strong", "p_weak", "p_two"}.intersection(set(df.columns)):
        return

    masses = df["mass_GeV"].to_numpy(float)
    if neff is None:
        # For UL-tail diagnostics we use the tested-grid count as a robust default.
        neff = max(1.0, float(len(masses)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = []
    for col, label, color in [
        ("p_strong", r"$p_{\rm strong}$", "C0"),
        ("p_weak", r"$p_{\rm weak}$", "C1"),
        ("p_two", r"$p_{\rm two}$", "C2"),
    ]:
        if col in df.columns:
            v = np.clip(df[col].to_numpy(float), 1e-300, 1.0)
            vals.append(v)
            ax.plot(masses, v, label=label, color=color)

    y_all = np.concatenate(vals) if vals else np.array([1.0])
    ymin_data = np.nanmin(y_all[np.isfinite(y_all)]) if np.isfinite(y_all).any() else 1e-3
    ymin_plot = max(1e-6, min(0.8 * ymin_data, 0.2))

    for z in [1.0, 2.0, 3.0]:
        p_local = _p_from_z_one_sided(z)
        p_global = _p_global_from_local(p_local, Neff=neff, method=lee_method)
        if z <= 2.0 or ymin_plot <= p_local * 1.2:
            ax.axhline(p_local, color="0.35", ls=":", lw=0.9)
            ax.text(masses.max(), p_local, f" local {int(z)}σ", va="bottom", ha="right", fontsize=8)
        if z <= 2.0 or ymin_plot <= p_global * 1.2:
            ax.axhline(p_global, color="0.15", ls="--", lw=0.9)
            ax.text(masses.min(), p_global, f"global {int(z)}σ ", va="bottom", ha="left", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylim(min(1.0, max(ymin_plot, 1e-6)), 1.0)
    ax.set_xlabel("Mass hypothesis m (GeV)")
    ax.set_ylabel("p-value")
    _set_title_above(ax, title or "UL-tail p-value components with local/global references")
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
    indep_width_sigma: float = 1.96,
    sigma_col: str = "sigma_mass_res_GeV",
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
        sigma_lines = [3.0]

    masses = df["mass_GeV"].to_numpy(float)
    p0_local = np.clip(df["p0_analytic"].to_numpy(float), 1e-300, 1.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masses, p0_local, label="Local p0 (analytic LRT)")

    p0_global = None
    if apply_lee:
        if neff is None:
            sig = df[sigma_col].to_numpy(float) if sigma_col in df.columns else None
            if sig is not None and np.isfinite(sig).any():
                dm = np.diff(masses)
                if dm.size > 0:
                    sig_mid = 0.5 * (sig[:-1] + sig[1:])
                    ok = np.isfinite(sig_mid) & (sig_mid > 0) & np.isfinite(dm) & (dm > 0)
                    neff = float(np.clip(np.sum(dm[ok] / (float(indep_width_sigma) * sig_mid[ok])), 1.0, float(len(masses))))
                else:
                    neff = 1.0
            else:
                neff = float(len(masses))
        p0_global = np.asarray([
            _p_global_from_local(float(p), Neff=neff, method=lee_method)
            for p in p0_local
        ], float)
        p0_global = np.clip(p0_global, 1e-300, 1.0)
        ax.plot(masses, p0_global, "--", label=f"Global p0 (N_eff={neff:.1f})")

    z_local = np.asarray([_z_from_p_one_sided(float(p)) for p in p0_local], float)
    z_global = np.asarray([_z_from_p_one_sided(float(p)) for p in p0_global], float) if p0_global is not None else np.array([0.0])
    z_peak = float(np.nanmax(np.concatenate([z_local, z_global]))) if (z_local.size or z_global.size) else 0.0
    z_floor = max(3.0, z_peak)

    sig_ref = sorted(set(float(z) for z in (sigma_lines or [3.0]) if float(z) <= z_floor + 1e-12))
    for z in sig_ref:
        p_ref = _p_from_z_one_sided(float(z))
        ax.axhline(p_ref, color="k", ls=":", lw=0.8, label=f"{z:.0f}σ")

    ax.set_yscale("log")
    p_floor = max(_p_from_z_one_sided(z_floor), 1e-300)
    ax.set_ylim(p_floor, 1.0)
    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("p-value (one-sided)")
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
    indep_width_sigma: float = 1.96,
    sigma_col: str = "sigma_mass_res_GeV",
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

    Z_local = np.clip(np.asarray(Z_local, float), 0.0, None)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masses, Z_local, label="Local Z")

    if apply_lee:
        if neff is None:
            sig = df[sigma_col].to_numpy(float) if sigma_col in df.columns else None
            if sig is not None and np.isfinite(sig).any():
                dm = np.diff(masses)
                if dm.size > 0:
                    sig_mid = 0.5 * (sig[:-1] + sig[1:])
                    ok = np.isfinite(sig_mid) & (sig_mid > 0) & np.isfinite(dm) & (dm > 0)
                    neff = float(np.clip(np.sum(dm[ok] / (float(indep_width_sigma) * sig_mid[ok])), 1.0, float(len(masses))))
                else:
                    neff = 1.0
            else:
                neff = float(len(masses))
        p_local = np.asarray([_p_from_z_one_sided(float(z)) for z in Z_local], float)
        p_global = np.asarray([
            _p_global_from_local(float(p), Neff=neff, method=lee_method)
            for p in p_local
        ], float)
        Z_global = np.clip(np.asarray([_z_from_p_one_sided(float(p)) for p in p_global], float), 0.0, None)
        ax.plot(masses, Z_global, "--", label=f"Global Z (N_eff={neff:.1f})")
        _add_info_box(ax, f"N_eff = {neff:.1f}\nmethod: {lee_method}", loc="upper right")

    for z in (z_lines or []):
        ax.axhline(float(z), color="k", ls=":", lw=0.8, label=f"{z:.0f}σ")

    ax.axhline(0, color="k", lw=0.5)
    zmax = max(3.2, float(np.nanmax(Z_local)) * 1.15 if np.isfinite(np.nanmax(Z_local)) else 3.2)
    ax.set_ylim(0.0, zmax)
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
    fig, axs, datasets = _dataset_axes(df_sum)
    c_map, m_map = _mass_style_map(df_sum["mass_GeV"].to_numpy(float))
    for i, ds in enumerate(datasets):
        ax = axs[i]
        sub_ds = df_sum[df_sum["dataset"].astype(str) == ds].copy()
        for mass, sub_m in sub_ds.groupby("mass_GeV"):
            sub_m = sub_m.sort_values(xvar)
            x = sub_m[xvar].to_numpy(float)
            y = sub_m["A_hat_mean"].to_numpy(float)
            xerr = sub_m["inj_nsigma_xerr"].to_numpy(float) if (xvar == "inj_nsigma" and "inj_nsigma_xerr" in sub_m.columns) else None
            yerr = sub_m["sigma_A_mean"].to_numpy(float) / np.sqrt(np.clip(sub_m.get("n_toys", pd.Series(np.ones(len(sub_m)))).to_numpy(float), 1.0, None)) if "sigma_A_mean" in sub_m.columns else None
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt=f"{m_map[mass]}-", color=c_map[mass], lw=1.6, ms=5.0, capsize=2,
                label=f"m={mass * 1e3:.0f} MeV",
            )
        xlim = ax.get_xlim()
        ax.plot(xlim, xlim, "k--", lw=1.2, label="ideal")
        ax.set_xlim(xlim)
        ax.set_ylabel(r"$\langle\hat{A}\rangle$")
        _set_title_above(ax, f"{ds}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        _grid(ax)
    axs[-1].set_xlabel(xvar)
    fig.suptitle(title or "Linearity: mean extracted vs injected", y=1.01)
    plt.tight_layout()
    if outpath:
        _savefig_png_pdf(fig, outpath, dpi=180)
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
    fig, axs, datasets = _dataset_axes(df_sum)
    c_map, m_map = _mass_style_map(df_sum["mass_GeV"].to_numpy(float))
    for i, ds in enumerate(datasets):
        ax = axs[i]
        sub_ds = df_sum[df_sum["dataset"].astype(str) == ds].copy()
        for mass, sub_m in sub_ds.groupby("mass_GeV"):
            sub_m = sub_m.sort_values(xvar)
            x = sub_m[xvar].to_numpy(float)
            bias = sub_m["A_hat_mean"].to_numpy(float) - sub_m["strength"].to_numpy(float)
            xerr = sub_m["inj_nsigma_xerr"].to_numpy(float) if (xvar == "inj_nsigma" and "inj_nsigma_xerr" in sub_m.columns) else None
            yerr = sub_m["sigma_A_mean"].to_numpy(float) / np.sqrt(np.clip(sub_m.get("n_toys", pd.Series(np.ones(len(sub_m)))).to_numpy(float), 1.0, None)) if "sigma_A_mean" in sub_m.columns else None
            ax.errorbar(
                x, bias, xerr=xerr, yerr=yerr,
                fmt=f"{m_map[mass]}-", color=c_map[mass], lw=1.6, ms=5.0, capsize=2,
                label=f"m={mass * 1e3:.0f} MeV",
            )
        ax.axhline(0, color="k", lw=1.0)
        ax.set_ylabel(r"$\langle\hat{A}\rangle - A_{\rm inj}$")
        _set_title_above(ax, f"{ds}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        _grid(ax)
    axs[-1].set_xlabel(xvar)
    fig.suptitle(title or "Extraction bias vs injected strength", y=1.01)
    plt.tight_layout()
    if outpath:
        _savefig_png_pdf(fig, outpath, dpi=180)
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
    fig, axs, datasets = _dataset_axes(df_sum)
    c_map, m_map = _mass_style_map(df_sum["mass_GeV"].to_numpy(float))
    for i, ds in enumerate(datasets):
        ax = axs[i]
        sub_ds = df_sum[df_sum["dataset"].astype(str) == ds].copy()
        for mass, sub_m in sub_ds.groupby("mass_GeV"):
            sub_m = sub_m.sort_values(xvar)
            x = sub_m[xvar].to_numpy(float)
            y = sub_m["pull_std"].to_numpy(float)
            ax.plot(
                x, y, f"{m_map[mass]}-", color=c_map[mass], lw=1.6, ms=5.0,
                label=f"m={mass * 1e3:.0f} MeV",
            )
        ax.axhline(1.0, color="k", ls="--", lw=1.0, label="ideal (σ=1)")
        ax.set_ylabel(r"std$((\hat{A}-A_{\rm inj})/\sigma_A)$")
        _set_title_above(ax, f"{ds}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        _grid(ax)
    axs[-1].set_xlabel(xvar)
    fig.suptitle(title or "Pull width vs injected strength", y=1.01)
    plt.tight_layout()
    if outpath:
        _savefig_png_pdf(fig, outpath, dpi=180)
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
    fig, axs, datasets = _dataset_axes(df_sum)
    c_map, m_map = _mass_style_map(df_sum["mass_GeV"].to_numpy(float))
    for i, ds in enumerate(datasets):
        ax = axs[i]
        sub_ds = df_sum[df_sum["dataset"].astype(str) == ds].copy()
        for mass, sub_m in sub_ds.groupby("mass_GeV"):
            sub_m = sub_m.sort_values(xvar)
            x = sub_m[xvar].to_numpy(float)
            ax.plot(
                x, sub_m["cov_1sigma"].to_numpy(float), f"{m_map[mass]}-", color=c_map[mass],
                lw=1.6, ms=5.0, label=f"m={mass * 1e3:.0f} MeV (1σ)",
            )
            ax.plot(
                x, sub_m["cov_2sigma"].to_numpy(float), f"{m_map[mass]}--", color=c_map[mass],
                lw=1.4, ms=5.0, alpha=0.9, label=f"m={mass * 1e3:.0f} MeV (2σ)",
            )
        ax.axhline(0.683, color="k", ls=":", lw=1.0, label="68.3%")
        ax.axhline(0.954, color="gray", ls=":", lw=1.0, label="95.4%")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage fraction")
        _set_title_above(ax, f"{ds}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
        _grid(ax)
    axs[-1].set_xlabel(xvar)
    fig.suptitle(title or "Extraction coverage vs injected strength", y=1.01)
    plt.tight_layout()
    if outpath:
        _savefig_png_pdf(fig, outpath, dpi=180)
        plt.close(fig)
    else:
        plt.show()



def plot_injection_heatmap(
    df_sum: pd.DataFrame,
    *,
    value_col: str = "pull_mean",
    title: str = "",
    outpath: Optional[str] = None,
    dataset_filter: Optional[str] = None,
) -> None:
    """Heatmap for injection/extraction summary over (mass, injected strength)."""
    if df_sum.empty or value_col not in df_sum.columns:
        return
    xcol = "mass_GeV" if "mass_GeV" in df_sum.columns else "mass"
    ycol = "inj_nsigma" if "inj_nsigma" in df_sum.columns else "strength"
    groups = df_sum.groupby("dataset")
    for ds, sub in groups:
        if dataset_filter is not None and str(ds) != str(dataset_filter):
            continue
        piv = sub.pivot_table(index=ycol, columns=xcol, values=value_col, aggfunc="mean")
        piv = piv.sort_index(axis=0).sort_index(axis=1)
        if piv.empty:
            continue
        z = piv.to_numpy(float)
        z_masked = np.ma.array(z, mask=~np.isfinite(z))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        im = ax.imshow(z_masked, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([f"{v:.2g}" for v in piv.index.to_numpy(float)])
        ax.set_xticks(np.arange(len(piv.columns))[::max(1, len(piv.columns)//8)])
        cols = piv.columns.to_numpy(float)
        idx = np.arange(len(cols))[::max(1, len(cols)//8)]
        ax.set_xticklabels([f"{cols[i]*1e3:.0f}" for i in idx])
        ax.set_xlabel("Mass hypothesis [MeV]")
        ax.set_ylabel("Injected strength [sigma_A]")
        _set_title_above(ax, title or f"{ds}: injection/extraction heatmap ({value_col})")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(value_col)
        plt.tight_layout()
        if outpath:
            if dataset_filter is not None:
                pth = outpath
            else:
                root, ext = os.path.splitext(outpath)
                pth = f"{root}_{ds}{ext or '.png'}"
            plt.savefig(pth, dpi=200)
            plt.close(fig)
        else:
            plt.show()


def _normality_pvalue(
    pull_vals: np.ndarray,
    *,
    method: str = "ks",
) -> float:
    """Compute a p-value for consistency with N(0,1)."""
    vals = np.asarray(pull_vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return float("nan")
    m = str(method).lower().strip()
    try:
        if m == "ad":
            res = stats.anderson(vals, dist="norm")
            z = np.asarray(res.significance_level, float)
            crit = np.asarray(res.critical_values, float)
            order = np.argsort(crit)
            z = z[order]
            crit = crit[order]
            p = np.interp(float(res.statistic), crit, z / 100.0, left=0.25, right=0.001)
            return float(np.clip(p, 0.0, 1.0))
        # KS test against standard normal
        return float(stats.kstest(vals, "norm", args=(0.0, 1.0)).pvalue)
    except Exception:
        return float("nan")


def plot_pull_histogram_by_mass(
    df_toys: pd.DataFrame,
    *,
    dataset_key: Optional[str] = None,
    group_by_strength: bool = True,
    bins: int = 30,
    pvalue_method: Optional[str] = "ks",
    title_prefix: str = "",
    outdir: Optional[str] = None,
) -> List[str]:
    """Plot toy pull histograms per mass (and optional strength), with N(0,1) reference."""
    if df_toys.empty or "pull_param" not in df_toys.columns or "mass_GeV" not in df_toys.columns:
        return []

    dft = df_toys.copy()
    if dataset_key is not None and "dataset" in dft.columns:
        dft = dft[dft["dataset"].astype(str) == str(dataset_key)].copy()
    if dft.empty:
        return []

    ensure_dir(outdir or ".")
    paths: List[str] = []
    grp_cols = ["mass_GeV"] + (["strength"] if group_by_strength and "strength" in dft.columns else [])
    base = str(dataset_key) if dataset_key is not None else "all"

    for keys, sub in dft.groupby(grp_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mass = float(keys[0])
        strength = float(keys[1]) if len(keys) > 1 else None
        pull = sub["pull_param"].to_numpy(float)
        pull = pull[np.isfinite(pull)]
        if pull.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        n, edges, _ = ax.hist(pull, bins=int(bins), density=True, alpha=0.65, color="C0", label="toys")
        xlo = float(min(np.nanmin(pull), -4.0))
        xhi = float(max(np.nanmax(pull), 4.0))
        xg = np.linspace(xlo, xhi, 500)
        ax.plot(xg, stats.norm.pdf(xg, loc=0.0, scale=1.0), "k--", lw=1.3, label=r"$\mathcal{N}(0,1)$")

        m = float(np.nanmean(pull))
        w = float(np.nanstd(pull, ddof=1)) if pull.size > 1 else float("nan")
        ptxt = ""
        if pvalue_method:
            pval = _normality_pvalue(pull, method=str(pvalue_method))
            if np.isfinite(pval):
                ptxt = f"\n{str(pvalue_method).upper()} p={pval:.3g}"
        ax.text(
            0.98,
            0.97,
            f"N={pull.size}\nmean={m:.3f}\nwidth={w:.3f}{ptxt}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
        )
        ax.set_xlabel(r"pull = $(\hat{A}-A_{inj})/\sigma_A$")
        ax.set_ylabel("density")
        t_bits = [f"m={mass*1e3:.1f} MeV"]
        if strength is not None:
            t_bits.append(f"A_inj={strength:.3g}")
        _set_title_above(ax, f"{title_prefix}{base}: pull histogram ({', '.join(t_bits)})".strip())
        ax.legend(loc="best", frameon=True)
        _grid(ax)
        plt.tight_layout()

        mtag = mass_tag(mass)
        stag = f"_A{strength:.3g}".replace("+", "").replace("-", "m").replace(".", "p") if strength is not None else ""
        outpath = os.path.join(outdir or ".", f"pull_hist_{base}_{mtag}{stag}.png")
        plt.savefig(outpath, dpi=220)
        plt.close(fig)
        paths.append(outpath)
    return paths


def plot_pull_vs_mass(
    df_toys: pd.DataFrame,
    *,
    dataset_key: Optional[str] = None,
    title: str = "",
    outpath: Optional[str] = None,
) -> None:
    """Plot pull mean and width versus mass for each injected strength."""
    if df_toys.empty or "pull_param" not in df_toys.columns:
        return
    req = {"mass_GeV", "strength", "pull_param"}
    if not req.issubset(df_toys.columns):
        return

    dft = df_toys.copy()
    if dataset_key is not None and "dataset" in dft.columns:
        dft = dft[dft["dataset"].astype(str) == str(dataset_key)].copy()
    if dft.empty:
        return

    rows = []
    for (m, s), sub in dft.groupby(["mass_GeV", "strength"], dropna=False):
        pull = sub["pull_param"].to_numpy(float)
        pull = pull[np.isfinite(pull)]
        if pull.size == 0:
            continue
        mu = float(np.mean(pull))
        sd = float(np.std(pull, ddof=1)) if pull.size > 1 else float("nan")
        sem = float(sd / np.sqrt(pull.size)) if pull.size > 1 else float("nan")
        sd_err = float(sd / np.sqrt(2 * (pull.size - 1))) if pull.size > 2 and np.isfinite(sd) else float("nan")
        rows.append(dict(mass_GeV=float(m), strength=float(s), pull_mean=mu, pull_sem=sem, pull_std=sd, pull_std_err=sd_err))

    if not rows:
        return
    dsum = pd.DataFrame(rows).sort_values(["strength", "mass_GeV"]).reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, gridspec_kw={"hspace": 0.08})
    for strength, sub in dsum.groupby("strength"):
        x = sub["mass_GeV"].to_numpy(float) * 1e3
        y = sub["pull_mean"].to_numpy(float)
        yerr = sub["pull_sem"].to_numpy(float)
        ax0.plot(x, y, "o-", label=rf"$A_{{inj}}={float(strength):.3g}$")
        if np.any(np.isfinite(yerr)):
            ax0.fill_between(x, y - yerr, y + yerr, alpha=0.20)

        w = sub["pull_std"].to_numpy(float)
        werr = sub["pull_std_err"].to_numpy(float)
        ax1.plot(x, w, "o-")
        if np.any(np.isfinite(werr)):
            ax1.fill_between(x, w - werr, w + werr, alpha=0.20)

    ax0.axhline(0.0, color="k", lw=0.8, ls="--")
    ax1.axhline(1.0, color="k", lw=0.8, ls="--")
    ax0.set_ylabel("pull mean")
    ax1.set_ylabel("pull width")
    ax1.set_xlabel("mass hypothesis [MeV]")
    _set_title_above(ax0, title or f"{dataset_key or 'all'}: pull moments vs mass")
    ax0.legend(loc="best", ncol=2)
    _grid(ax0)
    _grid(ax1)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=220)
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
    """Write full scan summary plots (UL, Ahat, p0, Z, and GP hyperparameters)."""
    ensure_dir(outdir)

    datasets = sorted(df_single["dataset"].unique()) if len(df_single) and "dataset" in df_single.columns else []

    def _finite(sub: pd.DataFrame, col: str) -> pd.DataFrame:
        return sub[np.isfinite(sub[col].to_numpy(float))].copy() if col in sub.columns else sub.iloc[0:0].copy()

    for ds in datasets:
        sub = df_single[df_single["dataset"] == ds].copy().sort_values("mass_GeV")
        if len(sub) == 0:
            continue

        # A_up summary
        sA = _finite(sub, "A_up")
        if len(sA):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(sA["mass_GeV"], sA["A_up"], color="C0")
            ax.set_yscale("log")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel(r"$A_{up}$")
            _set_title_above(ax, f"{ds}: signal-yield UL (95% CL)")
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"A_up_{ds}.png"), dpi=200)
            plt.close(fig)

        # eps2 UL summary
        se = _finite(sub, "eps2_up")
        if len(se):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(se["mass_GeV"], se["eps2_up"], color="C1")
            ax.set_yscale("log")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel(r"$\epsilon^2_{up}$")
            _set_title_above(ax, f"{ds}: epsilon^2 UL (95% CL)")
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"eps2_ul_{ds}.png"), dpi=200)
            plt.close(fig)

        # A_hat summary
        sh = _finite(sub, "A_hat")
        if len(sh):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(sh["mass_GeV"], sh["A_hat"], color="C2", label=r"$\hat{A}$")
            if "sigma_A" in sh.columns:
                sig = sh["sigma_A"].to_numpy(float)
                ah = sh["A_hat"].to_numpy(float)
                ax.fill_between(sh["mass_GeV"], ah - sig, ah + sig, color="C2", alpha=0.2, label=r"$\hat{A}\pm\sigma_A$")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel(r"$\hat{A}$")
            _set_title_above(ax, f"{ds}: extracted signal amplitude")
            ax.legend(loc="best", fontsize=8, frameon=True)
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"A_hat_{ds}.png"), dpi=200)
            plt.close(fig)

        # p0 and Z summaries (local+global)
        sp = _finite(sub, "p0_analytic")
        if len(sp):
            plot_analytic_p0(
                sp,
                title=f"{ds}: analytic local/global p0",
                outpath=os.path.join(outdir, f"p0_{ds}.png"),
                apply_lee=True,
                neff=float(max(len(sp), 1.0)),
            )
            plot_Z_local_global(
                sp,
                title=f"{ds}: local/global Z",
                outpath=os.path.join(outdir, f"Z_local_global_{ds}.png"),
                apply_lee=True,
                neff=float(max(len(sp), 1.0)),
            )

    # Overlay curves including combined eps2
    if len(datasets):
        fig, ax = plt.subplots(figsize=(9, 4))
        for ds in datasets:
            sub = _finite(df_single[df_single["dataset"] == ds].copy(), "eps2_up")
            if len(sub):
                ax.plot(sub["mass_GeV"], sub["eps2_up"], label=ds)
        if len(df_comb) and "eps2_up" in df_comb.columns:
            subc = _finite(df_comb.copy(), "eps2_up")
            if len(subc):
                ax.plot(subc["mass_GeV"], subc["eps2_up"], label="combined", lw=2.0, color="k")
        ax.set_yscale("log")
        ax.set_xlabel("m (GeV)")
        ax.set_ylabel(r"$\epsilon^2_{up}$")
        _set_title_above(ax, "epsilon^2 UL: datasets and combined")
        ax.legend(loc="best", fontsize=8, frameon=True)
        _grid(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "eps2_ul_overlay.png"), dpi=200)
        plt.close(fig)

    plot_gp_hyperparameters(df_single, os.path.join(outdir, "gp_hyperparameters"))

    # summary CSV for rapid publication workflows
    summary_csv = os.path.join(outdir, "scan_summary_single.csv")
    df_single.to_csv(summary_csv, index=False)
    print("Wrote summary plots to", outdir)
    print("Wrote:", summary_csv)


def plot_gp_hyperparameters(df_single: pd.DataFrame, outdir: str) -> None:
    """Write GP kernel/hyper-parameter QA plots from scan dataframe."""
    if df_single is None or df_single.empty or "dataset" not in df_single.columns:
        return
    ensure_dir(outdir)

    for ds in sorted(df_single["dataset"].unique()):
        sub = df_single[df_single["dataset"] == ds].copy().sort_values("mass_GeV")
        if len(sub) == 0:
            continue
        x = sub["mass_GeV"].to_numpy(float)

        if "ls_opt_over_sigma_x" in sub.columns and np.isfinite(sub["ls_opt_over_sigma_x"].to_numpy(float)).any():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, sub["ls_opt_over_sigma_x"], label=r"$\ell_{opt}/\sigma_x$")
            if "ls_hi_over_sigma_x" in sub.columns:
                ax.plot(x, sub["ls_hi_over_sigma_x"], "--", label=r"$\ell_{hi}/\sigma_x$")
            if "ls_lo_over_sigma_x" in sub.columns:
                ax.plot(x, sub["ls_lo_over_sigma_x"], "--", label=r"$\ell_{lo}/\sigma_x$")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel("dimensionless ratio")
            _set_title_above(ax, f"{ds}: GP length-scale / sigma_x")
            ax.legend(loc="best", fontsize=8, frameon=True)
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"gp_ls_ratio_{ds}.png"), dpi=200)
            plt.close(fig)

        if "ls_opt" in sub.columns and np.isfinite(sub["ls_opt"].to_numpy(float)).any():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, sub["ls_opt"], label=r"$\ell_{opt}$")
            if "ls_hi" in sub.columns:
                ax.plot(x, sub["ls_hi"], "--", label=r"$\ell_{hi}$")
            if "ls_lo" in sub.columns:
                ax.plot(x, sub["ls_lo"], "--", label=r"$\ell_{lo}$")
            if "sigma_x" in sub.columns:
                ax.plot(x, sub["sigma_x"], ":", label=r"$\sigma_x$")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel("x-space scale")
            _set_title_above(ax, f"{ds}: GP length scales")
            ax.legend(loc="best", fontsize=8, frameon=True)
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"gp_ls_abs_{ds}.png"), dpi=200)
            plt.close(fig)

        if "const_opt" in sub.columns and np.isfinite(sub["const_opt"].to_numpy(float)).any():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, sub["const_opt"], label="constant amplitude")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel("ConstantKernel")
            _set_title_above(ax, f"{ds}: ConstantKernel amplitude")
            ax.legend(loc="best", fontsize=8, frameon=True)
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"gp_const_{ds}.png"), dpi=200)
            plt.close(fig)

        if "lml" in sub.columns and np.isfinite(sub["lml"].to_numpy(float)).any():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, sub["lml"], label="log marginal likelihood")
            ax.set_xlabel("m (GeV)")
            ax.set_ylabel("LML")
            _set_title_above(ax, f"{ds}: GP log marginal likelihood")
            ax.legend(loc="best", fontsize=8, frameon=True)
            _grid(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"gp_lml_{ds}.png"), dpi=200)
            plt.close(fig)


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
