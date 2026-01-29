"""Plotting functions for HPS GPR analysis."""

import os
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .template import build_template

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig
    from .io import BlindPrediction


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
    x = pred.x_full
    y = pred.y_full
    mu = pred.mu_full

    plt.figure()
    plt.step(x, y, where="mid", label="Data")
    plt.plot(x, mu, label="GPR fit")
    plt.axvspan(pred.blind[0], pred.blind[1], alpha=0.2, label="Blind region")
    plt.xlabel("m (GeV)")
    plt.ylabel("Counts / bin")
    plt.title(f"{ds.label} — full range fit @ {mass:.3f} GeV {title_extra}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_blind_window(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    A_show: Optional[float],
    outpath: str,
    title_extra: str = "",
) -> None:
    """Plot blind window with optional signal overlay.

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

    plt.figure()
    plt.step(centers, y, where="mid", label="Data")
    plt.plot(centers, mu, label="Background")
    if A_show is not None and np.isfinite(A_show):
        plt.plot(centers, mu + float(A_show) * w, label=f"Bkg + signal (A={A_show:.1f})")
    plt.xlabel("m (GeV)")
    plt.ylabel("Counts / bin")
    plt.title(f"{ds.label} — blind window @ {mass:.3f} GeV {title_extra}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_s_over_b(
    ds: "DatasetConfig",
    mass: float,
    pred: "BlindPrediction",
    A_show: float,
    outpath: str,
) -> None:
    """Plot signal over background ratio.

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

    plt.figure()
    plt.plot(centers, s / b)
    plt.xlabel("m (GeV)")
    plt.ylabel("s/b (template)")
    plt.title(f"{ds.label} — s/b template @ {mass:.3f} GeV")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_eps2_curves(
    df_single: pd.DataFrame,
    df_comb: pd.DataFrame,
    outdir: str,
) -> None:
    """Plot epsilon^2 upper limit curves.

    Args:
        df_single: Single-dataset results DataFrame
        df_comb: Combined results DataFrame
        outdir: Output directory
    """
    ensure_dir(outdir)

    # Per-dataset curves
    for ds in sorted(df_single["dataset"].unique()):
        sub = df_single[df_single["dataset"] == ds].copy()
        sub = sub[np.isfinite(sub["eps2_up"].to_numpy(float))]
        if len(sub) == 0:
            continue

        plt.figure()
        plt.plot(sub["mass_GeV"], sub["eps2_up"])
        plt.yscale("log")
        plt.xlabel("m (GeV)")
        plt.ylabel(r"$\epsilon^2$ UL")
        plt.title(f"{ds}: epsilon^2 UL")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"eps2_ul_{ds}.png"), dpi=180)
        plt.close()

    # Combined curve
    subc = df_comb.copy()
    subc = subc[np.isfinite(subc["eps2_up"].to_numpy(float))]
    if len(subc) > 0:
        plt.figure()
        plt.plot(subc["mass_GeV"], subc["eps2_up"])
        plt.yscale("log")
        plt.xlabel("m (GeV)")
        plt.ylabel(r"$\epsilon^2$ UL (combined)")
        plt.title("Combined epsilon^2 UL (overlap points)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "eps2_ul_combined.png"), dpi=180)
        plt.close()

        # Overlay plot
        plt.figure()
        for ds in sorted(df_single["dataset"].unique()):
            sub = df_single[df_single["dataset"] == ds].copy()
            sub = sub[np.isfinite(sub["eps2_up"].to_numpy(float))]
            if len(sub) == 0:
                continue
            plt.plot(sub["mass_GeV"], sub["eps2_up"], label=ds)
        plt.plot(subc["mass_GeV"], subc["eps2_up"], label="combined", linewidth=2)
        plt.yscale("log")
        plt.xlabel("m (GeV)")
        plt.ylabel(r"$\epsilon^2$ UL")
        plt.title("epsilon^2 UL: individual + combined")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "eps2_ul_overlay.png"), dpi=180)
        plt.close()

    print("Wrote summary plots to", outdir)


def plot_bands(
    df_bands: pd.DataFrame,
    outpath: str,
    column_prefix: str = "A",
    ylabel: str = "A UL",
    title: str = "Expected UL bands",
) -> None:
    """Plot expected upper limit bands.

    Args:
        df_bands: DataFrame with band information
        outpath: Output file path
        column_prefix: Prefix for column names (A or eps2)
        ylabel: Y-axis label
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    masses = df_bands["mass_GeV"].values
    obs = df_bands[f"{column_prefix}_obs"].values

    # 2-sigma band
    plt.fill_between(
        masses,
        df_bands[f"{column_prefix}_lo2"].values,
        df_bands[f"{column_prefix}_hi2"].values,
        alpha=0.3,
        color="yellow",
        label=r"$\pm 2\sigma$",
    )

    # 1-sigma band
    plt.fill_between(
        masses,
        df_bands[f"{column_prefix}_lo1"].values,
        df_bands[f"{column_prefix}_hi1"].values,
        alpha=0.5,
        color="green",
        label=r"$\pm 1\sigma$",
    )

    # Median expected
    plt.plot(
        masses,
        df_bands[f"{column_prefix}_med"].values,
        "k--",
        label="Expected median",
    )

    # Observed
    plt.plot(masses, obs, "k-", linewidth=2, label="Observed")

    plt.xlabel("m (GeV)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
