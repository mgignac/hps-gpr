"""Scan driver and CSV writers."""

import json
import os
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .evaluation import evaluate_single_dataset, evaluate_combined
from .plotting import (
    make_mass_folder,
    ensure_dir,
    plot_full_range,
    plot_blind_window,
    plot_s_over_b,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def active_datasets_for_mass(
    mass: float, datasets: Dict[str, "DatasetConfig"]
) -> List["DatasetConfig"]:
    """Get list of datasets active at a given mass.

    Args:
        mass: Signal mass hypothesis (GeV)
        datasets: Dictionary of dataset configurations

    Returns:
        List of DatasetConfig objects covering this mass
    """
    out = []
    for ds in datasets.values():
        if (mass >= ds.m_low) and (mass <= ds.m_high):
            out.append(ds)
    return out


def union_scan_grid(
    datasets: Dict[str, "DatasetConfig"], step: float
) -> np.ndarray:
    """Generate mass scan grid covering all datasets.

    Args:
        datasets: Dictionary of dataset configurations
        step: Mass step size (GeV)

    Returns:
        Array of mass values
    """
    lo = min([d.m_low for d in datasets.values()])
    hi = max([d.m_high for d in datasets.values()])
    masses = np.arange(lo, hi + 0.5 * step, step)
    return np.round(masses, 3)


def _write_json(path: str, payload: dict) -> None:
    """Write a dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_scan(
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
    mass_min: float = None,
    mass_max: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full mass scan.

    Args:
        datasets: Dictionary of enabled datasets
        config: Global configuration
        mass_min: Minimum mass to scan (optional)
        mass_max: Maximum mass to scan (optional)

    Returns:
        Tuple of (single-dataset DataFrame, combined DataFrame)
    """
    masses = union_scan_grid(datasets, config.mass_step_gev)

    # Apply mass range filter if specified
    if mass_min is not None:
        masses = masses[masses >= mass_min]
    if mass_max is not None:
        masses = masses[masses <= mass_max]

    rows_single = []
    rows_comb = []
    err_count = 0

    for m in masses:
        ds_here = active_datasets_for_mass(float(m), datasets)
        if not ds_here:
            continue

        mass_dir = (
            make_mass_folder(config.output_dir, float(m))
            if config.save_per_mass_folders
            else config.output_dir
        )

        preds_here = []
        ds_list_here = []

        # Individual fits
        for ds in ds_here:
            ds_dir = os.path.join(mass_dir, ds.key)
            ensure_dir(ds_dir)

            try:
                res, pred = evaluate_single_dataset(ds, float(m), config, do_extraction=True)
                preds_here.append(pred)
                ds_list_here.append(ds)

                if config.save_plots:
                    plot_full_range(
                        ds, float(m), pred, os.path.join(ds_dir, "fit_full.png")
                    )
                    plot_blind_window(
                        ds,
                        float(m),
                        pred,
                        res.A_up,
                        os.path.join(ds_dir, "blind_ul.png"),
                        title_extra="(UL overlay)",
                    )
                    plot_blind_window(
                        ds,
                        float(m),
                        pred,
                        res.A_hat,
                        os.path.join(ds_dir, "blind_ahat.png"),
                        title_extra="(Ahat overlay)",
                    )
                    plot_s_over_b(
                        ds, float(m), pred, res.A_up, os.path.join(ds_dir, "s_over_b_ul.png")
                    )

                if config.save_fit_json:
                    _write_json(
                        os.path.join(ds_dir, "numbers.json"),
                        {
                            "dataset": ds.key,
                            "mass_GeV": float(m),
                            "A_up": res.A_up,
                            "eps2_up": res.eps2_up,
                            "p0_analytic": res.p0_analytic,
                            "Z_analytic": res.Z_analytic,
                            "A_hat": res.A_hat,
                            "sigma_A": res.sigma_A,
                            "extract_success": res.extract_success,
                            "sigma_val": pred.sigma_val,
                            "blind": list(pred.blind),
                            "integral_density": pred.integral_density,
                        },
                    )

                rows_single.append(
                    {
                        "dataset": ds.key,
                        "mass_GeV": res.mass,
                        "A_up": res.A_up,
                        "eps2_up": res.eps2_up,
                        "p0_analytic": res.p0_analytic,
                        "Z_analytic": res.Z_analytic,
                        "A_hat": res.A_hat,
                        "sigma_A": res.sigma_A,
                        "extract_success": res.extract_success,
                    }
                )
            except Exception as e:
                try:
                    with open(os.path.join(ds_dir, "error.txt"), "w") as ef:
                        ef.write(str(e) + "\n")
                except Exception:
                    pass

                if config.debug_print and err_count < config.debug_max_errors:
                    print(f"[ERROR] {ds.key} @ {float(m):.3f} GeV: {e}")
                    err_count += 1

                rows_single.append(
                    {
                        "dataset": ds.key,
                        "mass_GeV": float(m),
                        "A_up": np.nan,
                        "eps2_up": np.nan,
                        "p0_analytic": np.nan,
                        "Z_analytic": np.nan,
                        "A_hat": np.nan,
                        "sigma_A": np.nan,
                        "extract_success": False,
                        "error": str(e),
                    }
                )

        # Combined fit in overlap regions
        if len(ds_list_here) >= 2:
            try:
                comb = evaluate_combined(float(m), ds_list_here, preds_here, config)

                if config.save_plots:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    plt.figure()
                    plt.axis("off")
                    plt.text(0.05, 0.8, f"Combined @ {float(m):.3f} GeV", fontsize=12)
                    plt.text(0.05, 0.6, f"eps2_up = {comb.eps2_up:.3e}", fontsize=12)
                    plt.text(
                        0.05,
                        0.4,
                        f"p0 = {comb.p0_analytic:.3e}   Z = {comb.Z_analytic:.2f}",
                        fontsize=12,
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(cdir, "combined_summary.png"), dpi=160)
                    plt.close()

                if config.save_fit_json:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    _write_json(
                        os.path.join(cdir, "numbers.json"),
                        {
                            "mass_GeV": float(m),
                            "datasets": [d.key for d in ds_list_here],
                            "eps2_up": comb.eps2_up,
                            "p0_analytic": comb.p0_analytic,
                            "Z_analytic": comb.Z_analytic,
                        },
                    )

                rows_comb.append(
                    {
                        "mass_GeV": comb.mass,
                        "datasets": "+".join([d.key for d in ds_list_here]),
                        "eps2_up": comb.eps2_up,
                        "p0_analytic": comb.p0_analytic,
                        "Z_analytic": comb.Z_analytic,
                    }
                )
            except Exception as e:
                rows_comb.append(
                    {
                        "mass_GeV": float(m),
                        "datasets": "+".join([d.key for d in ds_list_here]),
                        "eps2_up": np.nan,
                        "p0_analytic": np.nan,
                        "Z_analytic": np.nan,
                        "error": str(e),
                    }
                )

        # Progress print every 25 MeV
        if int(round(float(m) * 1000)) % 25 == 0:
            print(f"[scan] reached {float(m):.3f} GeV")

    df_single = (
        pd.DataFrame(rows_single)
        .sort_values(["dataset", "mass_GeV"])
        .reset_index(drop=True)
    )
    df_comb = pd.DataFrame(rows_comb).sort_values(["mass_GeV"]).reset_index(drop=True)

    single_path = os.path.join(config.output_dir, "results_single.csv")
    comb_path = os.path.join(config.output_dir, "results_combined.csv")
    df_single.to_csv(single_path, index=False)
    df_comb.to_csv(comb_path, index=False)

    print("Wrote:", single_path)
    print("Wrote:", comb_path)

    return df_single, df_comb
