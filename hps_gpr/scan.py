"""Scan driver and CSV writers."""

import json
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from .evaluation import (
    evaluate_single_dataset,
    evaluate_combined,
    active_datasets_for_mass,
    _dataset_visibility,
)
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


def _should_make_scan_diag(
    m: float,
    *,
    diag_every_n: Optional[int] = None,
    step_gev: float = 0.001,
) -> bool:
    """Return True if a diagnostic plot should be made at this mass point."""
    if diag_every_n is not None and int(diag_every_n) > 0:
        idx = int(round(float(m) / float(step_gev)))
        return (idx % int(diag_every_n)) == 0
    return False


def run_scan(
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
    mass_min: float = None,
    mass_max: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full mass scan.

    Uses active_datasets_for_mass() from evaluation.py, which supports the
    config-based edge guards (scan_require_two_sidebands, scan_edge_guard_nsigma).
    Supports joblib parallelization via config.scan_parallel / scan_n_workers.

    Args:
        datasets: Dictionary of enabled datasets
        config: Global configuration
        mass_min: Minimum mass to scan (optional)
        mass_max: Maximum mass to scan (optional)

    Returns:
        Tuple of (single-dataset DataFrame, combined DataFrame)
    """
    masses = union_scan_grid(datasets, config.mass_step_gev)
    if mass_min is not None:
        masses = masses[masses >= mass_min]
    if mass_max is not None:
        masses = masses[masses <= mass_max]

    scan_parallel = bool(getattr(config, "scan_parallel", False))
    n_workers = int(getattr(config, "scan_n_workers", 1) or 1)
    backend = str(getattr(config, "scan_parallel_backend", "loky"))
    threads_per_worker = int(getattr(config, "scan_threads_per_worker", 1) or 1)
    diag_every_n = getattr(config, "scan_diagnostic_plot_every_n", None)
    do_combined = bool(getattr(config, "do_combined", False))

    def _process_one_mass(m: float) -> Tuple[List[dict], List[dict]]:
        """Process a single mass point. Returns (rows_single, rows_comb)."""
        rows_s: List[dict] = []
        rows_c: List[dict] = []

        ds_here = active_datasets_for_mass(float(m), datasets, config)
        if not ds_here:
            return rows_s, rows_c

        mass_dir = (
            make_mass_folder(config.output_dir, float(m))
            if config.save_per_mass_folders
            else config.output_dir
        )

        preds_here = []
        ds_list_here = []

        for ds in ds_here:
            ds_dir = os.path.join(mass_dir, ds.key)
            ensure_dir(ds_dir)
            compute_obs = (_dataset_visibility(ds, config) == "observed")
            make_diag = _should_make_scan_diag(
                m, diag_every_n=diag_every_n, step_gev=float(config.mass_step_gev)
            )

            try:
                with _threadpool_limits(limits=int(threads_per_worker)):
                    res, pred, fitd = evaluate_single_dataset(
                        ds, float(m), config,
                        do_extraction=True,
                        compute_observed=compute_obs,
                        return_fit_details=make_diag,
                    )

                preds_here.append(pred)
                ds_list_here.append(ds)

                if config.save_plots and compute_obs:
                    plot_full_range(
                        ds, float(m), pred,
                        os.path.join(ds_dir, "fit_full.png"),
                    )
                    plot_blind_window(
                        ds, float(m), pred, res.A_up,
                        os.path.join(ds_dir, "blind_ul.png"),
                        title_extra="(UL overlay)",
                    )
                    plot_blind_window(
                        ds, float(m), pred, res.A_hat,
                        os.path.join(ds_dir, "blind_ahat.png"),
                        title_extra="(Ahat overlay)",
                    )
                    plot_s_over_b(
                        ds, float(m), pred, res.A_up,
                        os.path.join(ds_dir, "s_over_b_ul.png"),
                    )

                if config.save_fit_json:
                    _write_json(
                        os.path.join(ds_dir, "numbers.json"),
                        {
                            "dataset": ds.key,
                            "mass_GeV": float(m),
                            "A_up": _jfloat(res.A_up),
                            "eps2_up": _jfloat(res.eps2_up),
                            "p0_analytic": _jfloat(res.p0_analytic),
                            "Z_analytic": _jfloat(res.Z_analytic),
                            "A_hat": _jfloat(res.A_hat),
                            "sigma_A": _jfloat(res.sigma_A),
                            "extract_success": bool(res.extract_success),
                            "sigma_val": _jfloat(pred.sigma_val),
                            "blind": [_jfloat(pred.blind[0]), _jfloat(pred.blind[1])],
                            "integral_density": _jfloat(pred.integral_density),
                            "visibility": "observed" if compute_obs else "expected_only",
                        },
                    )

                rows_s.append({
                    "dataset": ds.key,
                    "mass_GeV": float(res.mass),
                    "sigma_val": float(pred.sigma_val),
                    "blind_lo": float(pred.blind[0]),
                    "blind_hi": float(pred.blind[1]),
                    "A_up": float(res.A_up),
                    "eps2_up": float(res.eps2_up),
                    "p0_analytic": float(res.p0_analytic),
                    "Z_analytic": float(res.Z_analytic),
                    "A_hat": float(res.A_hat),
                    "sigma_A": float(res.sigma_A),
                    "extract_success": bool(res.extract_success),
                    "visibility": "observed" if compute_obs else "expected_only",
                })

            except Exception as e:
                try:
                    with open(os.path.join(ds_dir, "error.txt"), "w") as ef:
                        ef.write(str(e) + "\n")
                except Exception:
                    pass

                if config.debug_print:
                    print(f"[ERROR] {ds.key} @ {float(m):.4f} GeV: {e}")

                rows_s.append({
                    "dataset": ds.key,
                    "mass_GeV": float(m),
                    "sigma_val": float("nan"),
                    "blind_lo": float("nan"), "blind_hi": float("nan"),
                    "A_up": float("nan"), "eps2_up": float("nan"),
                    "p0_analytic": float("nan"), "Z_analytic": float("nan"),
                    "A_hat": float("nan"), "sigma_A": float("nan"),
                    "extract_success": False,
                    "visibility": "error",
                    "error": str(e),
                })

        # Combined fit (only when do_combined=True and >=2 datasets with data)
        if do_combined and len(ds_list_here) >= 2:
            all_obs = all(
                _dataset_visibility(ds, config) == "observed" for ds in ds_list_here
            )
            try:
                comb = evaluate_combined(float(m), ds_list_here, preds_here, config)

                if config.save_plots and all_obs:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.axis("off")
                    ax.text(0.05, 0.8, f"Combined @ {float(m):.4f} GeV",
                            fontsize=12, transform=ax.transAxes)
                    ax.text(0.05, 0.5, f"eps2_up = {comb.eps2_up:.3e}",
                            fontsize=12, transform=ax.transAxes)
                    ax.text(0.05, 0.2,
                            f"p0 = {comb.p0_analytic:.3e}   Z = {comb.Z_analytic:.2f}",
                            fontsize=12, transform=ax.transAxes)
                    plt.tight_layout()
                    plt.savefig(os.path.join(cdir, "combined_summary.png"), dpi=160)
                    plt.close(fig)

                if config.save_fit_json:
                    cdir = os.path.join(mass_dir, "combined")
                    ensure_dir(cdir)
                    _write_json(
                        os.path.join(cdir, "numbers.json"),
                        {
                            "mass_GeV": float(m),
                            "datasets": [d.key for d in ds_list_here],
                            "eps2_up": _jfloat(comb.eps2_up),
                            "p0_analytic": _jfloat(comb.p0_analytic),
                            "Z_analytic": _jfloat(comb.Z_analytic),
                        },
                    )

                rows_c.append({
                    "mass_GeV": float(comb.mass),
                    "datasets": "+".join([d.key for d in ds_list_here]),
                    "n_datasets": len(ds_list_here),
                    "eps2_up": float(comb.eps2_up),
                    "p0_analytic": float(comb.p0_analytic),
                    "Z_analytic": float(comb.Z_analytic),
                })

            except Exception as e:
                rows_c.append({
                    "mass_GeV": float(m),
                    "datasets": "+".join([d.key for d in ds_list_here]),
                    "n_datasets": len(ds_list_here),
                    "eps2_up": float("nan"),
                    "p0_analytic": float("nan"),
                    "Z_analytic": float("nan"),
                    "error": str(e),
                })

        # Progress print every 25 MeV
        if int(round(float(m) * 1000)) % 25 == 0:
            print(f"[scan] reached {float(m):.3f} GeV")

        return rows_s, rows_c

    # Run (parallel or sequential)
    if scan_parallel and n_workers > 1 and _HAVE_JOBLIB:
        results = joblib.Parallel(n_jobs=int(n_workers), backend=str(backend))(
            joblib.delayed(_process_one_mass)(float(m)) for m in masses
        )
    else:
        results = [_process_one_mass(float(m)) for m in masses]

    rows_single: List[dict] = []
    rows_comb: List[dict] = []
    for rs, rc in results:
        rows_single.extend(rs)
        rows_comb.extend(rc)

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


def _jfloat(x) -> object:
    """Convert to float for JSON, replacing inf/nan with None."""
    import math
    v = float(x)
    return None if (math.isnan(v) or math.isinf(v)) else v
