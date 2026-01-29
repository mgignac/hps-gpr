"""Dataset validation utilities."""

import json
import os
from typing import Any, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def _validate_hist_with_uproot(path: str, hist_name: str):
    """Load and validate a histogram from a ROOT file using uproot."""
    try:
        import uproot
    except ImportError as e:
        raise RuntimeError(
            "uproot is required for validation but not installed"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(f"ROOT file not found: {path}")

    f = uproot.open(path)
    if hist_name not in f:
        keys = list(f.keys())[:30]
        raise KeyError(
            f"Histogram '{hist_name}' not found in {path}. First keys: {keys}"
        )

    h = f[hist_name]
    vals, edges = h.to_numpy()
    vals = vals.astype(float)
    edges = edges.astype(float)
    return vals, edges


def validate_datasets(
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
) -> Dict[str, Dict[str, Any]]:
    """Validate all enabled datasets.

    Checks:
    - ROOT file exists
    - Histogram exists in file
    - Analysis range overlaps histogram edges
    - sigma(m) and frad(m) are valid at test mass

    Args:
        datasets: Dictionary of enabled datasets
        config: Global configuration

    Returns:
        Validation report dictionary
    """
    report = {}

    for k, ds in datasets.items():
        try:
            vals, edges = _validate_hist_with_uproot(ds.root_path, ds.hist_name)
            lo_e, hi_e = float(edges[0]), float(edges[-1])
            tot = float(np.sum(vals))

            # Range overlap check
            in_range = (
                (ds.m_low < ds.m_high)
                and (ds.m_low < hi_e)
                and (ds.m_high > lo_e)
            )

            # sigma/frad sanity at a representative mass
            m_test = float(0.5 * (max(ds.m_low, lo_e) + min(ds.m_high, hi_e)))
            sig = ds.sigma(m_test)
            fr = ds.frad(m_test)

            ok = True
            msgs = []

            if not in_range:
                ok = False
                msgs.append(
                    f"Dataset range [{ds.m_low},{ds.m_high}] does not overlap "
                    f"hist edges [{lo_e},{hi_e}]"
                )

            if not np.isfinite(sig) or sig <= 0:
                ok = False
                msgs.append(
                    f"sigma(m) invalid at m={m_test:.4f} GeV: sigma={sig}"
                )

            if not np.isfinite(fr) or fr <= 0:
                msgs.append(
                    f"WARNING: frad(m) <= 0 at m={m_test:.4f} GeV: frad={fr}. "
                    "eps2 conversion will be NaN."
                )

            report[k] = dict(
                ok=ok,
                messages=msgs,
                file=ds.root_path,
                hist=ds.hist_name,
                hist_edges=[lo_e, hi_e],
                total_counts=tot,
                test_mass=m_test,
                sigma_at_test=sig,
                frad_at_test=fr,
            )

            if config.debug_print:
                print(
                    f"[validate] {k}: hist edges [{lo_e:.4f},{hi_e:.4f}]  "
                    f"total={tot:.3g}  test m={m_test:.4f}  "
                    f"sigma={sig:.4g}  frad={fr:.4g}"
                )
                for m in msgs:
                    print("          ", m)

            if config.fail_fast and (not ok):
                raise RuntimeError(f"Validation failed for {k}: {msgs}")

        except Exception as e:
            report[k] = dict(
                ok=False,
                error=str(e),
                file=ds.root_path,
                hist=ds.hist_name,
            )
            print(f"[validate:ERROR] {k}: {e}")
            if config.fail_fast:
                raise

    # Write report
    out = os.path.join(config.output_dir, "validation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[validate] wrote", out)

    return report
