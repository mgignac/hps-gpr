"""Signal injection and extraction studies."""

import os
from typing import List, TYPE_CHECKING

import numpy as np
import pandas as pd

from .io import estimate_background_for_dataset
from .template import build_template
from .statistics import fit_A_profiled_gaussian
from .plotting import ensure_dir

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def inject_counts(
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    strength: int,
    rng: np.random.Generator,
    mode: str = "multinomial",
) -> np.ndarray:
    """Inject signal counts into bins.

    Args:
        edges: Bin edges
        mass: Signal mass (GeV)
        sigma_val: Mass resolution
        strength: Total number of signal events to inject
        rng: Random number generator
        mode: "multinomial" or "poisson"

    Returns:
        Array of injected counts per bin
    """
    if strength <= 0:
        return np.zeros(len(edges) - 1, dtype=int)

    w = build_template(edges, mass, sigma_val)

    if mode == "multinomial":
        return rng.multinomial(int(strength), w).astype(int)
    if mode == "poisson":
        return rng.poisson(float(strength) * w).astype(int)

    raise ValueError("mode must be multinomial or poisson")


def run_injection_extraction(
    ds: "DatasetConfig",
    masses: List[float],
    strengths: List[int],
    config: "Config",
    outdir: str = None,
    seed: int = 314159,
) -> pd.DataFrame:
    """Run injection/extraction study.

    Args:
        ds: Dataset configuration
        masses: List of mass values to test
        strengths: List of injection strengths to test
        config: Global configuration
        outdir: Output directory (defaults to config.output_dir/injection_extraction)
        seed: Random seed

    Returns:
        DataFrame with injection/extraction results
    """
    if outdir is None:
        outdir = os.path.join(config.output_dir, "injection_extraction")

    ensure_dir(outdir)
    rows = []
    base_rng = np.random.default_rng(seed)

    for m in masses:
        pred = estimate_background_for_dataset(ds, float(m), config)
        tmpl = build_template(pred.edges, float(m), pred.sigma_val)

        for N in strengths:
            seed_i = int(base_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed_i)

            n_sig = inject_counts(
                pred.edges,
                float(m),
                pred.sigma_val,
                int(N),
                rng,
                mode=config.inj_mode,
            )
            obs_inj = (pred.obs + n_sig).astype(int)

            fit = fit_A_profiled_gaussian(
                obs_inj,
                pred.mu,
                pred.cov,
                tmpl,
                allow_negative=config.extract_allow_negative,
            )

            rows.append(
                dict(
                    dataset=ds.key,
                    mass_GeV=float(m),
                    strength=int(N),
                    seed=seed_i,
                    A_hat=float(fit["A_hat"]),
                    sigma_A=float(fit["sigma_A"]),
                    success=bool(fit["success"]),
                    nll=float(fit["nll"]),
                    obs_sum=int(np.sum(obs_inj)),
                    bkg_sum=float(np.sum(pred.mu)),
                    bkg_unc_sum=float(np.sqrt(max(float(np.sum(pred.cov)), 0.0))),
                )
            )

        # Write per-mass file
        df_m = pd.DataFrame(
            [
                r
                for r in rows
                if (r["dataset"] == ds.key and abs(r["mass_GeV"] - float(m)) < 1e-12)
            ]
        )
        df_m.to_csv(
            os.path.join(outdir, f"injection_extraction_{ds.key}_{float(m):.3f}GeV.csv"),
            index=False,
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"injection_extraction_{ds.key}_grid.csv"), index=False)
    return df
