"""Expected upper limit bands calculation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .io import estimate_background_for_dataset
from .template import build_template, cls_limit_for_amplitude, _safe_mvn_draw
from .conversion import epsilon2_from_A

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def expected_ul_bands_for_dataset(
    ds: "DatasetConfig",
    masses: np.ndarray,
    config: "Config",
    n_toys: int = None,
    seed: int = 12345,
) -> pd.DataFrame:
    """Compute expected upper limit bands for a dataset.

    Args:
        ds: Dataset configuration
        masses: Array of mass values to evaluate
        config: Global configuration
        n_toys: Number of toys per mass point (defaults to config.ul_bands_toys)
        seed: Random seed

    Returns:
        DataFrame with band information
    """
    if n_toys is None:
        n_toys = config.ul_bands_toys

    rows = []
    rng = np.random.default_rng(seed)

    for m in masses:
        pred = estimate_background_for_dataset(ds, float(m), config)
        mu = pred.mu.astype(float)
        cov = pred.cov.astype(float)
        edges = pred.edges
        obs = pred.obs.astype(int)

        # Observed limit
        A_obs, _ = cls_limit_for_amplitude(
            obs,
            mu,
            cov,
            edges,
            float(m),
            pred.sigma_val,
            config,
            seed=1,
        )

        # Generate toys
        lam_draws = _safe_mvn_draw(mu, cov, size=int(n_toys), rng=rng)
        n_draws = rng.poisson(lam_draws)

        toy_uls = []
        for i in range(int(n_toys)):
            A_i, _ = cls_limit_for_amplitude(
                n_draws[i],
                mu,
                cov,
                edges,
                float(m),
                pred.sigma_val,
                config,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            toy_uls.append(A_i)
        toy_uls = np.asarray(toy_uls, float)

        # Quantiles
        lo2, lo1, med, hi1, hi2 = [
            float(x)
            for x in np.nanquantile(toy_uls, [0.025, 0.16, 0.5, 0.84, 0.975])
        ]

        # P-values
        p_strong = float(np.mean(toy_uls <= A_obs))
        p_weak = float(np.mean(toy_uls >= A_obs))
        p_two = float(2.0 * min(p_strong, p_weak))

        # Convert to epsilon^2
        if config.make_eps2_bands:
            eps_obs = epsilon2_from_A(ds, float(m), A_obs, pred.integral_density)
            eps_q = [
                epsilon2_from_A(ds, float(m), A, pred.integral_density)
                for A in [lo2, lo1, med, hi1, hi2]
            ]
        else:
            eps_obs = float("nan")
            eps_q = [float("nan")] * 5

        rows.append(
            dict(
                dataset=ds.key,
                mass_GeV=float(m),
                A_obs=float(A_obs),
                A_lo2=lo2,
                A_lo1=lo1,
                A_med=med,
                A_hi1=hi1,
                A_hi2=hi2,
                p_strong=p_strong,
                p_weak=p_weak,
                p_two=p_two,
                eps2_obs=float(eps_obs),
                eps2_lo2=float(eps_q[0]),
                eps2_lo1=float(eps_q[1]),
                eps2_med=float(eps_q[2]),
                eps2_hi1=float(eps_q[3]),
                eps2_hi2=float(eps_q[4]),
            )
        )

    return pd.DataFrame(rows)
