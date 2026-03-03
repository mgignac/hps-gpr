from pathlib import Path

import numpy as np
import pandas as pd

from hps_gpr.plotting import plot_z_calibration_residual


def test_plot_z_calibration_residual_writes_dataset_and_comparison_panels(tmp_path: Path):
    rng = np.random.default_rng(7)
    rows = []
    for ds in ("2015", "2016"):
        for m in (0.045, 0.055):
            for zinj in (0.0, 1.0):
                for toy in range(25):
                    zhat = zinj + 0.1 * np.sin(10 * m) + rng.normal(0.0, 0.2)
                    rows.append({
                        "dataset": ds,
                        "mass_GeV": m,
                        "inj_nsigma": zinj,
                        "Zhat": zhat,
                        "toy": toy,
                    })

    df = pd.DataFrame(rows)
    plot_z_calibration_residual(
        df,
        outdir=str(tmp_path),
        acceptance_bands=[0.5],
        band_semantic="toy spread",
    )

    assert (tmp_path / "z_calibration_residual_2015.png").exists()
    assert (tmp_path / "z_calibration_residual_2016.png").exists()
    assert (tmp_path / "z_calibration_residual_comparison.png").exists()
