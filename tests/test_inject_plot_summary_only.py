from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from hps_gpr.cli import main


def test_inject_plot_runs_with_summary_csvs_only(tmp_path: Path):
    rows = []
    for m in (0.050, 0.060):
        for z, a in ((1.0, 10.0), (2.0, 20.0)):
            rows.append(
                {
                    "dataset": "2015",
                    "mass_GeV": m,
                    "strength": a,
                    "inj_nsigma": z,
                    "n_toys": 100,
                    "A_hat_mean": a + 0.2,
                    "sigma_A_mean": 2.0,
                    "pull_mean": 0.02,
                    "pull_std": 1.01,
                    "cov_1sigma": 0.68,
                    "cov_2sigma": 0.95,
                    "sigmaA_ref": 2.0,
                }
            )
    df = pd.DataFrame(rows)
    inp = tmp_path / "injection_flat"
    inp.mkdir(parents=True, exist_ok=True)
    df.to_csv(inp / "inj_extract_summary_2015__jobds_2015__m_0p05__s_s1.csv", index=False)

    outdir = tmp_path / "injection_summary"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "inject-plot",
            "--input-dir",
            str(inp),
            "--output-dir",
            str(outdir),
        ],
    )

    assert res.exit_code == 0, res.output
    assert (outdir / "coverage_2015.png").exists()
    assert (outdir / "pull_vs_mass_2015.png").exists()
