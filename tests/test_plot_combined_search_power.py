import pandas as pd

from hps_gpr.plotting import plot_combined_search_power


def test_plot_combined_search_power_writes_expected_outputs(tmp_path):
    rows = []
    for ds, sref in (("2015", 2.0), ("2016", 1.5), ("2021", 1.2)):
        for m in (0.040, 0.080, 0.110):
            rows.append(
                {
                    "dataset": ds,
                    "mass_GeV": m,
                    "sigmaA_ref": sref,
                    "toy": 0,
                    "A_hat": 0.0,
                    "sigma_A": 1.0,
                    "Zhat": 0.0,
                }
            )
    df = pd.DataFrame(rows)

    saved = plot_combined_search_power(df, outdir=str(tmp_path))

    assert len(saved) == 5
    assert (tmp_path / "combined_search_power_scenarios.png").exists()
    assert (tmp_path / "combined_search_power_constituent_pvalues_5sigma.png").exists()
    assert (tmp_path / "combined_constituent_pvalues_target5sigma.csv").exists()
    assert (tmp_path / "combined_signal_allocation_m040MeV.png").exists()
    assert (tmp_path / "combined_signal_allocation_m080MeV.png").exists()
    assert (tmp_path / "combined_signal_allocation_m110MeV.png").exists()
    assert (tmp_path / "combined_signal_allocation_m040MeV.csv").exists()
