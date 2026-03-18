import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hps_gpr.plotting as plotting_module
from hps_gpr.plotting import (
    plot_combined_search_power,
    plot_projected_unblinded_eps2_reach,
    project_unblinded_eps2_reach_table,
)

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
    assert (tmp_path / "combined_signal_allocation_m120MeV.png").exists()
    assert (tmp_path / "combined_signal_allocation_m040MeV.csv").exists()
    req = pd.read_csv(tmp_path / "combined_constituent_pvalues_target5sigma.csv")
    assert "z_required_2021" in req.columns
    row = req.loc[np.isclose(req["mass_GeV"].to_numpy(float), 0.040)].iloc[0]
    weights = {"2015": 1.0 / 2.0**2, "2016": 1.0 / 1.5**2, "2021": 1.0 / 1.2**2}
    sum_w = sum(weights.values())
    assert row["info_frac_2015"] == pytest.approx(weights["2015"] / sum_w)
    assert row["z_required_2021"] == pytest.approx(5.0 * (weights["2021"] / sum_w) ** 0.5)
    alloc = pd.read_csv(tmp_path / "combined_signal_allocation_m120MeV.csv")
    assert "A_inj_2021" in alloc.columns
    assert alloc["mass_requested_GeV"].iloc[0] == pytest.approx(0.120)
    assert alloc["mass_selected_GeV"].iloc[0] == pytest.approx(0.110)


def test_plot_combined_search_power_uses_stacked_layout(tmp_path, monkeypatch):
    rows = []
    for ds, sref in (("2015", 2.0), ("2016_10pct", 1.5), ("2021_1pct", 1.2)):
        for m in (0.040, 0.080, 0.120):
            rows.append({"dataset": ds, "mass_GeV": m, "sigmaA_ref": sref})
    df = pd.DataFrame(rows)

    captured = {}

    def fake_save(fig, outpath, **_kwargs):
        captured[os.path.basename(outpath)] = fig

    monkeypatch.setattr(plotting_module, "_save_plot_outputs", fake_save)

    plot_combined_search_power(df, outdir=str(tmp_path), masses_focus=[0.040], z_targets=[1.0])

    scen_fig = captured["combined_search_power_scenarios.png"]
    const_fig = captured["combined_search_power_constituent_pvalues_5sigma.png"]
    assert len(scen_fig.axes) == 2
    assert scen_fig.axes[0].get_position().y0 > scen_fig.axes[1].get_position().y0
    assert len(const_fig.axes) == 2
    assert const_fig.axes[0].get_position().y0 > const_fig.axes[1].get_position().y0

    for fig in captured.values():
        plt.close(fig)


def test_project_unblinded_eps2_reach_table_applies_sqrt_l_scaling():
    df = pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.040, "eps2_up": 10.0},
            {"dataset": "2016", "mass_GeV": 0.040, "eps2_up": 20.0},
            {"dataset": "2021", "mass_GeV": 0.040, "eps2_up": 30.0},
        ]
    )

    out = project_unblinded_eps2_reach_table(df)

    row = out.iloc[0]
    expected_current = 1.0 / ((1.0 / 10.0**2 + 1.0 / 20.0**2 + 1.0 / 30.0**2) ** 0.5)
    expected_projected = 1.0 / ((1.0 / 10.0**2 + 10.0 / 20.0**2 + 100.0 / 30.0**2) ** 0.5)
    assert row["current_obs"] == pytest.approx(expected_current)
    assert row["projected_obs"] == pytest.approx(expected_projected)


def test_project_unblinded_eps2_reach_table_supports_band_columns():
    df = pd.DataFrame(
        [
            {"dataset_set": "2015", "mass_MeV": 40.0, "eps2_obs": 10.0, "eps2_med": 9.0, "eps2_lo1": 8.0, "eps2_hi1": 11.0},
            {"dataset_set": "2016_10pct", "mass_MeV": 40.0, "eps2_obs": 20.0, "eps2_med": 18.0, "eps2_lo1": 16.0, "eps2_hi1": 22.0},
            {"dataset_set": "2015_2016_combined", "mass_MeV": 40.0, "eps2_obs": 15.0, "eps2_med": 14.0, "eps2_lo1": 12.0, "eps2_hi1": 17.0},
        ]
    )

    out = project_unblinded_eps2_reach_table(df, lumi_scale_by_dataset={"2016": 4.0})

    row = out.iloc[0]
    assert row["mass_GeV"] == pytest.approx(0.040)
    assert "current_med" in out.columns
    assert "projected_hi1" in out.columns
    assert row["lumi_scale_2016"] == 4.0


def test_plot_projected_unblinded_eps2_reach_writes_outputs(tmp_path):
    df = pd.DataFrame(
        [
            {"dataset_set": "2015", "mass_MeV": 40.0, "eps2_obs": 10.0, "eps2_med": 9.0, "eps2_lo1": 8.0, "eps2_hi1": 11.0},
            {"dataset_set": "2021_1pct", "mass_MeV": 40.0, "eps2_obs": 30.0, "eps2_med": 27.0, "eps2_lo1": 24.0, "eps2_hi1": 33.0},
        ]
    )

    outpath = tmp_path / "projected_unblinded_reach_eps2.png"
    table = plot_projected_unblinded_eps2_reach(df, outpath=str(outpath))

    assert outpath.exists()
    assert (tmp_path / "projected_unblinded_reach_eps2.pdf").exists()
    assert "projected_obs_2021" in table.columns
