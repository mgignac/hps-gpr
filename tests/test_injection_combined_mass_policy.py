import pandas as pd

from hps_gpr.injection import combine_injection_toy_tables


def _make_toys(dataset: str, masses):
    rows = []
    for mass in masses:
        rows.append(
            {
                "dataset": dataset,
                "mass_GeV": float(mass),
                "strength": 100.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 100.0,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 2.0,
                "success": True,
            }
        )
    return pd.DataFrame(rows)


def test_combine_injection_toys_excludes_non_overlapping_masses_by_default():
    df_2015 = _make_toys("2015", [0.050, 0.060])
    df_2016 = _make_toys("2016", [0.070, 0.080])

    out = combine_injection_toy_tables({"2015": df_2015, "2016": df_2016})

    assert out.empty


def test_combine_injection_toys_union_min_n_requires_two_contributors():
    df_2015 = _make_toys("2015", [0.050, 0.060])
    df_2016 = _make_toys("2016", [0.060, 0.080])

    out = combine_injection_toy_tables(
        {"2015": df_2015, "2016": df_2016},
        mass_policy="union_min_n",
        min_n_contrib=2,
    )

    assert sorted(out["mass_GeV"].tolist()) == [0.06]
    assert int(out.iloc[0]["n_contrib"]) == 2
    assert out.iloc[0]["contrib_datasets"] == "2015+2016"


def test_combine_injection_toys_intersection_excludes_mismatched_strength_rows():
    df_2015 = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.060,
                "strength": 100.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 100.0,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )
    df_2016 = pd.DataFrame(
        [
            {
                "dataset": "2016",
                "mass_GeV": 0.060,
                "strength": 120.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 120.0,
                "inj_nsigma": 1.2,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )

    out = combine_injection_toy_tables({"2015": df_2015, "2016": df_2016})

    assert out.empty


def test_combine_injection_toys_sigma_mode_groups_by_inj_nsigma_with_rounding():
    df_2015 = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.060,
                "strength": 100.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 99.0,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )
    df_2016 = pd.DataFrame(
        [
            {
                "dataset": "2016",
                "mass_GeV": 0.060,
                "strength": 120.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 121.0,
                "inj_nsigma": 1.0 + 1e-10,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )

    out = combine_injection_toy_tables({"2015": df_2015, "2016": df_2016})

    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_contrib"]) == 2
    assert row["contrib_datasets"] == "2015+2016"
    assert row["inj_nsigma"] == 1.0
    assert row["strength"] == 110.0


def test_combine_injection_toys_sigma_mode_intersection_requires_all_datasets_per_group():
    df_2015 = pd.DataFrame(
        [
            {
                "dataset": "2015",
                "mass_GeV": 0.060,
                "strength": 100.0,
                "toy": 0,
                "sigma_A": 2.0,
                "A_hat": 100.0,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )
    df_2016 = pd.DataFrame(
        [
            {
                "dataset": "2016",
                "mass_GeV": 0.060,
                "strength": 120.0,
                "toy": 1,
                "sigma_A": 2.0,
                "A_hat": 120.0,
                "inj_nsigma": 1.0,
                "sigmaA_ref": 2.0,
                "success": True,
            },
        ]
    )

    out = combine_injection_toy_tables({"2015": df_2015, "2016": df_2016})

    assert out.empty
