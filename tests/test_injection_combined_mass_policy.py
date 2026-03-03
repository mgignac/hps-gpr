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
