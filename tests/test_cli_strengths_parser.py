import pandas as pd
from click.testing import CliRunner

from hps_gpr.cli import _parse_mass_tokens, _parse_strength_tokens, main


def test_parse_strength_tokens_supports_sigma_prefix():
    vals = _parse_strength_tokens("s1,s2,3,s5")
    assert vals == [1.0, 2.0, 3.0, 5.0]


def test_parse_strength_tokens_none_returns_empty():
    assert _parse_strength_tokens(None) == []


def test_parse_mass_tokens_supports_comma_separated_values():
    vals = _parse_mass_tokens("0.040,0.080,0.120")
    assert vals == [0.04, 0.08, 0.12]


def test_parse_mass_tokens_none_returns_empty():
    assert _parse_mass_tokens(None) == []


def test_project_eps2_reach_cli_creates_missing_output_directory(tmp_path):
    runner = CliRunner()
    input_csv = tmp_path / "combined_single.csv"
    pd.DataFrame(
        [
            {"dataset": "2015", "mass_GeV": 0.040, "eps2_up": 10.0},
            {"dataset": "2016", "mass_GeV": 0.040, "eps2_up": 20.0},
            {"dataset": "2021", "mass_GeV": 0.040, "eps2_up": 30.0},
        ]
    ).to_csv(input_csv, index=False)

    outpath = tmp_path / "projections" / "projected_unblinded_reach_eps2.png"
    result = runner.invoke(
        main,
        [
            "project-eps2-reach",
            "--input-csv",
            str(input_csv),
            "--output",
            str(outpath),
            "--scale-2015",
            "1",
            "--scale-2016",
            "10",
            "--scale-2021",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert outpath.exists()
    assert outpath.with_suffix(".pdf").exists()
    assert outpath.with_suffix(".csv").exists()
