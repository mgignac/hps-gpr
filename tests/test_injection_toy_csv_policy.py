from types import SimpleNamespace

import numpy as np

from hps_gpr.config import Config
from hps_gpr.dataset import DatasetConfig
from hps_gpr.injection import run_injection_extraction_toys


def _make_dataset():
    return DatasetConfig(
        key="2015",
        label="HPS 2015",
        root_path="dummy.root",
        hist_name="h",
        m_low=0.020,
        m_high=0.130,
        sigma_coeffs=[0.001],
        frad_coeffs=[0.1],
    )


def _install_fast_injection_mocks(monkeypatch):
    import hps_gpr.injection as inj

    def fake_estimate_background_for_dataset(ds, m, config):
        return SimpleNamespace(
            edges=np.array([0.0, 1.0, 2.0]),
            sigma_val=1.0,
            mu=np.array([2.0, 3.0]),
            cov=np.eye(2),
            edges_full=np.array([0.0, 1.0, 2.0]),
            x_full=np.array([0.5, 1.5]),
            blind=(0.0, 2.0),
            train_exclude_nsigma=1.64,
            mu_full=np.array([2.0, 3.0]),
            sigma_x=1.0,
        )

    monkeypatch.setattr(inj, "estimate_background_for_dataset", fake_estimate_background_for_dataset)
    monkeypatch.setattr(inj, "build_template", lambda edges, mass, sigma: np.array([0.6, 0.4]))
    monkeypatch.setattr(inj, "_sigmaA_reference", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(inj, "draw_bkg_mvn_nonneg", lambda mu, cov, n, rng, method, max_tries: np.tile(mu, (n, 1)))
    monkeypatch.setattr(
        inj,
        "fit_A_profiled_gaussian",
        lambda obs, mu, cov, tmpl_win, allow_negative: {
            "A_hat": float(np.sum(obs)),
            "sigma_A": 2.0,
            "success": True,
            "nll": 0.0,
        },
    )


def test_run_injection_extraction_toys_skips_writing_toy_csv_when_disabled(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=False)

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[0.0, 1.0],
        n_toys=2,
    )

    assert len(df) == 4
    assert not (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()


def test_run_injection_extraction_toys_writes_toy_csv_when_enabled(tmp_path, monkeypatch):
    _install_fast_injection_mocks(monkeypatch)
    cfg = Config(output_dir=str(tmp_path), inj_write_toy_csv=True)

    df = run_injection_extraction_toys(
        _make_dataset(),
        cfg,
        masses=[0.05],
        strengths=[0.0],
        n_toys=1,
    )

    assert len(df) == 1
    assert (tmp_path / "injection_extraction" / "inj_extract_toys_2015.csv").exists()
