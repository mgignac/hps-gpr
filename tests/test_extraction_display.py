import json

import numpy as np

from hps_gpr.config import Config
from hps_gpr.extraction_display import run_extraction_display_suite
from hps_gpr.io import BlindPrediction


def _fake_prediction(ds, mass, config, train_exclude_nsigma=None):
    sigma = 0.0019 if ds.key == "2015" else 0.0016
    edges_full = np.linspace(float(mass) - 0.012, float(mass) + 0.012, 25)
    x_full = 0.5 * (edges_full[:-1] + edges_full[1:])
    blind = (
        float(mass) - float(config.blind_nsigma) * sigma,
        float(mass) + float(config.blind_nsigma) * sigma,
    )
    mu_full = (28.0 + (6.0 if ds.key == "2016" else 0.0)) + 12.0 * np.exp(-0.5 * ((x_full - float(mass)) / 0.0075) ** 2)
    y_full = np.round(mu_full).astype(int)
    blind_mask = (x_full >= blind[0]) & (x_full <= blind[1])
    idx = np.where(blind_mask)[0]
    edges = edges_full[idx[0] : idx[-1] + 2]
    mu = mu_full[blind_mask]
    cov = np.diag(np.clip(mu, 1.0, None))
    obs = np.round(mu).astype(int)
    tn = float(train_exclude_nsigma if train_exclude_nsigma is not None else config.blind_nsigma)
    return BlindPrediction(
        mu=np.asarray(mu, float),
        cov=np.asarray(cov, float),
        obs=np.asarray(obs, int),
        edges=np.asarray(edges, float),
        sigma_val=float(sigma),
        blind=blind,
        x_full=np.asarray(x_full, float),
        y_full=np.asarray(y_full, int),
        mu_full=np.asarray(mu_full, float),
        edges_full=np.asarray(edges_full, float),
        integral_density=1.1e8 if ds.key == "2015" else 1.4e8,
        blind_train=(float(mass) - tn * sigma, float(mass) + tn * sigma),
    )


def test_single_extraction_display_suite_writes_outputs(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 14.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (21.0, None))

    cfg = Config(
        enable_2015=True,
        enable_2016=False,
        enable_2021=False,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="2015",
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[3.0],
        extraction_display_refit_gp_on_toy=False,
    )

    written = run_extraction_display_suite(cfg)

    assert len(written) == 1
    png = tmp_path / "extraction_display" / "2015" / "extract_display_2015_m040MeV_z3p0.png"
    pdf = tmp_path / "extraction_display" / "2015" / "extract_display_2015_m040MeV_z3p0.pdf"
    meta = tmp_path / "extraction_display" / "2015" / "extract_display_2015_m040MeV_z3p0.json"
    assert png.exists()
    assert pdf.exists()
    assert meta.exists()
    payload = json.loads(meta.read_text())
    assert payload["dataset"] == "2015"
    assert payload["inj_nsigma"] == 3.0


def test_combined_extraction_display_suite_writes_outputs(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 11.0 if pred.integral_density < 1.2e8 else 9.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (18.0, None))
    monkeypatch.setattr(exd, "combined_cls_limit_epsilon2", lambda mass, ds_list, preds, config: 6.5e-10)

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=False,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="combined",
        extraction_display_dataset_keys=["2015", "2016"],
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[5.0],
        extraction_display_refit_gp_on_toy=False,
    )

    written = run_extraction_display_suite(cfg)

    assert len(written) == 1
    png = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z5p0.png"
    pdf = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z5p0.pdf"
    meta = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z5p0.json"
    assert png.exists()
    assert pdf.exists()
    assert meta.exists()
    payload = json.loads(meta.read_text())
    assert payload["inj_nsigma_combined"] == 5.0
    assert payload["datasets"] == ["2015", "2016"]
    assert len(payload["per_dataset"]) == 2
