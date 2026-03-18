import json
from types import SimpleNamespace

from matplotlib.axes import Axes
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
        blind_mask=np.asarray(blind_mask, bool),
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
    assert "Zhat" in payload


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
    assert all("Zhat" in row for row in payload["per_dataset"])


def test_three_channel_combined_extraction_display_suite_writes_outputs(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 10.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (18.0, None))
    monkeypatch.setattr(exd, "combined_cls_limit_epsilon2", lambda mass, ds_list, preds, config: 5.5e-10)

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=True,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="combined",
        extraction_display_dataset_keys=["2015", "2016", "2021"],
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[3.0],
        extraction_display_refit_gp_on_toy=False,
    )

    written = run_extraction_display_suite(cfg)

    assert len(written) == 1
    png = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z3p0.png"
    meta = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z3p0.json"
    assert png.exists()
    assert meta.exists()
    payload = json.loads(meta.read_text())
    assert payload["datasets"] == ["2015", "2016", "2021"]
    assert len(payload["per_dataset"]) == 3


def test_combined_extraction_display_accepts_dataset_override(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 10.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (18.0, None))
    monkeypatch.setattr(exd, "combined_cls_limit_epsilon2", lambda mass, ds_list, preds, config: 5.5e-10)

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=True,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="combined",
        extraction_display_dataset_keys=["2015", "2016", "2021"],
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[3.0],
        extraction_display_refit_gp_on_toy=False,
    )

    run_extraction_display_suite(
        cfg,
        dataset_key="combined",
        dataset_keys=["2015", "2021"],
    )

    meta = tmp_path / "extraction_display" / "combined" / "extract_display_combined_m040MeV_z3p0.json"
    payload = json.loads(meta.read_text())
    assert payload["datasets"] == ["2015", "2021"]
    assert len(payload["per_dataset"]) == 2


def test_single_extraction_display_draws_residual_band_and_harmonized_ylabel(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    fill_calls = []
    ylabels = []
    orig_fill_between = Axes.fill_between
    orig_set_ylabel = Axes.set_ylabel

    def _record_fill_between(self, *args, **kwargs):
        fill_calls.append(True)
        return orig_fill_between(self, *args, **kwargs)

    def _record_set_ylabel(self, label, *args, **kwargs):
        ylabels.append(label)
        return orig_set_ylabel(self, label, *args, **kwargs)

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 14.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (21.0, None))
    monkeypatch.setattr(Axes, "fill_between", _record_fill_between)
    monkeypatch.setattr(Axes, "set_ylabel", _record_set_ylabel)

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

    exd.run_extraction_display_suite(cfg)

    assert fill_calls
    assert "Signal counts / bin" in ylabels


def test_extraction_display_reuses_mass_context_across_sigma_levels(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    calls = []

    def _counted_prediction(ds, mass, config, train_exclude_nsigma=None):
        calls.append((str(ds.key), float(mass)))
        return _fake_prediction(ds, mass, config, train_exclude_nsigma=train_exclude_nsigma)

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _counted_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 14.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (21.0, None))

    cfg = Config(
        enable_2015=True,
        enable_2016=False,
        enable_2021=False,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="2015",
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[3.0, 5.0, 7.0],
        extraction_display_refit_gp_on_toy=False,
    )

    exd.run_extraction_display_suite(cfg)

    assert calls == [("2015", 0.04)]


def test_combined_extraction_display_draws_residual_band_and_reports_zhat(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    fill_calls = []
    ylabels = []
    text_blocks = []
    orig_fill_between = Axes.fill_between
    orig_set_ylabel = Axes.set_ylabel
    orig_text = Axes.text

    def _record_fill_between(self, *args, **kwargs):
        fill_calls.append(True)
        return orig_fill_between(self, *args, **kwargs)

    def _record_set_ylabel(self, label, *args, **kwargs):
        ylabels.append(label)
        return orig_set_ylabel(self, label, *args, **kwargs)

    def _record_text(self, x, y, s, *args, **kwargs):
        text_blocks.append(s)
        return orig_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 11.0 if pred.integral_density < 1.2e8 else 9.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (18.0, None))
    monkeypatch.setattr(exd, "combined_cls_limit_epsilon2", lambda mass, ds_list, preds, config: 6.5e-10)
    monkeypatch.setattr(Axes, "fill_between", _record_fill_between)
    monkeypatch.setattr(Axes, "set_ylabel", _record_set_ylabel)
    monkeypatch.setattr(Axes, "text", _record_text)

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

    exd.run_extraction_display_suite(cfg)

    assert fill_calls
    assert "Signal counts / bin" in ylabels
    assert any("Zhat=" in block for block in text_blocks)


def test_combined_extraction_display_reuses_mass_contexts_across_sigma_levels(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    calls = []

    def _counted_prediction(ds, mass, config, train_exclude_nsigma=None):
        calls.append((str(ds.key), float(mass)))
        return _fake_prediction(ds, mass, config, train_exclude_nsigma=train_exclude_nsigma)

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _counted_prediction)
    monkeypatch.setattr(exd, "_sigmaA_reference", lambda pred, mass, source="asimov", rng=None: 10.0)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (18.0, None))
    monkeypatch.setattr(exd, "combined_cls_limit_epsilon2", lambda mass, ds_list, preds, config: 5.5e-10)

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=True,
        output_dir=str(tmp_path),
        extraction_display_dataset_key="combined",
        extraction_display_dataset_keys=["2015", "2016", "2021"],
        extraction_display_masses_gev=[0.040],
        extraction_display_sigma_multipliers=[3.0, 5.0, 7.0],
        extraction_display_refit_gp_on_toy=False,
    )

    exd.run_extraction_display_suite(cfg)

    assert calls == [("2015", 0.04), ("2016", 0.04), ("2021", 0.04)]


def test_combined_observed_display_suite_writes_outputs(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    def _fake_fit_details(obs, mu, cov, tmpl, allow_negative=True):
        mu = np.asarray(mu, float)
        tmpl = np.asarray(tmpl, float)
        return {
            "b_fit": mu,
            "lambda_hat": mu + 0.3 * tmpl,
            "A_hat": 6.0,
            "sigma_A": 2.0,
            "success": True,
        }

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (16.0, None))
    monkeypatch.setattr(exd, "fit_A_profiled_gaussian_details", _fake_fit_details)
    monkeypatch.setattr(exd, "p0_profiled_gaussian_LRT", lambda obs, mu, cov, tmpl: (1.2e-3, 3.03, 9.18, {}))
    monkeypatch.setattr(
        exd,
        "evaluate_combined",
        lambda mass, ds_list, preds, config: SimpleNamespace(eps2_up=6.2e-10, p0_analytic=8.0e-4, Z_analytic=3.16),
    )

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=True,
        output_dir=str(tmp_path),
        extraction_display_dataset_keys=["2015", "2016", "2021"],
    )

    written = exd.run_observed_display_suite(
        cfg,
        mass=0.040,
        dataset_key="combined",
        dataset_keys=["2015", "2016", "2021"],
    )

    mass_dir = tmp_path / "observed_display" / "m040MeV"
    hero = mass_dir / "observed_display_combined.png"
    meta = mass_dir / "metadata_combined.json"
    assert hero.exists()
    assert meta.exists()
    assert len(written) == 7
    payload = json.loads(meta.read_text())
    assert payload["datasets"] == ["2015", "2016", "2021"]
    assert len(payload["per_dataset"]) == 3
    for ds_key in ["2015", "2016", "2021"]:
        assert (mass_dir / f"observed_context_{ds_key}.png").exists()
        assert (mass_dir / f"observed_zoom_{ds_key}.png").exists()


def test_observed_display_uses_signal_counts_ylabel(tmp_path, monkeypatch):
    import hps_gpr.extraction_display as exd

    ylabels = []
    orig_set_ylabel = Axes.set_ylabel

    def _record_set_ylabel(self, label, *args, **kwargs):
        ylabels.append(label)
        return orig_set_ylabel(self, label, *args, **kwargs)

    def _fake_fit_details(obs, mu, cov, tmpl, allow_negative=True):
        mu = np.asarray(mu, float)
        tmpl = np.asarray(tmpl, float)
        return {
            "b_fit": mu,
            "lambda_hat": mu + 0.3 * tmpl,
            "A_hat": 6.0,
            "sigma_A": 2.0,
            "success": True,
        }

    monkeypatch.setattr(exd, "estimate_background_for_dataset", _fake_prediction)
    monkeypatch.setattr(exd, "cls_limit_for_amplitude", lambda **kwargs: (16.0, None))
    monkeypatch.setattr(exd, "fit_A_profiled_gaussian_details", _fake_fit_details)
    monkeypatch.setattr(exd, "p0_profiled_gaussian_LRT", lambda obs, mu, cov, tmpl: (1.2e-3, 3.03, 9.18, {}))
    monkeypatch.setattr(
        exd,
        "evaluate_combined",
        lambda mass, ds_list, preds, config: SimpleNamespace(eps2_up=6.2e-10, p0_analytic=8.0e-4, Z_analytic=3.16),
    )
    monkeypatch.setattr(Axes, "set_ylabel", _record_set_ylabel)

    cfg = Config(
        enable_2015=True,
        enable_2016=True,
        enable_2021=True,
        output_dir=str(tmp_path),
        extraction_display_dataset_keys=["2015", "2016", "2021"],
    )

    exd.run_observed_display_suite(
        cfg,
        mass=0.040,
        dataset_key="combined",
        dataset_keys=["2015", "2016", "2021"],
    )

    assert "Signal counts / bin" in ylabels
