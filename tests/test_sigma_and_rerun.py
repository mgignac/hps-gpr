import math

from hps_gpr.dataset import DatasetConfig
from hps_gpr.slurm import get_task_ids_for_masses


def test_sigma_2016_piecewise_linear_tail_behavior():
    ds = DatasetConfig(
        key="2016",
        label="HPS 2016",
        root_path="dummy.root",
        hist_name="h",
        m_low=0.035,
        m_high=0.190,
        sigma_coeffs=[0.00038, 0.041, -0.27, 3.49, -11.11],
        frad_coeffs=[0.05],
        enabled=True,
        sigma_tail_m0=0.18,
        sigma_tail_slope_floor=0.0,
        sigma_tail_slope_override=0.0239,
    )

    m0 = 0.18
    below = ds.sigma(0.17)
    at = ds.sigma(m0)
    above = ds.sigma(0.19)

    # continuity at transition and linear tail above m0
    assert math.isclose(at, ds._sigma_poly(m0), rel_tol=0.0, abs_tol=1e-12)
    expected_above = at + 0.0239 * (0.19 - m0)
    assert math.isclose(above, expected_above, rel_tol=0.0, abs_tol=1e-12)
    assert below > 0.0


def test_get_task_ids_for_masses_two_points():
    ds = {
        "2016": DatasetConfig(
            key="2016",
            label="HPS 2016",
            root_path="dummy.root",
            hist_name="h",
            m_low=0.035,
            m_high=0.190,
            sigma_coeffs=[0.00038, 0.041, -0.27, 3.49, -11.11],
            frad_coeffs=[0.05],
        )
    }

    # Grid has 156 masses (35..190 MeV, inclusive, 1 MeV step).
    # With 156 tasks, each mass maps to one task.
    tids = get_task_ids_for_masses(ds, mass_step=0.001, n_tasks=156, masses_gev=[0.037, 0.048])
    assert tids == [2, 13]
