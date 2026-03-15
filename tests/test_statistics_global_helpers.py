import numpy as np

from hps_gpr.statistics import (
    global_p_from_max_q0_toys,
    max_q0_from_local_p0_curve,
    q0_from_local_p0,
)


def test_q0_from_local_p0_matches_standard_gaussian_points():
    pvals = np.array([0.5, 0.15865525393145707, 0.02275013194817921])
    q0 = q0_from_local_p0(pvals)
    assert np.allclose(q0, [0.0, 1.0, 4.0], atol=1e-6)


def test_max_q0_from_local_p0_curve_uses_largest_finite_value():
    pvals = np.array([0.5, np.nan, 1.3498980316300933e-03])
    q0max = max_q0_from_local_p0_curve(pvals)
    assert np.isclose(q0max, 9.0, atol=1e-6)


def test_global_p_from_max_q0_toys_uses_smoothed_empirical_tail():
    toys = np.array([0.1, 4.0, 9.0, 12.0, 20.0])
    p = global_p_from_max_q0_toys(9.0, toys)
    assert np.isclose(p, 4.0 / 6.0)
