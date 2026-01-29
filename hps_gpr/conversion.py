"""Conversions between signal amplitude A and epsilon^2."""

import math
from typing import TYPE_CHECKING

import numpy as np

from .dataset import ALPHA_EM

if TYPE_CHECKING:
    from .dataset import DatasetConfig


def epsilon2_from_A(
    ds: "DatasetConfig", mass: float, A: float, integral_density: float
) -> float:
    """Convert signal amplitude A to epsilon^2.

    Uses the formula:
        epsilon^2 = (2 * alpha_em * A) / (3 * pi * m * f_rad * rho)

    where rho is the integral density (counts per GeV).

    Args:
        ds: Dataset configuration
        mass: Signal mass (GeV)
        A: Signal amplitude
        integral_density: Counts per GeV in signal region

    Returns:
        epsilon^2 value, or NaN if conversion is not possible
    """
    frad = ds.frad(mass)

    if (
        (not np.isfinite(frad))
        or frad <= 0
        or (not np.isfinite(integral_density))
        or integral_density <= 0
    ):
        return float("nan")

    return float(
        (2.0 * ALPHA_EM * float(A))
        / (3.0 * math.pi * float(mass) * frad * integral_density)
    )


def A_from_epsilon2(
    ds: "DatasetConfig", mass: float, eps2: float, integral_density: float
) -> float:
    """Convert epsilon^2 to signal amplitude A.

    Uses the formula:
        A = epsilon^2 * (3 * pi * m * f_rad * rho) / (2 * alpha_em)

    where rho is the integral density (counts per GeV).

    Args:
        ds: Dataset configuration
        mass: Signal mass (GeV)
        eps2: epsilon^2 value
        integral_density: Counts per GeV in signal region

    Returns:
        Signal amplitude A, or NaN if conversion is not possible
    """
    frad = ds.frad(mass)

    if (
        (not np.isfinite(frad))
        or frad <= 0
        or (not np.isfinite(integral_density))
        or integral_density <= 0
    ):
        return float("nan")

    return float(
        float(eps2)
        * (3.0 * math.pi * float(mass) * frad * integral_density)
        / (2.0 * ALPHA_EM)
    )
