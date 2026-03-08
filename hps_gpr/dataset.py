"""Dataset configuration and utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

ALPHA_EM = 1.0 / 137.0  # fine structure constant


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    key: str
    label: str
    root_path: str
    hist_name: str
    m_low: float
    m_high: float
    sigma_coeffs: List[float]
    frad_coeffs: List[float]
    enabled: bool = True
    radiative_penalty_on: bool = False
    radiative_penalty_frac: float = 0.0

    # Optional piecewise linear sigma(m) tail, used for 2016.
    sigma_tail_m0: Optional[float] = None
    sigma_tail_slope_floor: float = 0.0
    sigma_tail_slope_override: Optional[float] = None

    # Optional GP training range (separate from scan range).
    # If None, falls back to m_low/m_high.
    data_low: Optional[float] = None
    data_high: Optional[float] = None

    def _sigma_poly(self, m: float) -> float:
        return float(sum(c * (m**i) for i, c in enumerate(self.sigma_coeffs)))

    def _sigma_poly_deriv(self, m: float) -> float:
        return float(sum(i * c * (m ** (i - 1)) for i, c in enumerate(self.sigma_coeffs) if i > 0))

    def sigma(self, m: float) -> float:
        """Compute mass resolution sigma(m), with optional linear tail extension."""
        if self.sigma_tail_m0 is None or m <= float(self.sigma_tail_m0):
            return self._sigma_poly(float(m))

        m0 = float(self.sigma_tail_m0)
        sigma_m0 = self._sigma_poly(m0)
        slope = self._sigma_poly_deriv(m0)
        if self.sigma_tail_slope_override is not None:
            slope = float(self.sigma_tail_slope_override)
        slope = max(float(slope), float(self.sigma_tail_slope_floor))
        return float(sigma_m0 + slope * (float(m) - m0))

    def frad(self, m: float) -> float:
        """Compute radiative fraction f_rad(m) from polynomial coefficients."""
        return float(sum(c * (m**i) for i, c in enumerate(self.frad_coeffs)))

    def frad_penalty_scale(self) -> float:
        """Multiplicative sensitivity penalty applied to f_rad when enabled."""
        if not bool(self.radiative_penalty_on):
            return 1.0
        frac = float(self.radiative_penalty_frac)
        if frac <= 0:
            return 1.0
        return float(max(0.0, 1.0 - frac))

    def frad_effective(self, m: float) -> float:
        """Effective radiative fraction after any configured penalty."""
        return float(self.frad(float(m)) * self.frad_penalty_scale())


def poly_str(coeffs: List[float], name: str = "p") -> str:
    """Format polynomial coefficients as a string."""
    return name + "(m)=" + " + ".join(
        [f"{c:.3g}*m^{i}" for i, c in enumerate(coeffs)]
    )


def make_datasets(config: "Config") -> Dict[str, DatasetConfig]:
    """Create dataset configurations from the global config.

    Args:
        config: Global configuration object

    Returns:
        Dictionary mapping dataset keys to DatasetConfig objects
    """
    # Determine 2021 path based on MC mode
    p2021 = config.path_2021_mc if config.only_2021_mc else config.path_2021

    ds = {
        "2015": DatasetConfig(
            key="2015",
            label="HPS 2015",
            root_path=config.path_2015,
            hist_name=config.hist_2015,
            m_low=config.range_2015[0],
            m_high=config.range_2015[1],
            sigma_coeffs=config.sigma_coeffs_2015,
            frad_coeffs=config.frad_coeffs_2015,
            enabled=config.enable_2015 and (not config.only_2021_mc),
            radiative_penalty_on=config.radiative_penalty_on,
            radiative_penalty_frac=float(config.radiative_penalty_frac_2015),
            data_low=(config.data_range_2015[0] if config.data_range_2015 is not None else None),
            data_high=(config.data_range_2015[1] if config.data_range_2015 is not None else None),
        ),
        "2016": DatasetConfig(
            key="2016",
            label="HPS 2016",
            root_path=config.path_2016,
            hist_name=config.hist_2016,
            m_low=config.range_2016[0],
            m_high=config.range_2016[1],
            sigma_coeffs=config.sigma_coeffs_2016,
            frad_coeffs=config.frad_coeffs_2016,
            enabled=config.enable_2016 and (not config.only_2021_mc),
            radiative_penalty_on=config.radiative_penalty_on,
            radiative_penalty_frac=float(config.radiative_penalty_frac_2016),
            sigma_tail_m0=config.sigma_tail_m0_2016,
            sigma_tail_slope_floor=config.sigma_tail_slope_floor_2016,
            sigma_tail_slope_override=config.sigma_tail_slope_override_2016,
            data_low=(config.data_range_2016[0] if config.data_range_2016 is not None else None),
            data_high=(config.data_range_2016[1] if config.data_range_2016 is not None else None),
        ),
        "2021": DatasetConfig(
            key="2021",
            label="HPS 2021" + (" (MC)" if config.only_2021_mc else ""),
            root_path=p2021,
            hist_name=config.hist_2021,
            m_low=config.range_2021[0],
            m_high=config.range_2021[1],
            sigma_coeffs=config.sigma_coeffs_2021,
            frad_coeffs=config.frad_coeffs_2021,
            enabled=config.enable_2021,
            radiative_penalty_on=config.radiative_penalty_on,
            radiative_penalty_frac=float(config.radiative_penalty_frac_2021),
            data_low=(config.data_range_2021[0] if config.data_range_2021 is not None else None),
            data_high=(config.data_range_2021[1] if config.data_range_2021 is not None else None),
        ),
    }

    # Filter to only enabled datasets
    return {k: v for k, v in ds.items() if v.enabled}


def print_datasets(datasets: Dict[str, DatasetConfig]) -> None:
    """Print information about enabled datasets."""
    print("Enabled datasets:", list(datasets.keys()))
    for k, d in datasets.items():
        print(
            f"  {k}: range=[{d.m_low:.3f},{d.m_high:.3f}]  "
            f"sigma: {poly_str(d.sigma_coeffs, 'σ')}  "
            f"frad: {poly_str(d.frad_coeffs, 'f')}  "
            f"penalty={'on' if d.radiative_penalty_on else 'off'}({100.0 * d.radiative_penalty_frac:.1f}%)"
        )
