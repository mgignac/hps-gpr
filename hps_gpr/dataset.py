"""Dataset configuration and utilities."""

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

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

    def sigma(self, m: float) -> float:
        """Compute mass resolution sigma(m) from polynomial coefficients."""
        return float(sum(c * (m**i) for i, c in enumerate(self.sigma_coeffs)))

    def frad(self, m: float) -> float:
        """Compute radiative fraction f_rad(m) from polynomial coefficients."""
        return float(sum(c * (m**i) for i, c in enumerate(self.frad_coeffs)))


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
            f"frad: {poly_str(d.frad_coeffs, 'f')}"
        )
