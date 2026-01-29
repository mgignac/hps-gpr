"""Configuration management for HPS GPR analysis."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
import sklearn.gaussian_process as skgp


@dataclass
class DatasetPaths:
    """Paths to ROOT files for each dataset."""
    path_2015: str = ""
    path_2016: str = ""
    path_2021: str = ""
    path_2021_mc: str = ""


@dataclass
class HistogramNames:
    """Histogram names within ROOT files."""
    hist_2015: str = "invariant_mass"
    hist_2016: str = "h_Minv_General_Final_1"
    hist_2021: str = "unc_vtx_mass_hist"


@dataclass
class AnalysisRanges:
    """Analysis mass ranges (GeV) for each dataset."""
    range_2015: Tuple[float, float] = (0.015, 0.140)
    range_2016: Tuple[float, float] = (0.035, 0.190)
    range_2021: Tuple[float, float] = (0.035, 0.230)


@dataclass
class Config:
    """Complete configuration for HPS GPR analysis."""

    # Dataset ROOT paths
    path_2015: str = ""
    path_2016: str = ""
    path_2021: str = ""
    path_2021_mc: str = ""
    only_2021_mc: bool = False

    # Histogram names
    hist_2015: str = "invariant_mass"
    hist_2016: str = "h_Minv_General_Final_1"
    hist_2021: str = "unc_vtx_mass_hist"

    # Analysis ranges (GeV)
    range_2015: Tuple[float, float] = (0.015, 0.140)
    range_2016: Tuple[float, float] = (0.035, 0.190)
    range_2021: Tuple[float, float] = (0.035, 0.230)

    # Mass resolution sigma(m) polynomial coefficients
    # sigma(m) = sum_i coeffs[i] * m**i
    sigma_coeffs_2015: List[float] = field(
        default_factory=lambda: [-0.0000922283032152, 0.0532190838657]
    )
    sigma_coeffs_2016: List[float] = field(
        default_factory=lambda: [0.00038, 0.041, -0.27, 3.49, -11.11]
    )
    sigma_coeffs_2021: List[float] = field(
        default_factory=lambda: [0.00286957, -0.00851449, 0.25362319]
    )

    # Radiative fraction f_rad(m) polynomial coefficients
    frad_coeffs_2015: List[float] = field(default_factory=lambda: [0.085])
    frad_coeffs_2016: List[float] = field(
        default_factory=lambda: [-0.104070, 9.59977, -212.211, 2148.12, -10140.1, 18048.5]
    )
    frad_coeffs_2021: List[float] = field(
        default_factory=lambda: [-0.211, 10.5, -161.8, 1189.0, -4165.0, 5565.0]
    )

    # Dataset enable switches
    enable_2015: bool = True
    enable_2016: bool = False
    enable_2021: bool = False

    # Kernel hyperparameters
    kernel_constant_init: float = 1.0
    kernel_constant_bounds: Tuple[float, float] = (1e-8, 1e18)
    kernel_ls_init: float = 1.0
    kernel_ls_bounds: Tuple[float, float] = (0.001, 10.0)

    # Preprocessing knobs
    pre_log: bool = True
    pre_zero_alpha: float = 1.0
    alpha_model: str = "1/y"
    pre_alpha_first_n: int = 0
    pre_alpha_first_factor: float = 0.1

    # Scan settings
    mass_step_gev: float = 0.001
    blind_nsigma: float = 1.96
    neighborhood_rebin: int = 10
    n_restarts: int = 20

    # CLs settings
    cls_alpha: float = 0.05
    cls_mode: str = "toys"
    cls_num_toys: int = 100
    make_ul_bands: bool = True
    ul_bands_toys: int = 100

    # Injection + extraction settings
    inject_signal: bool = False
    inj_dataset_key: str = "2015"
    inj_masses_gev: List[float] = field(default_factory=lambda: [0.030, 0.060, 0.090])
    inj_strengths: List[int] = field(default_factory=lambda: [0, 100, 200, 500, 1000, 2000, 5000])
    inj_mode: str = "multinomial"
    extract_allow_negative: bool = True

    # Limit-band dataset selector
    run_limit_bands_on: str = "2015"
    make_eps2_bands: bool = True

    # Outputs
    output_dir: str = "outputs/hps_gpr"

    # Validation/debug controls
    debug_print: bool = True
    debug_max_errors: int = 10
    fail_fast: bool = False
    save_per_mass_folders: bool = True
    save_plots: bool = True
    save_fit_json: bool = True

    def get_kernel(self):
        """Build the sklearn kernel from config parameters."""
        return (
            skgp.kernels.ConstantKernel(
                self.kernel_constant_init, self.kernel_constant_bounds
            )
            * skgp.kernels.RBF(
                length_scale=self.kernel_ls_init,
                length_scale_bounds=self.kernel_ls_bounds,
            )
        )

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Handle tuple fields that come as lists from YAML
    tuple_fields = [
        "range_2015",
        "range_2016",
        "range_2021",
        "kernel_constant_bounds",
        "kernel_ls_bounds",
    ]
    for field_name in tuple_fields:
        if field_name in data and isinstance(data[field_name], list):
            data[field_name] = tuple(data[field_name])

    return Config(**data)


def save_config(config: Config, path: str) -> None:
    """Save configuration to a YAML file."""
    data = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        # Convert tuples to lists for YAML
        if isinstance(value, tuple):
            value = list(value)
        data[field_name] = value

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
