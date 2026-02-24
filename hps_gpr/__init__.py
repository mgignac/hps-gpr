"""HPS Gaussian Process Regression analysis package."""

__version__ = "0.1.0"

from .config import Config, load_config
from .dataset import DatasetConfig, make_datasets, ALPHA_EM
from .validation import validate_datasets
from .gpr import fit_gpr, predict_counts_from_log_gpr
from .template import build_template, cls_limit_for_amplitude, cls_limit_for_template
from .statistics import p0_from_blind_vectors, fit_A_profiled_gaussian
from .io import BlindPrediction, estimate_background_for_dataset
from .conversion import epsilon2_from_A, A_from_epsilon2
from .evaluation import (
    SingleResult, CombinedResult,
    evaluate_single_dataset, evaluate_combined,
    active_datasets_for_mass,
    combined_cls_limit_epsilon2_from_vectors,
)
from .scan import run_scan, union_scan_grid
from .bands import expected_ul_bands_for_dataset, expected_ul_bands_for_combination
from .injection import inject_counts, run_injection_extraction, run_injection_extraction_toys, summarize_injection_grid

__all__ = [
    # Config
    "Config",
    "load_config",
    # Dataset
    "DatasetConfig",
    "make_datasets",
    "ALPHA_EM",
    # Validation
    "validate_datasets",
    # GPR
    "fit_gpr",
    "predict_counts_from_log_gpr",
    # Template
    "build_template",
    "cls_limit_for_amplitude",
    "cls_limit_for_template",
    # Statistics
    "p0_from_blind_vectors",
    "fit_A_profiled_gaussian",
    # IO
    "BlindPrediction",
    "estimate_background_for_dataset",
    # Conversion
    "epsilon2_from_A",
    "A_from_epsilon2",
    # Evaluation
    "SingleResult",
    "CombinedResult",
    "evaluate_single_dataset",
    "evaluate_combined",
    "active_datasets_for_mass",
    "combined_cls_limit_epsilon2_from_vectors",
    # Scan
    "run_scan",
    "union_scan_grid",
    # Bands
    "expected_ul_bands_for_dataset",
    "expected_ul_bands_for_combination",
    # Injection
    "inject_counts",
    "run_injection_extraction",
    "run_injection_extraction_toys",
    "summarize_injection_grid",
]
