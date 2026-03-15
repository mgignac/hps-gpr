"""HPS Gaussian Process Regression analysis package."""

__version__ = "0.1.0"

from .config import Config, load_config
from .dataset import DatasetConfig, make_datasets, ALPHA_EM
from .validation import validate_datasets
from .gpr import fit_gpr, predict_counts_from_log_gpr
from .template import (
    build_template,
    build_full_template,
    build_window_template_from_full,
    cls_limit_for_amplitude,
    cls_limit_for_template,
)
from .plotting import (
    set_plot_style, set_injection_plot_style,
    plot_full_range, plot_blind_window, plot_s_over_b,
    plot_scan_diagnostic_panels,
    plot_ul_bands, plot_ul_pvalues,
    plot_analytic_p0, plot_Z_local_global,
    plot_linearity, plot_bias_vs_injected_strength, plot_pull_width, plot_coverage,
    plot_z_calibration_residual, plot_delta_z_minus_pull_vs_injected_sigma,
    plot_eps2_curves, plot_bands,
)
from .statistics import (
    p0_from_blind_vectors,
    fit_A_profiled_gaussian,
    _p_local_from_global_summary,
    q0_from_local_p0,
    max_q0_from_local_p0_curve,
    global_p_from_max_q0_toys,
)
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
from .injection import (
    inject_counts,
    run_injection_extraction,
    run_injection_extraction_toys,
    run_injection_extraction_streaming,
    run_injection_extraction_streaming_combined,
    summarize_injection_grid,
)
from .extraction_display import (
    make_single_extraction_display,
    make_combined_extraction_display,
    make_single_observed_display,
    make_combined_observed_display,
    run_extraction_display_suite,
    run_observed_display_suite,
)

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
    "build_full_template",
    "build_window_template_from_full",
    "cls_limit_for_amplitude",
    "cls_limit_for_template",
    # Statistics
    "p0_from_blind_vectors",
    "fit_A_profiled_gaussian",
    "_p_local_from_global_summary",
    "q0_from_local_p0",
    "max_q0_from_local_p0_curve",
    "global_p_from_max_q0_toys",
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
    "run_injection_extraction_streaming",
    "run_injection_extraction_streaming_combined",
    "summarize_injection_grid",
    "make_single_extraction_display",
    "make_combined_extraction_display",
    "make_single_observed_display",
    "make_combined_observed_display",
    "run_extraction_display_suite",
    "run_observed_display_suite",
    # Plotting
    "set_plot_style",
    "set_injection_plot_style",
    "plot_full_range",
    "plot_blind_window",
    "plot_s_over_b",
    "plot_scan_diagnostic_panels",
    "plot_ul_bands",
    "plot_ul_pvalues",
    "plot_analytic_p0",
    "plot_Z_local_global",
    "plot_linearity",
    "plot_bias_vs_injected_strength",
    "plot_pull_width",
    "plot_coverage",
    "plot_z_calibration_residual",
    "plot_delta_z_minus_pull_vs_injected_sigma",
    "plot_eps2_curves",
    "plot_bands",
]
