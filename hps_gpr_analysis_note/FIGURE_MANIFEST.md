# Figure Manifest

This manifest maps the figure files bundled directly inside
`hps_gpr_analysis_note/` to their original local sources and records whether each item
is a stable copied asset, a provisional pre-rerun product, or a true placeholder. The
LaTeX note references only the note-local paths listed in the `Bundle path` column, so
the entire `hps_gpr_analysis_note/` directory can be uploaded to Overleaf as a
standalone project.

## Status conventions

- `stable`: copied asset whose interpretation is not materially changed by the updated
  leakage-retaining signal-template convention.
- `provisional`: copied asset generated under the pre-rerun workflow and retained for
  structure, comparison, or baseline diagnostics.
- `placeholder`: no source file exists yet; the note compiles through the
  `\graphicorplaceholder` macro.
- `generated placeholder`: note-local stand-in created to reserve the final upload
  location for a figure that still needs to be regenerated.

## Published reference figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `published_reference_figs/hps2015_published_limit.png` | Cropped from `https://arxiv.org/pdf/1807.11530.pdf` | stable | Public 2015 prompt-search limit figure used for historical context. |
| `published_reference_figs/hps2016_published_prompt_limit.png` | Cropped from `https://arxiv.org/pdf/2212.10629.pdf` | stable | Public 2016 prompt-search upper-limit figure. |
| `published_reference_figs/hps2016_published_prompt_pvalue.png` | Cropped from `https://arxiv.org/pdf/2212.10629.pdf` | stable | Public 2016 prompt-search local/global significance figure. |

## Upper-limit figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `upper_limit_figs/2015/ul_bands_signal_yield_obsexp.png` | `summary_combined_2015/ul_bands_signal_yield_obsexp.png` | provisional | Pre-rerun 2015 signal-yield limit bands. |
| `upper_limit_figs/2015/ul_bands_eps2_obsexp.png` | `summary_combined_2015/ul_bands_eps2_obsexp.png` | provisional | Pre-rerun 2015 `\eps^2` limit bands. |
| `upper_limit_figs/2016_10pct/ul_bands_signal_yield_obsexp.png` | `summary_combined_2016/ul_bands_signal_yield_obsexp.png` | provisional | Pre-rerun 2016 10% signal-yield limit bands. |
| `upper_limit_figs/2016_10pct/ul_bands_eps2_obsexp.png` | `summary_combined_2016/ul_bands_eps2_obsexp.png` | provisional | Pre-rerun 2016 10% `\eps^2` limit bands. |
| `upper_limit_figs/2015_2016_combined/ul_bands_signal_yield_obsexp.png` | Generated note-local PNG stand-in | generated placeholder | Reserve this exact final Overleaf path for the combined signal-yield bands after the rerun. |
| `upper_limit_figs/2015_2016_combined/ul_bands_eps2_obsexp.png` | `summary_combined_all/ul_bands_eps2_obsexp.png` | provisional | Pre-rerun 2015+2016 combined `\eps^2` limit bands. |
| `upper_limit_figs/2021_0pt03pct/ul_bands_signal_yield_obsexp.png` | not yet exported | placeholder | Planned 2021 0.03% signal-yield bands. |
| `upper_limit_figs/2021_0pt03pct/ul_bands_eps2_obsexp.png` | not yet exported | placeholder | Planned 2021 0.03% `\eps^2` bands. |
| `upper_limit_figs/2021_1pct/ul_bands_signal_yield_obsexp.png` | not yet exported | placeholder | Planned 2021 1% signal-yield bands. |
| `upper_limit_figs/2021_1pct/ul_bands_eps2_obsexp.png` | not yet exported | placeholder | Planned 2021 1% `\eps^2` bands. |
| `upper_limit_figs/2015_2016_2021_1pct_combined/ul_bands_signal_yield_obsexp.png` | not yet exported | placeholder | Planned 2015+2016+2021 1% combined signal-yield bands. |
| `upper_limit_figs/2015_2016_2021_1pct_combined/ul_bands_eps2_obsexp.png` | not yet exported | placeholder | Planned 2015+2016+2021 1% combined `\eps^2` bands. |
| `upper_limit_figs/first_round_unblinding_2015_2016_2021_10pct/ul_bands_signal_yield_obsexp.png` | not yet exported | placeholder | Planned first-round unblinding signal-yield bands. |
| `upper_limit_figs/first_round_unblinding_2015_2016_2021_10pct/ul_bands_eps2_obsexp.png` | not yet exported | placeholder | Planned first-round unblinding `\eps^2` bands. |

## Significance figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `significance_figs/2015/p0_analytic_local_global.png` | `summary_combined_2015/p0_analytic_local_global.png` | provisional | Pre-rerun 2015 local/global `p_0` bookkeeping. |
| `significance_figs/2015/Z_local_global.png` | `summary_combined_2015/Z_local_global.png` | provisional | Pre-rerun 2015 local/global `Z` bookkeeping. |
| `significance_figs/2016_10pct/p0_analytic_local_global.png` | `summary_combined_2016/p0_analytic_local_global.png` | provisional | Pre-rerun 2016 10% local/global `p_0` bookkeeping. |
| `significance_figs/2016_10pct/Z_local_global.png` | `summary_combined_2016/Z_local_global.png` | provisional | Pre-rerun 2016 10% local/global `Z` bookkeeping. |
| `significance_figs/2015_2016_combined/p0_analytic_local_global.png` | `summary_combined_all/p0_analytic_local_global.png` | provisional | Pre-rerun combined local/global `p_0` bookkeeping. |
| `significance_figs/2015_2016_combined/Z_local_global.png` | `summary_combined_all/Z_local_global.png` | provisional | Pre-rerun combined local/global `Z` bookkeeping. |

## Fit-example figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `fit_example_figs/2015/m030MeV_blind_fit.png` | `addl_plots/m030MeV/2015/blind_fit.png` | stable | Representative 2015 blind-window fit display. |
| `fit_example_figs/2016/m058MeV_blind_fit.png` | `addl_plots/m058MeV/2016/blind_fit.png` | stable | Representative 2016 blind-window fit display with negative `\hat A`. |

## Combined-search figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `combined_search_figs/extract_display_2015_m025MeV_z7p0.png` | `z7 copy/extract_display_2015_m025MeV_z7p0 copy.png` | provisional | Individual 2015 common-signal extraction display. |
| `combined_search_figs/extract_display_2016_m080MeV_z7p0.png` | `z7 copy/extract_display_2016_m080MeV_z7p0.png` | provisional | Individual 2016 common-signal extraction display. |
| `combined_search_figs/extract_display_combined_m040MeV_z7p0.png` | `z7 copy/extraction_display_combined/combined/extract_display_combined_m040MeV_z7p0.png` | provisional | Combined 40 MeV display retained as an illustrative diagnostic. |
| `combined_search_figs/combined_search_power_scenarios.png` | `injection_summary_9/combined_search_power_scenarios.png` | provisional | Scenario-level combined search-power study. |
| `combined_search_figs/combined_search_power_constituent_pvalues_5sigma.png` | `injection_summary_9/combined_search_power_constituent_pvalues_5sigma.png` | provisional | Constituent-dataset significance requirements for a target combined excess. |
| `combined_search_figs/combined_signal_allocation_m040MeV.png` | `injection_summary_9/combined_signal_allocation_m040MeV.png` | provisional | Signal-allocation study at 40 MeV. |
| `combined_search_figs/combined_signal_allocation_m080MeV.png` | `injection_summary_9/combined_signal_allocation_m080MeV.png` | provisional | Signal-allocation study at 80 MeV. |
| `combined_search_figs/combined_signal_allocation_m115MeV.png` | `injection_summary_9/combined_signal_allocation_m115MeV.png` | provisional | Signal-allocation study at 115 MeV. |

## Injection and extraction validation figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `injection_extraction_figs/linearity_all.png` | `injection_summary_9/linearity_all.png` | provisional | Cross-dataset extraction linearity summary. |
| `injection_extraction_figs/bias_all.png` | `injection_summary_9/bias_all.png` | provisional | Cross-dataset extraction bias summary. |
| `injection_extraction_figs/pull_width_all.png` | `injection_summary_9/pull_width_all.png` | provisional | Cross-dataset pull-width summary. |
| `injection_extraction_figs/coverage_all.png` | `injection_summary_9/coverage_all.png` | provisional | Cross-dataset empirical coverage summary. |
| `injection_extraction_figs/heatmap_pull_mean_2015.png` | `injection_summary_9/heatmap_pull_mean_2015.png` | provisional | 2015 mean-pull heatmap. |
| `injection_extraction_figs/heatmap_pull_mean_2016.png` | `injection_summary_9/heatmap_pull_mean_2016.png` | provisional | 2016 mean-pull heatmap. |
| `injection_extraction_figs/heatmap_pull_mean_combined.png` | `injection_summary_9/heatmap_pull_mean_combined.png` | provisional | Combined mean-pull heatmap. |
| `injection_extraction_figs/heatmap_pull_width_2015.png` | `injection_summary_9/heatmap_pull_width_2015.png` | provisional | 2015 pull-width heatmap. |
| `injection_extraction_figs/heatmap_pull_width_2016.png` | `injection_summary_9/heatmap_pull_width_2016.png` | provisional | 2016 pull-width heatmap. |
| `injection_extraction_figs/heatmap_pull_width_combined.png` | `injection_summary_9/heatmap_pull_width_combined.png` | provisional | Combined pull-width heatmap. |
| `injection_extraction_figs/z_calibration_residual_comparison.png` | `injection_summary_9/z_calibration_residual_comparison.png` | provisional | Significance-calibration residual comparison. |
| `injection_extraction_figs/pull_vs_mass_combined.png` | `injection_summary_9/pull_vs_mass_combined.png` | provisional | Combined pull evolution versus mass. |

## Toy-generation figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `toy_generation_figs/good_fit_2021_0pt03_pct.png` | `good_fit_2021_0pt03_pct.png` | stable | Representative analytic functional-form fit for the 2021 0.03% subset. |
| `toy_generation_figs/good_fit_2015_placeholder.png` | not yet exported | placeholder | Planned analytic-fit toy seed for 2015. |
| `toy_generation_figs/good_fit_2016_placeholder.png` | not yet exported | placeholder | Planned analytic-fit toy seed for 2016. |

## Kernel and length-scale figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `kernel_lengthscale_figs/ls_hi_bounds_logm_overlay.png` | `ls_bound_plots/ls_hi_bounds_logm_overlay.png` | stable | Length-scale upper bound in log-mass units. |
| `kernel_lengthscale_figs/dm_equiv_from_lhi_overlay.png` | `ls_bound_plots/dm_equiv_from_lhi_overlay.png` | stable | Equivalent mass-scale interpretation of the upper bound. |
| `kernel_lengthscale_figs/ls_hi_vs_k_2021.png` | `ls_bound_plots/ls_hi_vs_k_2021.png` | stable | Additional 2021 length-scale study retained for later use. |

## Reserved folders currently without exported figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `dataset_summary_figs/` | not yet populated | placeholder | Reserved for dataset overview plots or scan-range schematics if needed later. |

## Notes

- Every path referenced by the LaTeX source is relative to the root of
  `hps_gpr_analysis_note/`; there are no remaining references to repository-root figure
  locations.
- The `provisional` label is used intentionally for methodology-sensitive outputs that
  should be regenerated after the leakage-retaining rerun.
- The placeholder macro in `main.tex` allows the note to compile even when future
  figure exports have not yet been added.
