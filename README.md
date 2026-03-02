# HPS GPR Analysis Package

A Python package for performing Gaussian Process Regression (GPR) based bump hunt analyses for dark photon searches in the Heavy Photon Search (HPS) experiment.

## Overview

This package implements a simultaneous bump hunt analysis across multiple HPS datasets (2015, 2016, 2021) using Gaussian Process Regression for background estimation. It provides:

- **Background estimation** using GPR with configurable kernels
- **Signal template** construction using binned Gaussian shapes
- **CLs upper limits** via asymptotic approximation or toy Monte Carlo
- **Combined fits** across multiple datasets in overlap regions
- **Expected limit bands** computation
- **Signal injection/extraction** studies for closure tests
- **SLURM integration** for batch processing on computing clusters

The package converts the original Jupyter notebook analysis into a modular, batch-ready Python package with a command-line interface.

## Installation

### Prerequisites

- Python 3.9 or higher

### Setup

1. Clone or download the repository:
   ```bash
   cd /path/to/HPS/GPR
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install hps_gpr in development mode:
   ```bash
   pip install -e .
   ```

5. Verify installation:
   ```bash
   hps-gpr --help
   ```

## Configuration

The analysis is configured via a YAML file. Copy the example configuration and modify it for your analysis:

```bash
cp config_example.yaml my_config.yaml
```

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `path_*` | Paths to ROOT files containing invariant mass histograms |
| `hist_*` | Histogram names within each ROOT file |
| `range_*` | Analysis mass ranges (GeV) for each dataset |
| `sigma_coeffs_*` | Mass resolution polynomial coefficients |
| `sigma_tail_*_2016` | Optional 2016 high-mass linear-tail controls for σ(m) |
| `frad_coeffs_*` | Radiative fraction polynomial coefficients |
| `enable_*` | Enable/disable individual datasets |
| `cls_*` | CLs calculation settings (alpha, mode, toys) |
| `output_dir` | Output directory for results |

### Example Configuration

```yaml
# Enable only 2015 dataset
enable_2015: true
enable_2016: false
enable_2021: false

# Paths to data
path_2015: "/data/hps/2015_invariant_mass.root"
hist_2015: "invariant_mass"
range_2015: [0.020, 0.130]

# CLs settings
cls_alpha: 0.05
cls_mode: "asymptotic"
cls_num_toys: 100

# Output
output_dir: "outputs/my_analysis"
```

## Command-Line Interface

The package provides the `hps-gpr` command with several subcommands:

### Run Full Mass Scan

```bash
# Full scan with all configured datasets
hps-gpr scan --config my_config.yaml

# Scan a specific mass range
hps-gpr scan --config my_config.yaml --mass-min 0.05 --mass-max 0.10

# Override output directory
hps-gpr scan --config my_config.yaml --output-dir results/run1/
```

### Smoke Test

Run a quick test on a single mass point to verify setup:

```bash
hps-gpr test --config my_config.yaml
```

### Expected Limit Bands

Compute expected upper limit bands for a specific dataset:

```bash
hps-gpr bands --config my_config.yaml --dataset 2015 --n-toys 100
```

### Re-run Selected Failed Mass Points

If only a few mass points failed in a SLURM production, re-run just the owning task(s) and overwrite those task outputs in place:

```bash
hps-gpr re-run --config config_2016_10pct_10k.yaml -m 37 -m 48
```

Compatibility alias matching existing workflow wording:

```bash
hps-gpr re-run-2016-bands --config config_2016_10pct_10k.yaml -mass 37 -mass 48
```

Masses are given in **MeV**; the command reuses the original mass-step and toy settings from the config.

### Injection/Extraction Study

Run signal injection and extraction closure tests:

```bash
hps-gpr inject --config my_config.yaml --dataset 2015 --masses 0.03,0.06,0.09 --n-toys 10000
```


### Signal-injection studies (copy/paste examples)

The repository supports injection/extraction studies for:
- **2015-only**
- **2016 10%-only**
- **2015+2016 combined**

Ready-to-run study configs live in `study_configs/` for both legacy 90% CL and new 95% CL settings (for blind widths 1.64 and 1.96).

```bash
# 2015 injection study (95% CL, blind width 1.96)
hps-gpr inject --config study_configs/config_2015_blind1p96_95CL_10k_injection.yaml --dataset 2015 --masses 0.020,0.040,0.060,0.080,0.100,0.120

# 2016 10% injection study (95% CL, blind width 1.96)
hps-gpr inject --config study_configs/config_2016_10pct_blind1p96_95CL_10k_injection.yaml --dataset 2016 --masses 0.040,0.060,0.080,0.100,0.140,0.180,0.210

# Combined 2015+2016 production scan + summary suite (95% CL)

# Combined injection/extraction matrix (single-process run)
# strengths in sigma_A: 1,2,3,5 ; masses in GeV: 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200
hps-gpr inject --config study_configs/config_2015_2016_combined_blind1p64_95CL_10k_injection.yaml --dataset combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths 1,2,3,5 --n-toys 10000

# Batch production: one job per (dataset, mass, strength)
# datasets include individual and combined extraction to compare behavior directly
hps-gpr slurm-gen-inject --config study_configs/config_2015_2016_combined_blind1p64_95CL_10k_injection.yaml --datasets 2015,2016,combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths 1,2,3,5 --n-toys 10000 --job-name hps2015_2016_inj_95CL_w164 --partition milano --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_2016_injection_95CL_w164.slurm
bash submit_injection_all.sh
```

Mass-range convention in all production configs:
- 2015: **20–130 MeV**
- 2016: **35–210 MeV**
- Combined scan window: **20–210 MeV** (combined-fit significance populated in overlap region where multiple datasets are active)

### SLURM Batch Processing

Generate a SLURM array job script for parallel processing:

```bash
# Generate SLURM script
hps-gpr slurm-gen --config my_config.yaml --n-jobs 100 --output submit.slurm

# Submit to cluster
sbatch submit.slurm

# After jobs complete, combine results and auto-build publication-style summary suites
hps-gpr slurm-combine --output-dir outputs/my_analysis/
```

`slurm-combine` now writes `summary_combined_<dataset_tag>/` folders (for example,
`summary_combined_2015` or `summary_combined_2015_2016`) containing: The suite is generated
from merged UL-band CSVs with priority `ul_bands_combined_*` → `ul_bands_eps2_*` → `ul_bands_*`:
- expected/observed UL bands for signal yield and $\epsilon^2$
- observed-only UL curves (signal yield and $\epsilon^2$)
- UL-tail p-value summaries (`p_strong`, `p_weak`, `p_two`)
- analytic local/global $p_0$ and local/global $Z$ (with Sidak LEE correction and $N_{\mathrm{eff}}$ derived from mass-resolution spacing via the configured blind-window width, typically 1.64 or 1.96)
- p-value component overlays with local/global 1$\sigma$, 2$\sigma$, and (when visible) 3$\sigma$ references.


Example: generate job files for 10k-toy limit-band production on S3DF (`milano`),
including explicit account charging:

```bash
# 2015-only limit bands (111 mass points => 111 array jobs)
hps-gpr slurm-gen \
  --config config_2015_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_bands_10k \
  --partition milano \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_bands_10k.slurm

# 2016 10% limit bands (176 mass points => 176 array jobs)
hps-gpr slurm-gen \
  --config config_2016_10pct_10k.yaml \
  --n-jobs 176 \
  --job-name hps2016_10pct_bands_10k \
  --partition milano \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2016_10pct_bands_10k.slurm

# 2015+2016 combined scan window (20–210 MeV => 191 array jobs)
hps-gpr slurm-gen \
  --config config_2015_2016_combined_10k.yaml \
  --n-jobs 191 \
  --job-name hps2015_2016_combined_bands_10k \
  --partition milano \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_combined_bands_10k.slurm
```


Submit all three scripts (**run directly; do not wrap with `sbatch submit_all.sh`**):

```bash
./submit_all.sh \
  submit_2015_bands_10k.slurm \
  submit_2016_10pct_bands_10k.slurm \
  submit_2015_2016_combined_bands_10k.slurm
```


If your site requires submission-time account/QOS flags, pass them through:

```bash
./submit_all.sh --account hps:hps-prod --qos normal \
  submit_2015_bands_10k.slurm \
  submit_2016_10pct_bands_10k.slurm \
  submit_2015_2016_combined_bands_10k.slurm
```

If you see `sbatch: command not found`, you are not on a SLURM submit node. Verify with:

```bash
command -v sbatch
```

Then SSH to your SLURM login host (example):

```bash
ssh <your_user>@s3dflogin.slac.stanford.edu
```

## Output Files

The scan produces the following outputs:

```
outputs/
├── validation_report.json          # Dataset validation results
├── results_single.csv              # Per-dataset scan results (+ GP diagnostics columns)
├── results_combined.csv            # Combined-fit scan results
├── combined.csv                    # Backward-compatible alias of results_combined.csv
├── summary_plots/
│   ├── scan_summary_single.csv     # Copy of per-dataset scan table for plotting workflows
│   ├── A_up_<dataset>.png          # 95% CL amplitude UL vs mass
│   ├── eps2_ul_<dataset>.png       # 95% CL epsilon^2 UL vs mass
│   ├── A_hat_<dataset>.png         # Extracted signal yield (with ±1σ band)
│   ├── p0_<dataset>.png            # Local/global p0 summaries
│   ├── Z_local_global_<dataset>.png# Local/global significance summaries
│   ├── eps2_ul_overlay.png         # Overlay of all datasets + combined eps2 UL
│   └── gp_hyperparameters/
│       ├── gp_ls_ratio_<dataset>.png # Length-scale ratios (l/sigma_x)
│       ├── gp_ls_abs_<dataset>.png   # Absolute GP length scales (l_hi, l_lo, l_opt)
│       ├── gp_const_<dataset>.png    # ConstantKernel amplitude vs mass
│       └── gp_lml_<dataset>.png      # GP log marginal likelihood vs mass
├── summary_combined_<dataset_tag>/ # Created by `hps-gpr slurm-combine`
│   ├── ul_bands_signal_yield_obsexp.png
│   ├── ul_bands_eps2_obsexp.png
│   ├── ul_observed_only_signal_yield.png
│   ├── ul_observed_only_eps2.png
│   ├── ul_pvalues.png
│   ├── ul_pvalues_components_local_global_refs.png
│   ├── p0_analytic_local_global.png
│   └── Z_local_global.png
└── mXXXMeV/                        # Optional per-mass folders (if save_per_mass_folders=true)
    ├── <dataset>/
    │   ├── fit_full.png            # Full-range fit diagnostic
    │   ├── blind_fit.png           # Blind-window fit diagnostic
    │   ├── s_over_b_ul.png         # Signal/background ratio for UL
    │   ├── numbers.json            # Per-mass numerical summary
    │   └── error.txt               # Present only when that mass-point fit fails
    └── combined/
        ├── combined_summary.png    # Combined-fit text summary plot
        └── numbers.json            # Combined per-mass numerical summary
```


## Package Structure

| Module | Description |
|--------|-------------|
| `config.py` | Configuration dataclass and YAML loading |
| `dataset.py` | Dataset configuration and polynomial utilities |
| `validation.py` | ROOT file and histogram validation |
| `gpr.py` | GPR preprocessing and fitting functions |
| `template.py` | Signal template and CLs calculations |
| `statistics.py` | p-value calculations and signal extraction |
| `io.py` | Histogram loading and background estimation |
| `conversion.py` | A ↔ ε² unit conversions |
| `plotting.py` | All visualization functions |
| `evaluation.py` | Single and combined dataset evaluation |
| `scan.py` | Main scan driver |
| `bands.py` | Expected upper limit bands |
| `injection.py` | Signal injection studies |
| `slurm.py` | SLURM job utilities |
| `cli.py` | Command-line interface |

## Python API

The package can also be used programmatically:

```python
from hps_gpr import (
    Config, load_config,
    make_datasets, validate_datasets,
    run_scan, evaluate_single_dataset,
    expected_ul_bands_for_dataset,
)

# Load configuration
config = load_config("my_config.yaml")

# Create datasets
datasets = make_datasets(config)

# Validate
validate_datasets(datasets, config)

# Run scan
df_single, df_combined = run_scan(datasets, config)

# Or evaluate a single mass point
from hps_gpr import evaluate_single_dataset
result, prediction = evaluate_single_dataset(
    datasets["2015"],
    mass=0.05,  # GeV
    config=config
)
print(f"A_up = {result.A_up}, eps2_up = {result.eps2_up}")
```

## Physics Background

The analysis searches for dark photon (A') signals in the e+e- invariant mass spectrum. Key physics quantities:

- **A**: Signal amplitude (number of signal events)
- **ε²**: Dark photon coupling squared
- **σ(m)**: Mass resolution as a function of mass
- **f_rad(m)**: Radiative fraction

The conversion between A and ε² uses:

```
ε² = (2 α_em A) / (3 π m f_rad ρ)
```

where ρ is the integral density (counts per GeV) in the signal region.

## References

- HPS Collaboration dark photon search publications
- Gaussian Process Regression: Rasmussen & Williams, "Gaussian Processes for Machine Learning"
- CLs method: Read, A.L., "Presentation of search results: the CL_s technique"


For combined runs, `slurm-combine` also writes per-dataset publication overlays inside the summary suite, e.g. `2015_UL_sig_yield_bands.png`, `2015_UL_eps2_yield_bands.png`, `2016_UL_sig_yield_bands.png`, `2016_UL_eps2_yield_bands.png`, plus dataset-specific `*_p0_local_global.png` and `*_Z_local_global.png`.
