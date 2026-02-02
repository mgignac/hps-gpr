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
range_2015: [0.015, 0.140]

# CLs settings
cls_alpha: 0.05
cls_mode: "toys"
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

### Injection/Extraction Study

Run signal injection and extraction closure tests:

```bash
hps-gpr inject --config my_config.yaml --dataset 2015 --masses 0.03,0.06,0.09
```

### SLURM Batch Processing

Generate a SLURM array job script for parallel processing:

```bash
# Generate SLURM script
hps-gpr slurm-gen --config my_config.yaml --n-jobs 100 --output submit.slurm

# Submit to cluster
sbatch submit.slurm

# After jobs complete, combine results
hps-gpr slurm-combine --output-dir outputs/my_analysis/
```

## Output Files

The scan produces the following outputs:

```
outputs/
├── validation_report.json    # Dataset validation results
├── results_single.csv        # Per-dataset results
├── results_combined.csv      # Combined fit results (overlap regions)
├── summary_plots/
│   ├── eps2_ul_2015.png     # Per-dataset epsilon^2 limits
│   ├── eps2_ul_combined.png # Combined limits
│   └── eps2_ul_overlay.png  # All limits overlaid
└── m030MeV/                  # Per-mass-point folders
    ├── 2015/
    │   ├── fit_full.png     # Full range fit
    │   ├── blind_ul.png     # Blind window with UL
    │   ├── s_over_b_ul.png  # Signal/background ratio
    │   └── numbers.json     # Numerical results
    └── combined/
        └── numbers.json
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
