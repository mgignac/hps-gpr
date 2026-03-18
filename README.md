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

### Local Analysis Note Build

The analysis note in `hps_gpr_analysis_note/` can be built locally with `tectonic`:

```bash
cd hps_gpr_analysis_note
tectonic main.tex
```

That writes `hps_gpr_analysis_note/main.pdf`. The same folder is also upload-ready for Overleaf.
If `tectonic` is not available on your machine, the lightweight macOS install is:

```bash
brew install tectonic
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

For ROOT file paths/histogram names used in the legacy validation notebook workflow,
use [`v15_8_HPS_simultaneous_GP_notebook_quality.ipynb`](v15_8_HPS_simultaneous_GP_notebook_quality.ipynb)
as the reference map (especially for local path conventions and histogram-key checks).

Current 2021 production defaults in this repo use:
- ROOT file: `/sdf/home/e/epeets/run/2021_bump/preselection_invM_psumlt2p8_hists.root`
- histogram: `preselection/h_invM_8000`
- scan window: `30-250 MeV`
- resolution-scaled GP upper length-scale cap: `9 sigma_m`

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
- **2021 1%-only**
- **2015+2016 combined**
- **2015+2016 10%+2021 1% combined**

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
hps-gpr slurm-gen-inject --config study_configs/config_2015_2016_combined_blind1p64_95CL_10k_injection.yaml --datasets 2015,2016,combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths s1,s2,s3,s5 --n-toys 10000 --no-write-toy-csv --job-name hps2015_2016_inj_95CL_w164 --partition roma --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_2016_injection_95CL_w164.slurm
# note: submission script auto-skips out-of-range masses (2015: 20-130 MeV, 2016: 34-210 MeV)
bash submit_injection_all.sh

# After jobs finish, merge flat CSV outputs and build publication-style summaries
hps-gpr inject-plot --input-dir outputs/study_2015_2016_combined_w1p64_95CL/injection_flat --output-dir outputs/study_2015_2016_combined_w1p64_95CL/injection_summary
```

New 2021 1% and three-way combined copy/paste examples:

```bash
# 2021 1% injection/extraction study
hps-gpr inject \
  --config study_configs/config_2021_1pct_blind1p64_95CL_10k_injection.yaml \
  --dataset 2021 \
  --masses 0.040,0.060,0.080,0.100,0.120,0.160 \
  --strengths 1,2,3,5 \
  --n-toys 10000

# Combined 2015 + 2016 10% + 2021 1% injection/extraction study
# The YAML is configured for intersection-only masses where all three datasets contribute.
hps-gpr inject \
  --config study_configs/config_2015_2016_10pct_2021_1pct_blind1p64_95CL_10k_injection.yaml \
  --dataset combined \
  --masses 0.040,0.060,0.080,0.100,0.120 \
  --strengths 1,2,3,5 \
  --n-toys 10000

# Batch production for the three-way study: generate per-dataset and combined outputs together
hps-gpr slurm-gen-inject \
  --config study_configs/config_2015_2016_10pct_2021_1pct_blind1p64_95CL_10k_injection.yaml \
  --datasets 2015,2016,2021,combined \
  --masses 0.040,0.060,0.080,0.100,0.120 \
  --strengths s1,s2,s3,s5 \
  --n-toys 10000 \
  --no-write-toy-csv \
  --job-name hps2015_2016_2021_1pct_inj_95CL_w164 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_2021_1pct_injection_95CL_w164.slurm

# After the batch jobs finish, merge and plot the three-way injection study
hps-gpr inject-plot \
  --input-dir outputs/study_2015_2016_10pct_2021_1pct_w1p64_95CL/injection_flat \
  --output-dir outputs/study_2015_2016_10pct_2021_1pct_w1p64_95CL/injection_summary
```

Notes:
- In `inj_strength_mode: sigmaA`, `--strengths` accepts both `1,2,3,5` and `s1,s2,s3,s5`.
- SLURM injection jobs now run only the explicitly requested strength per job point (no implicit rerun of all configured strengths).
- `inject` now defaults to streaming toy aggregation (`inj_stream_aggregate: true`) with per-point batching (`inj_aggregate_every`, default 100) and toy workers (`inj_n_workers`, default 5). Use `--legacy-toys` to force the old full toy-table path.
- Use `--no-write-toy-csv` (or `inj_write_toy_csv: false`) for large productions when toy-level tables are not needed; this avoids multi-million-row `inj_extract_toys_*` files in `injection_flat`.
- `inject-plot` supports summary-only inputs (from `inj_extract_summary_*.csv`) and still builds linearity/bias/pull-width/coverage/heatmap/pull-vs-mass summaries plus Z-calibration residual panels (using `Zhat_mean`/`Zhat_q16`/`Zhat_q84` when available); pull histograms still require toy CSVs.


### GP-mean/global-fit pseudoexperiment mode (v15_8-style full procedural toys)

The injection framework already supports pseudoexperiments in two modes:
- `inj_refit_gp_on_toy: false` (default fast mode): conditional GP toys in the blind window.
- `inj_refit_gp_on_toy: true` (full procedural mode): build full-range pseudo-data from the GP global-fit mean, inject signal, and refit GP on sidebands toy-by-toy before extraction.

For reviewer-facing extraction displays there is a second, separate switch:
- `extraction_display_refit_gp_on_toy: false`: make the closure-style no-refit display. This keeps the sideband-trained GP fixed and is the right choice when you want the pseudoexperiment display to match the observed-data validation display as closely as possible.
- `extraction_display_refit_gp_on_toy: true`: make the full-refit absorption diagnostic. This regenerates the toy across the full mass range and retrains the GP on the toy sidebands before extraction.

These settings answer different questions:
- `inj_refit_gp_on_toy` controls how large toy ensembles are generated for summary/closure studies.
- `extraction_display_refit_gp_on_toy` controls the one-pseudoexperiment reviewer plots used in the note.

The second mode corresponds to the requested "GP mean/global fit" pseudoexperiment workflow and is the closest match to the v15_8 notebook methodology.

Ready-made configs are included:
- `study_configs/config_2015_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml`
- `study_configs/config_2016_10pct_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml`
- `study_configs/config_2015_2016_combined_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml`

Example runs:

```bash
# 2015 full procedural pseudoexperiments from GP mean/global fit
hps-gpr inject --config study_configs/config_2015_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml --dataset 2015 --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135 --strengths 1,2,3,5 --n-toys 10000

# 2016 10% full procedural pseudoexperiments
hps-gpr inject --config study_configs/config_2016_10pct_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml --dataset 2016 --masses 0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths 1,2,3,5 --n-toys 10000

# Combined (plus per-dataset in batch) full procedural pseudoexperiments
hps-gpr slurm-gen-inject --config study_configs/config_2015_2016_combined_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml --datasets 2015,2016,combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths s1,s2,s3,s5 --n-toys 10000 --no-write-toy-csv --job-name hps2015_2016_inj_gpmean --partition roma --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_2016_injection_gpmean.slurm
bash submit_injection_all.sh
hps-gpr inject-plot --input-dir outputs/study_2015_2016_combined_w1p64_95CL_gpmean_pseudoexp/injection_flat --output-dir outputs/study_2015_2016_combined_w1p64_95CL_gpmean_pseudoexp/injection_summary
```

Mass-range convention in all production configs:
- 2015: **20–130 MeV**
- 2016: **35–210 MeV**
- Combined scan window: **20–210 MeV** (combined-fit significance populated in overlap region where multiple datasets are active)

#### Generating example extraction displays from pseudoexperiments

For reviewer-facing figures it is often more useful to show one carefully constructed pseudoexperiment than a large toy ensemble summary. The `extract-display` workflow generates those representative plots directly from YAML config, using the same v15_8-style GP-mean/global-fit pseudoexperiment logic:
- one pseudoexperiment per requested `(mass, injected significance)` point
- a full-range context panel
- a blind-window fit panel zoomed to `blind window ± 0.5 sigma`
- an extracted-signal panel with residual uncertainty bands and the injected/extracted Gaussian extending outside the blind window
- a right-side boxed summary with the injected level, realized event count, extracted yield, per-dataset extracted significance, and epsilon-squared numbers

Five ready-made configs are included:
- `study_configs/config_2015_extraction_display_v15p8.yaml`
- `study_configs/config_2016_extraction_display_v15p8.yaml`
- `study_configs/config_2021_1pct_extraction_display_v15p8.yaml`
- `study_configs/config_2015_2016_combined_extraction_display_v15p8.yaml`
- `study_configs/config_2015_2016_2021_1pct_combined_extraction_display_v15p8.yaml`

Copy/paste:

```bash
# 2015 representative extraction displays
hps-gpr extract-display --config study_configs/config_2015_extraction_display_v15p8.yaml

# 2016 representative extraction displays
hps-gpr extract-display --config study_configs/config_2016_extraction_display_v15p8.yaml

# Combined 2015+2016 representative extraction displays
hps-gpr extract-display --config study_configs/config_2015_2016_combined_extraction_display_v15p8.yaml

# 2021 1% representative extraction displays
hps-gpr extract-display --config study_configs/config_2021_1pct_extraction_display_v15p8.yaml

# Combined 2015+2016 10%+2021 1% representative extraction displays
hps-gpr extract-display --config study_configs/config_2015_2016_2021_1pct_combined_extraction_display_v15p8.yaml

# Optional: override the sigma-level list for this run only
hps-gpr extract-display --config study_configs/config_2015_extraction_display_v15p8.yaml --strengths 5

# Optional: override both the representative masses and the combined dataset list
hps-gpr extract-display \
  --config study_configs/config_2015_2016_2021_1pct_combined_extraction_display_v15p8.yaml \
  --dataset combined \
  --datasets 2015,2016,2021 \
  --masses 0.040,0.080,0.120 \
  --strengths 5
```

What these plots mean:
- They are not coverage plots or ensemble summaries; they are single representative pseudoexperiments meant to expose the mechanics of the search and extraction.
- For single-dataset displays, the injected strength is specified in units of the local reference `sigma_A` and then converted into a full-range injected signal whose expected yield inside the blind window matches the fitted `A` convention.
- For combined displays, the injected signal is built from a common `epsilon^2` model, not by forcing the same per-dataset event yield in each year. The target combined significance is translated into one shared `epsilon^2`, then mapped to dataset-specific injected amplitudes using each dataset's `A(epsilon^2)` response and extraction resolution.
- The no-refit display is the clean closure test: it asks whether the extraction fit can recover the injected signal when the sideband-conditioned background model is treated as fixed.
- The refit display is the absorption diagnostic: it asks how much signal can be absorbed when the GP is retrained on the toy sidebands.
- The lower panel is therefore directly interpretable as "what signal was truly injected" versus "what the blind-window fit extracted" for that pseudoexperiment, while the side box shows the observed UL scale for context.

To customize later:
- edit `extraction_display_masses_gev` in the YAML to add or remove mass points
- edit `extraction_display_sigma_multipliers` to change the injected `n sigma` values
- for the combined config, edit `extraction_display_dataset_keys` if you want to extend the common-signal display to a different enabled dataset set
- use `--strengths` and `--masses` on the CLI when you want a one-off subset without editing the YAML
- use `--datasets` with `--dataset combined` when you want to render a different enabled dataset combination than the YAML default
- outputs are written under `output_dir/extraction_display/<dataset-or-combined>/` as both PNG and PDF, with a JSON sidecar for the numerical values shown in the figure

#### Observed-data mass-validation displays

The `observed-display` workflow is the data-facing companion to `extract-display`. It is meant for reviewer-grade bump validation at one specified mass hypothesis:
- combined mode produces one hero figure with one fit panel and one residual/extracted-signal panel per active dataset, plus a shared combined bump panel
- per-dataset context and zoom views are written alongside the hero figure
- by default, outputs are written under `config.output_dir/observed_display/mXXXMeV/`; if `--output-dir` is supplied, the mass folders are written directly under that root
- metadata sidecars are dataset-specific (`metadata_combined.json`, `metadata_2015.json`, etc.), so combined and per-dataset commands can safely write into the same mass folder

Copy/paste example series for a `123 MeV` hypothesis:

```bash
# Choose the production config:
# current post-leakage baseline:
CFG=config_2015_2016_10pct_2021_1pct_10k.yaml
# for the radiative-penalty rerun instead, use:
# CFG=config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml

# Combined observed-data validation display at 123 MeV
hps-gpr observed-display \
  --config "${CFG}" \
  --mass 0.123 \
  --dataset combined \
  --datasets 2015,2016,2021 \
  --output-dir outputs/observed_validation_123

# Individual observed-data cross-checks at the same mass
hps-gpr observed-display \
  --config "${CFG}" \
  --mass 0.123 \
  --dataset 2015 \
  --output-dir outputs/observed_validation_123

hps-gpr observed-display \
  --config "${CFG}" \
  --mass 0.123 \
  --dataset 2016 \
  --output-dir outputs/observed_validation_123

hps-gpr observed-display \
  --config "${CFG}" \
  --mass 0.123 \
  --dataset 2021 \
  --output-dir outputs/observed_validation_123
```

Typical products in `outputs/observed_validation_123/m123MeV/`:
- `observed_display_combined.png/.pdf/.json`
- `observed_context_2015.png/.pdf`
- `observed_zoom_2015.png/.pdf`
- `observed_context_2016.png/.pdf`
- `observed_zoom_2016.png/.pdf`
- `observed_context_2021.png/.pdf`
- `observed_zoom_2021.png/.pdf`
- `metadata_combined.json`

Batch workflow tip:
- extraction displays are independent across `(dataset set, mass, injected sigma)` points, so the recommended cluster workflow is now one batch job per point
- this avoids running `40 MeV`, `80 MeV`, and `120 MeV` serially inside one long walltime slot and makes the reviewer figures easier to rerun selectively

Recommended SLURM launcher:

```bash
hps-gpr slurm-gen-extract-display \
  --config study_configs/config_2015_2016_2021_1pct_combined_extraction_display_v15p8.yaml \
  --dataset combined \
  --datasets 2015,2016,2021 \
  --masses 0.040,0.080,0.120 \
  --strengths 3,5 \
  --job-name hps151621_exdisp \
  --partition roma \
  --account hps:hps-prod \
  --time 06:00:00 \
  --memory 8G \
  --output submit_extract_display_151621.slurm

bash submit_extract_display_all.sh
```

For local one-off reruns, the CLI overrides are usually enough:

```bash
hps-gpr extract-display \
  --config study_configs/config_2015_2016_2021_1pct_combined_extraction_display_v15p8.yaml \
  --dataset combined \
  --datasets 2015,2016,2021 \
  --masses 0.040 \
  --strengths 5 \
  --output-dir outputs/extraction_display_m040_z5
```

That layout keeps the outputs separated by mass/significance and gives a straightforward reviewer-production path: one reviewer figure per batch job, with selective reruns when a single display changes.

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


Example: generate job files for 10k-toy limit-band production on S3DF (`roma`),
including explicit account charging:

```bash
# 2015-only limit bands (111 mass points => 111 array jobs)
hps-gpr slurm-gen \
  --config config_2015_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_bands_10k.slurm

# 2016 10% limit bands (176 mass points => 176 array jobs)
hps-gpr slurm-gen \
  --config config_2016_10pct_10k.yaml \
  --n-jobs 176 \
  --job-name hps2016_10pct_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2016_10pct_bands_10k.slurm

# 2015+2016 combined scan window (20–210 MeV => 191 array jobs)
hps-gpr slurm-gen \
  --config config_2015_2016_combined_10k.yaml \
  --n-jobs 191 \
  --job-name hps2015_2016_combined_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_combined_bands_10k.slurm
```

For the staged `1%` pass, the recommended order is to finish the three individual
summary suites first, then launch the three-way combination:

```bash
# 1. 2015-only limit bands (111 mass points => 111 jobs)
hps-gpr slurm-gen \
  --config config_2015_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_bands_10k.slurm

./submit_all.sh submit_2015_bands_10k.slurm
hps-gpr slurm-combine --output-dir outputs/prod_2015_10k_3

# 2. 2016 10% limit bands (176 mass points => 176 jobs)
hps-gpr slurm-gen \
  --config config_2016_10pct_10k.yaml \
  --n-jobs 176 \
  --job-name hps2016_10pct_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2016_10pct_bands_10k.slurm

./submit_all.sh submit_2016_10pct_bands_10k.slurm
hps-gpr slurm-combine --output-dir outputs/prod_2016_10pct_10k

# 3. 2021 1% limit bands (30-250 MeV => 221 mass points => 221 jobs)
hps-gpr slurm-gen \
  --config config_2021_1pct_10k.yaml \
  --n-jobs 221 \
  --job-name hps2021_1pct_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2021_1pct_bands_10k.slurm

./submit_all.sh submit_2021_1pct_bands_10k.slurm
hps-gpr slurm-combine --output-dir outputs/prod_2021_1pct_10k

# 4. Combined 2015 + 2016 10% + 2021 1% limit bands
# Union scan window: 20-250 MeV => 231 mass points => 231 jobs
hps-gpr slurm-gen \
  --config config_2015_2016_10pct_2021_1pct_10k.yaml \
  --n-jobs 231 \
  --job-name hps2015_2016_10pct_2021_1pct_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_10pct_2021_1pct_bands_10k.slurm

./submit_all.sh submit_2015_2016_10pct_2021_1pct_bands_10k.slurm
hps-gpr slurm-combine --output-dir outputs/prod_2015_2016_10pct_2021_1pct_10k_bands

# 5. Same three-way production with a 7% radiative-fraction penalty on all datasets
hps-gpr slurm-gen \
  --config config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml \
  --n-jobs 231 \
  --job-name hps2015_2016_10pct_2021_1pct_bands_10k_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_10pct_2021_1pct_bands_10k_rpen7.slurm

./submit_all.sh submit_2015_2016_10pct_2021_1pct_bands_10k_rpen7.slurm
hps-gpr slurm-combine --output-dir outputs/prod_2015_2016_10pct_2021_1pct_10k_bands_rpen7
```

If you want quick local non-SLURM smoke passes before the full productions:

```bash
hps-gpr scan --config config_2021_1pct_10k.yaml
hps-gpr scan --config config_2015_2016_10pct_2021_1pct_10k.yaml
hps-gpr scan --config config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml
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
├── injection_flat/                 # Created by `submit_injection_all.sh` jobs (single folder, uniquely named CSVs)
│   ├── inj_extract_toys_<dataset>__jobds_<jobds>__m_<mass>__s_<strength>.csv   # optional (`--write-toy-csv`)
│   └── inj_extract_summary_<dataset>__jobds_<jobds>__m_<mass>__s_<strength>.csv
├── injection_summary/              # Created by `hps-gpr inject-plot`
│   ├── inj_extract_toys_<dataset>.csv
│   ├── inj_extract_summary_<dataset>.csv
│   ├── linearity_all.png
│   ├── bias_all.png
│   ├── pull_width_all.png
│   ├── coverage_all.png
│   ├── linearity_<dataset>.png
│   ├── bias_<dataset>.png
│   ├── pull_width_<dataset>.png
│   ├── coverage_<dataset>.png
│   ├── heatmap_pull_mean_<dataset>.png
│   ├── heatmap_pull_width_<dataset>.png
│   ├── z_calibration_residual_<dataset>.png
│   ├── z_calibration_residual_comparison.png
│   ├── combined_search_power_scenarios.png
│   ├── combined_search_power_constituent_pvalues_5sigma.png
│   ├── combined_constituent_pvalues_target5sigma.csv
│   └── combined_signal_allocation_mXXXMeV.png/.csv
├── projections/                    # Suggested home for `hps-gpr project-eps2-reach`
│   ├── projected_unblinded_reach_eps2.png
│   ├── projected_unblinded_reach_eps2.pdf
│   └── projected_unblinded_reach_eps2.csv
├── extraction_display/             # Created by `hps-gpr extract-display`
│   ├── 2015/ or 2016/ or combined/
│   │   ├── extract_display_<tag>.png
│   │   ├── extract_display_<tag>.pdf
│   │   └── extract_display_<tag>.json
│   └── ...                         # tag includes mass and injected sigma level
├── observed_display/               # Created by `hps-gpr observed-display`
│   └── mXXXMeV/
│       ├── observed_display_combined.png/.pdf/.json
│       ├── observed_display_<dataset>.png/.pdf/.json
│       ├── observed_context_<dataset>.png/.pdf
│       ├── observed_zoom_<dataset>.png/.pdf
│       ├── metadata_combined.json
│       └── metadata_<dataset>.json
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




Injection/extraction defaults now use `inj_mode: poisson` (Poissonian signal-count fluctuations per template bin), consistent with counting-experiment pseudo-data generation used in modern HEP profile-likelihood workflows.

Injection/extraction plotting suite (`inject-plot`) covers the v15_8 closure checks used for robustness studies:
- linearity: `⟨Â⟩` vs injected strength (or injected `n_σ`), with ideal reference line
- bias: `⟨Â⟩ − A_inj` vs injected strength
- pull-width stability: `std((Â−A_inj)/σ_A)` with unit-width reference
- coverage: fractions within `|pull|<1` and `|pull|<2` with Gaussian expectations (68.3%, 95.4%); dataset panels now split by injection level when mass overlays would otherwise overdraw
- mass/strength heatmaps for pull mean and pull width, per dataset and for combined extraction
- pull-vs-mass panels with connected lines and sigma-level legend labels (`1σ`, `2σ`, `3σ`, `5σ`) when `inj_nsigma` is available



### Combined-search power study outputs

`hps-gpr inject-plot` now also produces a focused combined-sensitivity study for whatever enabled dataset set is present in the injection summary, including the full `2015+2016+2021` shared search:

- **Scenario plot**: `combined_search_power_scenarios.png`
  - for the three-dataset workflow this includes `1σ(2015)+1σ(2016)+1σ(2021)`
  - and `1σ(2015)+2σ(2016)+3σ(2021)`
  - uses inverse-variance weighting based on `sigmaA_ref(m)` from toys.
- **Constituent p-value requirement plot**: `combined_search_power_constituent_pvalues_5sigma.png`
  - shows per-dataset local `Z` and one-sided `p0` required to realize a combined `5σ` excess.
  - writes a reproducible table: `combined_constituent_pvalues_target5sigma.csv`.
- **Allocation plots (publication-ready)** for representative masses (default 40, 80, 120 MeV):
  - `combined_signal_allocation_m040MeV.png`
  - `combined_signal_allocation_m080MeV.png`
  - `combined_signal_allocation_m120MeV.png`
  - plus matching `.csv` tables with per-dataset injected amplitudes for target combined `Z=1,3,5`.
- **Projection plot**: `projected_unblinded_reach_eps2.png`
  - combines dataset-level `eps2` curves with `sqrt(L)` scaling assumptions
  - defaults to `2015 x1`, `2016 x10`, `2021 x100`
  - writes a sidecar table: `projected_unblinded_reach_eps2.csv`

These plots are designed to show *why* combined searches improve sensitivity:
- statistically optimal weighting emphasizes the dataset with smaller `sigmaA_ref` (higher information content);
- combined significance scales with the quadrature of independent information channels, making modest per-dataset excesses more impactful when combined;
- explicit allocation tables improve reproducibility for internal notes/publication follow-up and allow direct cross-checks against UL-derived sensitivity expectations.

Methodology follows standard profile-likelihood asymptotics and inverse-variance combination conventions used broadly in HEP analyses (e.g. Cowan et al., EPJC 71 (2011) 1554).

Copy/paste: generate combined-search scenario plots (and constituent 5σ table) from an existing injection summary folder:

```bash
python - <<'PY'
import glob
import os
import pandas as pd
from hps_gpr.plotting import plot_combined_search_power

in_dir = "outputs/study_2015_2016_combined_w1p64_95CL/injection_summary"
out_dir = os.path.join(in_dir, "combined_search_power_scenarios_only")
paths = sorted(glob.glob(os.path.join(in_dir, "inj_extract_toys_*.csv")))
if paths:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
else:
    paths = sorted(glob.glob(os.path.join(in_dir, "inj_extract_summary_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
written = plot_combined_search_power(df, outdir=out_dir)
print("wrote", len(written), "plots to", out_dir)
PY
```

Expected scenario products in `combined_search_power_scenarios_only/`:
- `combined_search_power_scenarios.png`
- `combined_search_power_constituent_pvalues_5sigma.png`
- `combined_constituent_pvalues_target5sigma.csv`

Copy/paste: generate combined signal-allocation products for custom masses and targets:

```bash
python - <<'PY'
import glob
import os
import pandas as pd
from hps_gpr.plotting import plot_combined_search_power

in_dir = "outputs/study_2015_2016_combined_w1p64_95CL/injection_summary"
out_dir = os.path.join(in_dir, "combined_signal_allocation_only")
paths = sorted(glob.glob(os.path.join(in_dir, "inj_extract_toys_*.csv")))
if paths:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
else:
    paths = sorted(glob.glob(os.path.join(in_dir, "inj_extract_summary_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
written = plot_combined_search_power(
    df,
    outdir=out_dir,
    masses_focus=[0.040, 0.080, 0.120],
    z_targets=[1.0, 3.0, 5.0],
)
print("wrote", len(written), "plots to", out_dir)
PY
```

Expected allocation products include:
- `combined_signal_allocation_m040MeV.png` + `.csv`
- `combined_signal_allocation_m080MeV.png` + `.csv`
- `combined_signal_allocation_m120MeV.png` + `.csv`

Copy/paste: generate the projected unblinded `eps^2` reach from a dataset-level CSV:

```bash
hps-gpr project-eps2-reach \
  --input-csv outputs/prod_2015_2016_10pct_2021_1pct_10k_bands/summary_plots/scan_summary_single.csv \
  --output outputs/prod_2015_2016_10pct_2021_1pct_10k_bands/projections/projected_unblinded_reach_eps2.png \
  --scale-2015 1 \
  --scale-2016 10 \
  --scale-2021 100
```

The input CSV just needs dataset-level rows with `dataset`, `mass_GeV`, and one supported `eps2` column such as `eps2_up`, `eps2_obs`, or the expected-band columns (`eps2_med`, `eps2_lo1`, `eps2_hi1`, ...).

### Statistical validation checklist (publication gate)

Use this checklist before freezing plots for notes/papers:
- Pull calibration: `pull_mean ~ 0` and `pull_width ~ 1` across masses and injection levels.
- Coverage calibration: `cov_1sigma ~ 0.683` and `cov_2sigma ~ 0.954` without systematic mass trends.
- Z-calibration residual: `Delta Z = Zhat - Zinj` centered near zero with stable q16/q84 bands.
- Strength-mode consistency: in `sigmaA` mode, each SLURM `(dataset, mass, strength)` job contributes only its requested strength point.
- Combined-power consistency: constituent `Z_i`/`p0_i` values for a fixed combined target (e.g. 5sigma) should follow inverse-variance information fractions (`1/sigmaA_ref^2`).
- Global-significance reporting: local/global `p0` and `Z` must include the configured LEE treatment (`N_eff`, Sidak correction) in summary suites.

### Injection plotting style profile

`hps_gpr.plotting.set_injection_plot_style(mode="paper")` configures a publication-ready injection style with:
- consistent `rcParams` (font scale, linewidths, marker sizes, legend framing/columns, grid alpha)
- constrained-layout figures and legends placed outside axes for crowded overlays
- axis labels standardized with HEP notation (`A_{inj}`, `\hat{A}`, `\sigma_A`) and explicit units where relevant
- colorblind-safe Okabe–Ito-like palette and deterministic mass-to-color assignment
- automatic dual-output save policy for injection summary plots: high-DPI PNG + vector PDF
- faceted all-dataset summaries (via `subplot_mosaic`) when overlays become crowded
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
- G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, EPJC 71 (2011) 1554.
- Gaussian Process Regression: Rasmussen & Williams, "Gaussian Processes for Machine Learning"
- CLs method: Read, A.L., "Presentation of search results: the CL_s technique"
- Asymptotic profile-likelihood tests: Cowan, Cranmer, Gross, Vitells, EPJC 71 (2011) 1554
- Look-Elsewhere correction / trial factors: Gross & Vitells, EPJC 70 (2010) 525
- ATLAS/CMS Higgs combinations for modern combined profile-likelihood methodology


For combined runs, `slurm-combine` also writes per-dataset publication overlays inside the summary suite, e.g. `2015_UL_sig_yield_bands.png`, `2015_UL_eps2_yield_bands.png`, `2016_UL_sig_yield_bands.png`, `2016_UL_eps2_yield_bands.png`, plus dataset-specific `*_p0_local_global.png` and `*_Z_local_global.png`.
