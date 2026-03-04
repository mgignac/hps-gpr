# Global significance and signal-injection production guide

This guide documents reproducible commands for:
- local/global significance with LEE using independent-region width from mass resolution,
- 95% CLs limit bands,
- 10k pseudoexperiment injection/extraction studies,
- and 2015, 2016-10%, and combined submissions.

## Statistical choices

1. **Global significance (LEE)**
   - Compute effective independent trials using
     \(N_\mathrm{eff} \approx \sum_i \Delta m_i / (W\,\sigma_m(m_i))\),
     where \(W\in\{1.64,1.96\}\) and \(\sigma_m\) is the mass resolution.
   - Convert local to global p-values with the **Šidák correction**:
     \(p_\mathrm{global}=1-(1-p_\mathrm{local})^{N_\mathrm{eff}}\).

2. **95% CLs**
   - Use `cls_alpha: 0.05` in 95% CL configs (new) and `cls_alpha: 0.10` in retained 90% CL configs for legacy comparisons.

3. **Injection/extraction pseudoexperiments**
   - Use 10k toys and inject strengths in \(\{0,1,2,3,4,5\}\sigma_A\).
   - Compare pull mean, pull width, and 68/95% coverage targets.

## Literature touchstones

- Gross & Vitells, *Trial factors for the look elsewhere effect in high energy physics*, EPJC 70 (2010), arXiv:1005.1891.
- Cowan et al., *Asymptotic formulae for likelihood-based tests of new physics*, EPJC 71 (2011), arXiv:1007.1727.
- Read, *Presentation of search results: the CLs technique*, J. Phys. G 28 (2002).


## Short theory note (for interpretation and reporting)

- **Asymptotic profile-likelihood calibration (Cowan et al.)**: local test statistics and quoted local significances should be interpreted through asymptotic profile-likelihood results, with toy studies used as closure checks for finite-sample effects and nuisance-treatment fidelity.
- **Look-Elsewhere Effect / global significance (Gross & Vitells)**: when scanning many mass hypotheses, convert local to global significance using an effective-trials treatment (or equivalent upcrossing-based approximation) to avoid overstating discovery significance.
- **Frequentist limits with CLs (Read)**: reported upper limits should be based on the CLs construction to control exclusions in low-sensitivity regimes, with expected bands from background-only pseudoexperiments.

These references define the statistical backbone of this workflow: asymptotic local tests, LEE-corrected global claims, and CLs limits.

## Configs in this folder

- 90% CL (legacy):
  - `config_2015_blind1p64_90CL_10k_injection.yaml`
  - `config_2015_blind1p96_90CL_10k_injection.yaml`
  - `config_2016_10pct_blind1p64_90CL_10k_injection.yaml`
  - `config_2016_10pct_blind1p96_90CL_10k_injection.yaml`
  - `config_2015_2016_combined_blind1p64_90CL_10k_injection.yaml`
  - `config_2015_2016_combined_blind1p96_90CL_10k_injection.yaml`
- 95% CL (new):
  - `config_2015_blind1p64_95CL_10k_injection.yaml`
  - `config_2015_blind1p96_95CL_10k_injection.yaml`
  - `config_2016_10pct_blind1p64_95CL_10k_injection.yaml`
  - `config_2016_10pct_blind1p96_95CL_10k_injection.yaml`
  - `config_2015_2016_combined_blind1p64_95CL_10k_injection.yaml`
  - `config_2015_2016_combined_blind1p96_95CL_10k_injection.yaml`

## Suggested production commands

### 1) Generate SLURM scripts (95% CL, 10k toys)

```bash
hps-gpr slurm-gen --config study_configs/config_2015_blind1p96_95CL_10k_injection.yaml --n-jobs 111 --job-name hps2015_95CL_w196 --partition milano --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_95CL_w196.slurm

hps-gpr slurm-gen --config study_configs/config_2016_10pct_blind1p96_95CL_10k_injection.yaml --n-jobs 176 --job-name hps2016_10pct_95CL_w196 --partition milano --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2016_10pct_95CL_w196.slurm

hps-gpr slurm-gen --config study_configs/config_2015_2016_combined_blind1p96_95CL_10k_injection.yaml --n-jobs 191 --job-name hps2015_2016_comb_95CL_w196 --partition milano --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_2016_combined_95CL_w196.slurm
```

Repeat with `blind1p64` configs for the 1.64-width study.

### 2) Submit

```bash
./submit_all.sh submit_2015_95CL_w196.slurm submit_2016_10pct_95CL_w196.slurm submit_2015_2016_combined_95CL_w196.slurm
```

### 3) Combine and make global-significance suites

```bash
hps-gpr slurm-combine --output-dir outputs/study_2015_w1p96_95CL
hps-gpr slurm-combine --output-dir outputs/study_2016_10pct_w1p96_95CL
hps-gpr slurm-combine --output-dir outputs/study_2015_2016_combined_w1p96_95CL
```

`slurm-combine` will make local/global p0 and Z plots using LEE based on the blind-window width and mass-resolution columns from the merged tables.

## Injection/extraction job matrix (10k toys)

Run toys at selected masses (example list) for each dataset/config:

```bash
hps-gpr inject --config study_configs/config_2015_blind1p96_95CL_10k_injection.yaml --dataset 2015 --masses 0.030,0.040,0.050,0.060,0.070,0.080,0.090
hps-gpr inject --config study_configs/config_2016_10pct_blind1p96_95CL_10k_injection.yaml --dataset 2016 --masses 0.040,0.060,0.080,0.100,0.120,0.140,0.160
```

For large-scale 10k toys per mass/strength point, use Python batch wrappers calling `run_injection_extraction_toys(..., n_toys=10000)` in array jobs and aggregate summary tables.

Example SLURM matrix generation (S3DF/milano) with explicit per-point strengths:

```bash
hps-gpr slurm-gen-inject --config study_configs/config_2015_2016_combined_blind1p96_95CL_10k_injection.yaml --datasets 2015,2016,combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths s1,s2,s3,s5 --n-toys 10000 --no-write-toy-csv --job-name hps2015_2016_inj_95CL --partition milano --account hps:hps-prod --time 24:00:00 --memory 8G --output submit_2015_2016_injection_95CL.slurm
bash submit_injection_all.sh
```

Notes:
- In `inj_strength_mode: sigmaA`, both `--strengths 1,2,3,5` and `--strengths s1,s2,s3,s5` are valid.
- Each generated `(dataset, mass, strength)` job now runs only that requested strength (no implicit rerun of all configured strengths).
- `inject-plot` can run in summary-only mode from `inj_extract_summary_*.csv`; toy-only diagnostics still require `inj_extract_toys_*.csv`.

## Injection output policy and file-size scaling

`hps-gpr inject` supports `--write-toy-csv/--no-write-toy-csv` (default controlled by `inj_write_toy_csv` in config).

- With `--write-toy-csv` (default), per-toy tables are written as `inj_extract_toys_<dataset>.csv` (and `inj_extract_toys_combined.csv` for combined mode).
- With `--no-write-toy-csv`, toy tables are kept in memory for summary statistics and plotting, but toy CSV serialization is skipped.
- Compact summary tables are always written: `inj_extract_summary_<dataset>.csv` plus unified `inj_extract_summary_all.csv`.

Expected scaling:

- Toy tables scale approximately as
  N_rows ~= N_masses * N_strengths * N_toys per dataset,
  so disk size grows linearly with each axis.
- Summary tables scale as
  N_rows ~= N_masses * N_strengths per dataset,
  independent of `n_toys`.


For 10k-toy production scans, disabling toy CSV output is recommended unless toy-level reprocessing is explicitly needed.

## Mass ranges used in all production configs

- 2015: **20–130 MeV** (`range_2015: [0.020, 0.130]`)
- 2016 (10%): **35–210 MeV** (`range_2016: [0.035, 0.210]`)
- Combined 2015+2016 scan window: **20–210 MeV** (combined-fit rows populated where >=2 datasets overlap)

These ranges are now synchronized across all top-level and study configs (both 90% and 95% CL variants).


### Combined injection/extraction matrix (publication workflow)

```bash
hps-gpr inject --config study_configs/config_2015_2016_combined_blind1p96_95CL_10k_injection.yaml --dataset combined --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135,0.150,0.170,0.200 --strengths 1,2,3,5 --n-toys 10000
```

Outputs now include per-dataset and combined pull/coverage/heatmap products (`heatmap_pull_mean_<dataset>.png`, `heatmap_pull_width_<dataset>.png`) suitable for publication appendices and closure summaries.
Combined-power outputs also include constituent p-value requirements for a 5sigma combined target (`combined_search_power_constituent_pvalues_5sigma.png`, `combined_constituent_pvalues_target5sigma.csv`).
