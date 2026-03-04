"""SLURM job generation and result combination utilities."""

import glob
import os
import re
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def generate_slurm_script(
    config_path: str,
    n_jobs: int,
    output_path: str,
    job_name: str = "hps-gpr",
    partition: str = "batch",
    time_limit: str = "4:00:00",
    memory: str = "4G",
    conda_env: Optional[str] = None,
    extra_sbatch: Optional[List[str]] = None,
) -> tuple:
    """Generate a single-job SLURM script and a bash submission loop script.

    Produces two files:
      - output_path (e.g. job.slurm): the single-job SLURM script that reads
        TASK_ID and N_TASKS from environment variables passed at sbatch time.
      - submit_all.sh (alongside output_path): a bash loop that calls sbatch
        once per task, passing TASK_ID and N_TASKS via --export.

    Args:
        config_path: Path to configuration YAML file
        n_jobs: Number of individual jobs to submit
        output_path: Path to write the SLURM job script
        job_name: SLURM job name
        partition: SLURM partition
        time_limit: Time limit per job
        memory: Memory per job
        conda_env: Conda environment to activate (optional)
        extra_sbatch: Additional SBATCH directives

    Returns:
        Tuple of (job_script_path, submit_script_path)
    """
    # --- Job script (no --array; TASK_ID/N_TASKS injected at submission) ---
    job_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ]

    if extra_sbatch:
        for directive in extra_sbatch:
            job_lines.append(f"#SBATCH {directive}")

    job_lines.append("")
    job_lines.append("mkdir -p logs")
    job_lines.append("")

    if conda_env:
        job_lines.append("# Activate conda environment")
        job_lines.append("source $(conda info --base)/etc/profile.d/conda.sh")
        job_lines.append(f"conda activate {conda_env}")
        job_lines.append("")

    job_lines.extend([
        "# TASK_ID and N_TASKS are passed via --export at submission time",
        f"hps-gpr scan \\",
        f"    --config {config_path} \\",
        f"    --array-task ${{TASK_ID}} \\",
        f"    --n-tasks ${{N_TASKS}}",
    ])

    job_content = "\n".join(job_lines) + "\n"

    with open(output_path, "w") as f:
        f.write(job_content)
    os.chmod(output_path, 0o755)
    print(f"Wrote SLURM job script to {output_path}")

    # --- Submit loop script ---
    submit_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "submit_all.sh")
    abs_job = os.path.abspath(output_path)

    submit_lines = [
        "#!/bin/bash",
        f"# Submit {n_jobs} individual SLURM jobs for hps-gpr scan",
        f"N_TASKS={n_jobs}",
        f'JOB_SCRIPT="{abs_job}"',
        "",
        "mkdir -p logs",
        "",
        f"for TASK_ID in $(seq 0 $(( N_TASKS - 1 ))); do",
        f'    sbatch --export=ALL,TASK_ID=${{TASK_ID}},N_TASKS=${{N_TASKS}} "${{JOB_SCRIPT}}"',
        "done",
        "",
        f'echo "Submitted ${{N_TASKS}} jobs."',
    ]

    submit_content = "\n".join(submit_lines) + "\n"

    with open(submit_path, "w") as f:
        f.write(submit_content)
    os.chmod(submit_path, 0o755)
    print(f"Wrote submission loop script to {submit_path}")

    return output_path, submit_path


def generate_injection_slurm_scripts(
    config_path: str,
    output_path: str,
    datasets: List[str],
    masses: List[float],
    strengths: List[float],
    n_toys: int,
    output_root: str,
    job_name: str = "hps-gpr-inj",
    partition: str = "batch",
    time_limit: str = "4:00:00",
    memory: str = "4G",
    conda_env: Optional[str] = None,
    extra_sbatch: Optional[List[str]] = None,
    mass_ranges_by_dataset: Optional[dict] = None,
    write_toy_csv: Optional[bool] = None,
) -> tuple:
    """Generate SLURM scripts for one injection job per (dataset, mass, strength)."""
    job_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ]

    if extra_sbatch:
        for directive in extra_sbatch:
            job_lines.append(f"#SBATCH {directive}")

    job_lines.extend([
        "",
        "mkdir -p logs",
        "",
    ])

    if conda_env:
        job_lines.extend([
            "# Activate conda environment",
            "source $(conda info --base)/etc/profile.d/conda.sh",
            f"conda activate {conda_env}",
            "",
        ])

    csv_flag_line = None
    if write_toy_csv is True:
        csv_flag_line = "    --write-toy-csv \\"
    elif write_toy_csv is False:
        csv_flag_line = "    --no-write-toy-csv \\"

    job_lines.extend([
        "# INJECT_* and BASE_OUTPUT_DIR are passed via --export at submission time",
        'JOB_OUTDIR="${BASE_OUTPUT_DIR}/injection_jobs/${INJECT_DATASET}/m_${INJECT_MASS_TAG}/s_${INJECT_STRENGTH_TAG}"',
        'FLAT_OUTDIR="${BASE_OUTPUT_DIR}/injection_flat"',
        "mkdir -p \"${JOB_OUTDIR}\"",
        "mkdir -p \"${FLAT_OUTDIR}\"",
        "",
        "hps-gpr inject \\",
        f"    --config {config_path} \\",
        "    --dataset ${INJECT_DATASET} \\",
        "    --masses ${INJECT_MASS} \\",
        "    --strengths ${INJECT_STRENGTH} \\",
        f"    --n-toys {int(n_toys)} \\",
    ])
    if csv_flag_line is not None:
        job_lines.append(csv_flag_line)
    job_lines.extend([
        "    --output-dir \"${JOB_OUTDIR}\"",
        "",
        "if [ \"${INJECT_DATASET}\" = \"combined\" ]; then",
        '  FILE_GLOB="${JOB_OUTDIR}/injection_extraction/*_combined.csv"',
        "else",
        '  FILE_GLOB="${JOB_OUTDIR}/injection_extraction/*_${INJECT_DATASET}.csv"',
        "fi",
        "for f in ${FILE_GLOB}; do",
        "  [ -f \"$f\" ] || continue",
        "  b=$(basename \"$f\" .csv)",
        '  cp "$f" "${FLAT_OUTDIR}/${b}__jobds_${INJECT_DATASET}__m_${INJECT_MASS_TAG}__s_${INJECT_STRENGTH_TAG}.csv"',
        "done",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(job_lines) + "\n")
    os.chmod(output_path, 0o755)
    print(f"Wrote injection SLURM job script to {output_path}")

    submit_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "submit_injection_all.sh")
    abs_job = os.path.abspath(output_path)

    submit_lines = [
        "#!/bin/bash",
        "# Submit one SLURM job per (dataset, mass, strength) injection point",
        f'JOB_SCRIPT="{abs_job}"',
        f'BASE_OUTPUT_DIR="{output_root}"',
        "",
        "mkdir -p logs",
        "",
    ]

    n_jobs = 0
    n_skipped = 0
    for ds in datasets:
        ds_range = (mass_ranges_by_dataset or {}).get(str(ds))
        for mass in masses:
            mass_f = float(mass)
            if ds_range is not None and str(ds) != "combined":
                lo, hi = float(ds_range[0]), float(ds_range[1])
                if mass_f < lo or mass_f > hi:
                    n_skipped += 1
                    continue
            mass_str = f"{mass_f:.6f}".rstrip("0").rstrip(".")
            mass_tag = mass_str.replace("-", "m").replace(".", "p")
            for strength in strengths:
                strength_str = f"{float(strength):.6g}"
                strength_tag = strength_str.replace("-", "m").replace(".", "p")
                submit_lines.append(
                    "sbatch --export=ALL,"
                    f"INJECT_DATASET={ds},"
                    f"INJECT_MASS={mass_str},"
                    f"INJECT_MASS_TAG={mass_tag},"
                    f"INJECT_STRENGTH={strength_str},"
                    f"INJECT_STRENGTH_TAG={strength_tag},"
                    "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR} "
                    "\"${JOB_SCRIPT}\""
                )
                n_jobs += 1

    submit_lines.extend([
        "",
        f'echo "Submitted {n_jobs} injection jobs."',
        f'echo "Skipped {n_skipped} out-of-range (dataset,mass) combinations."',
    ])

    with open(submit_path, "w") as f:
        f.write("\n".join(submit_lines) + "\n")
    os.chmod(submit_path, 0o755)
    print(f"Wrote injection submission loop script to {submit_path}")

    return output_path, submit_path, n_jobs

def get_mass_range_for_task(
    datasets: dict,
    mass_step: float,
    task_id: int,
    n_tasks: int,
) -> tuple:
    """Get the mass range for a specific array task.

    Args:
        datasets: Dictionary of dataset configurations
        mass_step: Mass step size (GeV)
        task_id: Array task ID (0-indexed)
        n_tasks: Total number of tasks

    Returns:
        Tuple of (mass_min, mass_max) for this task
    """
    lo = min([d.m_low for d in datasets.values()])
    hi = max([d.m_high for d in datasets.values()])

    all_masses = np.arange(lo, hi + 0.5 * mass_step, mass_step)
    all_masses = np.round(all_masses, 3)

    n_masses = len(all_masses)
    chunk_size = n_masses // n_tasks
    remainder = n_masses % n_tasks

    # Distribute remainder across first tasks
    if task_id < remainder:
        start_idx = task_id * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = task_id * chunk_size + remainder
        end_idx = start_idx + chunk_size

    if start_idx >= n_masses:
        return None, None

    end_idx = min(end_idx, n_masses)

    mass_min = float(all_masses[start_idx])
    mass_max = float(all_masses[end_idx - 1])

    return mass_min, mass_max


def infer_n_tasks_from_output_dir(output_dir: str) -> Optional[int]:
    """Infer the total task count from ``task_####`` folders in an output directory."""
    if not os.path.isdir(output_dir):
        return None

    task_ids: List[int] = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"task_(\d{4})", str(name))
        if m:
            task_ids.append(int(m.group(1)))

    if not task_ids:
        return None

    # Tasks are generated with contiguous IDs [0, N_TASKS-1].
    return max(task_ids) + 1


def get_task_ids_for_masses(
    datasets: dict,
    mass_step: float,
    n_tasks: int,
    masses_gev: List[float],
) -> List[int]:
    """Map requested masses to the SLURM task IDs that own those mass points."""
    lo = min([d.m_low for d in datasets.values()])
    hi = max([d.m_high for d in datasets.values()])

    all_masses = np.arange(lo, hi + 0.5 * mass_step, mass_step)
    all_masses = np.round(all_masses, 3)
    n_masses = len(all_masses)

    if n_tasks <= 0:
        raise ValueError(f"n_tasks must be > 0, got {n_tasks}")

    task_ids = set()
    for req_mass in masses_gev:
        req_mass = float(np.round(req_mass, 3))
        idx = np.where(np.isclose(all_masses, req_mass, atol=1e-9))[0]
        if idx.size == 0:
            raise ValueError(
                f"Requested mass {req_mass:.3f} GeV is not on the scan grid "
                f"[{lo:.3f}, {hi:.3f}] with step {mass_step:.3f} GeV"
            )
        i = int(idx[0])

        chunk_size = n_masses // n_tasks
        remainder = n_masses % n_tasks

        boundary = (chunk_size + 1) * remainder
        if i < boundary:
            task_id = i // (chunk_size + 1)
        else:
            task_id = remainder + (i - boundary) // max(chunk_size, 1)

        task_ids.add(int(task_id))

    return sorted(task_ids)




def _combine_band_family(output_dir: str, output_prefix: str, stem: str, subset_cols: List[str]):
    """Combine per-task UL-band CSV files with matching stem pattern."""
    files = glob.glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)
    if not files:
        return {}

    out = {}
    by_name = {}
    pat = re.compile(rf"^{re.escape(stem)}_(.+)\.csv$")
    for f in files:
        base = os.path.basename(f)
        m = pat.match(base)
        if not m:
            continue
        # keep families disjoint: ul_bands_* should not absorb ul_bands_eps2_* or ul_bands_combined_*
        name = str(m.group(1))
        if stem == "ul_bands" and (name.startswith("eps2_") or name.startswith("combined_")):
            continue
        by_name.setdefault(name, []).append(f)

    for name, paths in by_name.items():
        dfs = []
        for fp in paths:
            try:
                dfs.append(pd.read_csv(fp))
            except Exception as e:
                print(f"Warning: Could not read {fp}: {e}")
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        keep = [c for c in subset_cols if c in df.columns]
        if keep:
            df = df.drop_duplicates(subset=keep)
        sort_cols = [c for c in ["dataset", "dataset_set", "mass_GeV"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        out_path = os.path.join(output_dir, f"{output_prefix}_{stem}_{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote combined band table to {out_path}")
        out[name] = out_path

    return out
def combine_results(output_dir: str, output_prefix: str = "combined") -> tuple:
    """Combine results from parallel SLURM jobs.

    Args:
        output_dir: Directory containing task output subdirectories
        output_prefix: Prefix for combined output files

    Returns:
        Tuple of (combined_single_df, combined_comb_df)
    """
    # Find all single results files
    single_files = glob.glob(os.path.join(output_dir, "**/results_single.csv"), recursive=True)
    comb_files = glob.glob(os.path.join(output_dir, "**/results_combined.csv"), recursive=True)

    if not single_files:
        print(f"No results_single.csv files found in {output_dir}")
        return None, None

    # Combine single results
    single_dfs = []
    for f in single_files:
        try:
            df = pd.read_csv(f)
            single_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if single_dfs:
        df_single = pd.concat(single_dfs, ignore_index=True)
        df_single = df_single.drop_duplicates(subset=["dataset", "mass_GeV"])
        df_single = df_single.sort_values(["dataset", "mass_GeV"]).reset_index(drop=True)

        single_out = os.path.join(output_dir, f"{output_prefix}_single.csv")
        df_single.to_csv(single_out, index=False)
        print(f"Wrote combined single results to {single_out}")
    else:
        df_single = None

    # Combine combined results
    comb_dfs = []
    for f in comb_files:
        try:
            df = pd.read_csv(f)
            comb_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if comb_dfs:
        df_comb = pd.concat(comb_dfs, ignore_index=True)
        df_comb = df_comb.drop_duplicates(subset=["mass_GeV"])
        df_comb = df_comb.sort_values("mass_GeV").reset_index(drop=True)

        comb_out = os.path.join(output_dir, f"{output_prefix}_combined.csv")
        df_comb.to_csv(comb_out, index=False)
        print(f"Wrote combined results to {comb_out}")
    else:
        df_comb = None

    combined_ul_bands = _combine_band_family(output_dir, output_prefix, "ul_bands", ["dataset", "mass_GeV"])
    combined_ul_bands_eps2 = _combine_band_family(output_dir, output_prefix, "ul_bands_eps2", ["dataset", "mass_GeV"])
    combined_ul_bands_combined = _combine_band_family(output_dir, output_prefix, "ul_bands_combined", ["dataset_set", "mass_GeV"])

    return df_single, df_comb, combined_ul_bands, combined_ul_bands_eps2, combined_ul_bands_combined
