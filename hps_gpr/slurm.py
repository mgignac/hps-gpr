"""SLURM job generation and result combination utilities."""

import glob
import os
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

    return df_single, df_comb
