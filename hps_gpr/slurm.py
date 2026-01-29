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
) -> str:
    """Generate a SLURM array job script for parallel mass scans.

    Args:
        config_path: Path to configuration YAML file
        n_jobs: Number of array tasks to split the scan into
        output_path: Path to write the SLURM script
        job_name: SLURM job name
        partition: SLURM partition
        time_limit: Time limit per task
        memory: Memory per task
        conda_env: Conda environment to activate (optional)
        extra_sbatch: Additional SBATCH directives

    Returns:
        Path to generated script
    """
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{n_jobs - 1}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        "#SBATCH --output=logs/%A_%a.out",
        "#SBATCH --error=logs/%A_%a.err",
    ]

    if extra_sbatch:
        for directive in extra_sbatch:
            script_lines.append(f"#SBATCH {directive}")

    script_lines.append("")
    script_lines.append("# Create logs directory")
    script_lines.append("mkdir -p logs")
    script_lines.append("")

    if conda_env:
        script_lines.append(f"# Activate conda environment")
        script_lines.append(f"source $(conda info --base)/etc/profile.d/conda.sh")
        script_lines.append(f"conda activate {conda_env}")
        script_lines.append("")

    script_lines.extend([
        "# Run the scan for this array task",
        f"hps-gpr scan \\",
        f"    --config {config_path} \\",
        f"    --array-task $SLURM_ARRAY_TASK_ID \\",
        f"    --n-tasks {n_jobs}",
    ])

    script_content = "\n".join(script_lines) + "\n"

    with open(output_path, "w") as f:
        f.write(script_content)

    os.chmod(output_path, 0o755)
    print(f"Wrote SLURM script to {output_path}")

    return output_path


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
