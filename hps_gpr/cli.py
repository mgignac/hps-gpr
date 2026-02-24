"""Command-line interface for HPS GPR analysis."""

import os
import sys

import click
import numpy as np


@click.group()
@click.version_option(version="0.1.0")
def main():
    """HPS Gaussian Process Regression analysis CLI."""
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Override output directory from config",
)
@click.option(
    "--mass-min",
    type=float,
    help="Minimum mass to scan (GeV)",
)
@click.option(
    "--mass-max",
    type=float,
    help="Maximum mass to scan (GeV)",
)
@click.option(
    "--array-task",
    type=int,
    help="SLURM array task ID (0-indexed)",
)
@click.option(
    "--n-tasks",
    type=int,
    help="Total number of SLURM array tasks",
)
def scan(config, output_dir, mass_min, mass_max, array_task, n_tasks):
    """Run the full mass scan."""
    from .config import load_config
    from .dataset import make_datasets, print_datasets
    from .validation import validate_datasets
    from .scan import run_scan
    from .plotting import plot_eps2_curves
    from .slurm import get_mass_range_for_task

    cfg = load_config(config)

    if output_dir:
        cfg.output_dir = output_dir

    # Handle SLURM array task
    if array_task is not None and n_tasks is not None:
        datasets = make_datasets(cfg)
        task_mass_min, task_mass_max = get_mass_range_for_task(
            datasets, cfg.mass_step_gev, array_task, n_tasks
        )
        if task_mass_min is None:
            print(f"Task {array_task}: No masses to process")
            return
        mass_min = task_mass_min
        mass_max = task_mass_max
        # Create task-specific output directory
        cfg.output_dir = os.path.join(cfg.output_dir, f"task_{array_task:04d}")
        print(f"Task {array_task}: Processing masses {mass_min:.3f} to {mass_max:.3f} GeV")

    cfg.ensure_output_dir()

    print(f"Loading configuration from {config}")
    print(f"Output directory: {cfg.output_dir}")

    datasets = make_datasets(cfg)
    print_datasets(datasets)

    if not datasets:
        print("No datasets enabled. Check configuration.")
        sys.exit(1)

    print("\nValidating datasets...")
    validate_datasets(datasets, cfg)

    print("\nRunning scan...")
    df_single, df_comb = run_scan(datasets, cfg, mass_min=mass_min, mass_max=mass_max)

    if cfg.save_plots:
        print("\nGenerating summary plots...")
        plot_eps2_curves(
            df_single, df_comb, os.path.join(cfg.output_dir, "summary_plots")
        )

    print("\nScan complete!")


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Dataset key (2015, 2016, or 2021)",
)
@click.option(
    "--n-toys",
    type=int,
    help="Number of toys per mass point (overrides config)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Override output directory from config",
)
def bands(config, dataset, n_toys, output_dir):
    """Compute expected upper limit bands for a dataset."""
    from .config import load_config
    from .dataset import make_datasets
    from .bands import expected_ul_bands_for_dataset
    from .plotting import plot_bands

    cfg = load_config(config)

    if output_dir:
        cfg.output_dir = output_dir
    if n_toys:
        cfg.ul_bands_toys = n_toys

    cfg.ensure_output_dir()

    datasets = make_datasets(cfg)

    if dataset not in datasets:
        available = list(datasets.keys())
        print(f"Dataset '{dataset}' not found or not enabled. Available: {available}")
        sys.exit(1)

    ds = datasets[dataset]

    print(f"Computing expected bands for {ds.label}")
    print(f"Mass range: {ds.m_low:.3f} to {ds.m_high:.3f} GeV")
    print(f"Number of toys: {cfg.ul_bands_toys}")

    masses = np.round(
        np.arange(ds.m_low, ds.m_high + cfg.mass_step_gev / 2, cfg.mass_step_gev), 3
    )

    df_bands = expected_ul_bands_for_dataset(ds, masses, cfg)

    out_csv = os.path.join(cfg.output_dir, f"ul_bands_{ds.key}.csv")
    df_bands.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Plot bands
    out_plot = os.path.join(cfg.output_dir, f"ul_bands_{ds.key}.png")
    plot_bands(df_bands, out_plot, column_prefix="A", ylabel="A UL", title=f"{ds.label}: Expected A UL bands")
    print(f"Wrote {out_plot}")

    if cfg.make_eps2_bands:
        out_plot_eps2 = os.path.join(cfg.output_dir, f"ul_bands_eps2_{ds.key}.png")
        plot_bands(
            df_bands,
            out_plot_eps2,
            column_prefix="eps2",
            ylabel=r"$\epsilon^2$ UL",
            title=f"{ds.label}: Expected $\\epsilon^2$ UL bands",
        )
        print(f"Wrote {out_plot_eps2}")


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Dataset key (2015, 2016, or 2021)",
)
@click.option(
    "--masses",
    "-m",
    required=True,
    help="Comma-separated list of masses (GeV)",
)
@click.option(
    "--strengths",
    "-s",
    help="Comma-separated list of injection strengths (overrides config)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Override output directory from config",
)
def inject(config, dataset, masses, strengths, output_dir):
    """Run injection/extraction study."""
    from .config import load_config
    from .dataset import make_datasets
    from .injection import run_injection_extraction

    cfg = load_config(config)

    if output_dir:
        cfg.output_dir = output_dir

    cfg.ensure_output_dir()

    datasets = make_datasets(cfg)

    if dataset not in datasets:
        available = list(datasets.keys())
        print(f"Dataset '{dataset}' not found or not enabled. Available: {available}")
        sys.exit(1)

    ds = datasets[dataset]

    # Parse masses
    mass_list = [float(m.strip()) for m in masses.split(",")]

    # Parse strengths
    if strengths:
        strength_list = [int(s.strip()) for s in strengths.split(",")]
    else:
        strength_list = cfg.inj_strengths

    print(f"Running injection study for {ds.label}")
    print(f"Masses: {mass_list}")
    print(f"Strengths: {strength_list}")

    df = run_injection_extraction(ds, mass_list, strength_list, cfg)

    print(f"\nResults summary:")
    print(df.head(20).to_string())


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Override output directory from config",
)
def test(config, output_dir):
    """Run smoke test on a single mass point."""
    from .config import load_config
    from .dataset import make_datasets, print_datasets
    from .validation import validate_datasets
    from .evaluation import evaluate_single_dataset
    from .plotting import ensure_dir, plot_full_range

    cfg = load_config(config)

    if output_dir:
        cfg.output_dir = output_dir

    cfg.ensure_output_dir()

    print(f"Loading configuration from {config}")
    datasets = make_datasets(cfg)
    print_datasets(datasets)

    if not datasets:
        print("No datasets enabled. Check configuration.")
        sys.exit(1)

    print("\nValidating datasets...")
    validate_datasets(datasets, cfg)

    # Pick test mass
    lows = [d.m_low for d in datasets.values()]
    highs = [d.m_high for d in datasets.values()]
    lo = max(lows)
    hi = min(highs)

    if lo < hi:
        test_mass = float(np.round(0.5 * (lo + hi), 3))
    else:
        d0 = list(datasets.values())[0]
        test_mass = float(np.round(0.5 * (d0.m_low + d0.m_high), 3))

    print(f"\nTest mass: {test_mass:.3f} GeV")

    for key, ds in datasets.items():
        if ds.m_low <= test_mass <= ds.m_high:
            print(f"\n--- Testing {key} ({ds.label}) ---")
            try:
                res, pred, _ = evaluate_single_dataset(ds, test_mass, cfg, do_extraction=True)
                print(f"A_up = {res.A_up:.2f}  eps2_up = {res.eps2_up:.3e}")
                print(f"p0 = {res.p0_analytic:.3e}  Z = {res.Z_analytic:.2f}")
                print(f"A_hat = {res.A_hat:.2f} +/- {res.sigma_A:.2f}  success = {res.extract_success}")

                # Make one plot
                tmp_dir = os.path.join(cfg.output_dir, "smoke_test")
                ensure_dir(tmp_dir)
                plot_path = os.path.join(tmp_dir, f"{key}_fit_full.png")
                plot_full_range(ds, test_mass, pred, plot_path)
                print(f"Wrote {plot_path}")

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()

    print("\nSmoke test complete!")


@main.command("slurm-gen")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--n-jobs",
    "-n",
    required=True,
    type=int,
    help="Number of array tasks",
)
@click.option(
    "--output",
    "-o",
    default="submit.slurm",
    help="Output SLURM script path",
)
@click.option(
    "--job-name",
    default="hps-gpr",
    help="SLURM job name",
)
@click.option(
    "--partition",
    default="batch",
    help="SLURM partition",
)
@click.option(
    "--time",
    default="4:00:00",
    help="Time limit per task",
)
@click.option(
    "--memory",
    default="4G",
    help="Memory per task",
)
@click.option(
    "--conda-env",
    help="Conda environment to activate",
)
def slurm_gen(config, n_jobs, output, job_name, partition, time, memory, conda_env):
    """Generate SLURM array job script."""
    from .slurm import generate_slurm_script

    generate_slurm_script(
        config_path=config,
        n_jobs=n_jobs,
        output_path=output,
        job_name=job_name,
        partition=partition,
        time_limit=time,
        memory=memory,
        conda_env=conda_env,
    )


@main.command("slurm-combine")
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing task output subdirectories",
)
@click.option(
    "--prefix",
    default="combined",
    help="Prefix for combined output files",
)
def slurm_combine(output_dir, prefix):
    """Combine results from parallel SLURM jobs."""
    from .slurm import combine_results

    df_single, df_comb = combine_results(output_dir, prefix)

    if df_single is not None:
        print(f"\nCombined single results: {len(df_single)} rows")
    if df_comb is not None:
        print(f"Combined results: {len(df_comb)} rows")


if __name__ == "__main__":
    main()
