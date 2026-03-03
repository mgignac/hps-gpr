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

    if bool(getattr(cfg, "make_ul_bands", False)):
        from .bands import expected_ul_bands_for_dataset, expected_ul_bands_for_combination
        from .plotting import plot_ul_bands

        masses_all = np.round(np.arange(min(d.m_low for d in datasets.values()),
                                        max(d.m_high for d in datasets.values()) + cfg.mass_step_gev / 2,
                                        cfg.mass_step_gev), 3)
        if mass_min is not None:
            masses_all = masses_all[masses_all >= float(mass_min)]
        if mass_max is not None:
            masses_all = masses_all[masses_all <= float(mass_max)]

        ds_key = str(getattr(cfg, "run_limit_bands_on", "2015"))
        if ds_key in datasets:
            ds = datasets[ds_key]
            masses_ds = [float(m) for m in masses_all if float(ds.m_low) <= float(m) <= float(ds.m_high)]
            if masses_ds:
                print(f"\nComputing UL bands for {ds_key} ({len(masses_ds)} masses)")
                df_bands = expected_ul_bands_for_dataset(ds, masses_ds, cfg)
                out_csv = os.path.join(cfg.output_dir, f"ul_bands_{ds.key}.csv")
                out_png = os.path.join(cfg.output_dir, f"ul_bands_{ds.key}.png")
                out_eps2 = os.path.join(cfg.output_dir, f"ul_bands_eps2_{ds.key}.png")
                df_bands.to_csv(out_csv, index=False)
                plot_ul_bands(df_bands, use_eps2=False, title=f"Expected UL bands ({ds.key})", outpath=out_png)
                if bool(getattr(cfg, "make_eps2_bands", True)):
                    plot_ul_bands(df_bands, use_eps2=True, title=f"Expected $\epsilon^2$ UL bands ({ds.key})", outpath=out_eps2)

        if bool(getattr(cfg, "do_combined_bands", False)):
            keys = list(datasets.keys())
            print(f"\nComputing combined UL bands for datasets: {keys}")
            df_cb = expected_ul_bands_for_combination(keys, datasets, [float(m) for m in masses_all], cfg)
            out_csv_c = os.path.join(cfg.output_dir, "ul_bands_combined_all.csv")
            out_png_c = os.path.join(cfg.output_dir, "ul_bands_combined_all.png")
            df_cb.to_csv(out_csv_c, index=False)
            plot_ul_bands(df_cb, use_eps2=True, title="Expected combined $\epsilon^2$ UL bands", outpath=out_png_c)

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
    help="Dataset key (2015, 2016, 2021, or combined)",
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
    "--n-toys",
    type=int,
    default=1,
    show_default=True,
    help="Number of pseudoexperiments per (mass,strength) point",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Override output directory from config",
)
def inject(config, dataset, masses, strengths, n_toys, output_dir):
    """Run injection/extraction study."""
    import pandas as pd

    from .config import load_config
    from .dataset import make_datasets
    from .injection import (
        run_injection_extraction_toys,
        summarize_injection_grid,
        combine_injection_toy_tables,
        _combined_mass_support_summary,
        format_combined_mass_support_summary,
    )
    from .plotting import (
        ensure_dir,
        plot_linearity,
        plot_bias_vs_injected_strength,
        plot_pull_width,
        plot_coverage,
        plot_injection_heatmap,
    )

    cfg = load_config(config)

    if output_dir:
        cfg.output_dir = output_dir

    cfg.ensure_output_dir()
    datasets = make_datasets(cfg)

    available = list(datasets.keys())
    if dataset != "combined" and dataset not in datasets:
        print(f"Dataset '{dataset}' not found or not enabled. Available: {available + ['combined']}")
        sys.exit(1)

    mass_list = [float(m.strip()) for m in masses.split(",")]
    strength_list = [float(s.strip()) for s in strengths.split(",")] if strengths else [float(s) for s in cfg.inj_strengths]

    outdir = os.path.join(cfg.output_dir, "injection_extraction")
    ensure_dir(outdir)
    strengths_mode = str(getattr(cfg, "inj_strength_mode", "absolute")).lower().strip()

    if dataset == "combined":
        print("Running combined injection study over all enabled datasets")
        print(f"Masses: {mass_list}")
        print(f"Strengths: {strength_list}")

        df_map = {}
        for key, ds in datasets.items():
            print(f"  -> {ds.label}")
            df_map[key] = run_injection_extraction_toys(
                ds,
                cfg,
                masses=mass_list,
                strengths=[float(x) for x in strength_list],
                n_toys=int(n_toys),
                strengths_mode=strengths_mode,
            )

        mass_policy = str(getattr(cfg, "inj_combined_mass_policy", "intersection")).strip().lower()
        min_n_contrib = int(getattr(cfg, "inj_combined_min_n_contrib", 2))
        support = _combined_mass_support_summary(
            df_map,
            mass_policy=mass_policy,
            min_n_contrib=min_n_contrib,
        )
        print(format_combined_mass_support_summary(support))

        df_comb_toys = combine_injection_toy_tables(
            df_map,
            mass_policy=mass_policy,
            min_n_contrib=min_n_contrib,
        )
        if not df_comb_toys.empty:
            comb_toys_path = os.path.join(outdir, "inj_extract_toys_combined.csv")
            df_comb_toys.to_csv(comb_toys_path, index=False)
            print(f"Wrote {comb_toys_path}")

        summary_frames = []
        for key, dfi in df_map.items():
            dsum = summarize_injection_grid(dfi)
            dsum["dataset"] = str(key)
            summary_frames.append(dsum)
            dsum.to_csv(os.path.join(outdir, f"inj_extract_summary_{key}.csv"), index=False)

        if not df_comb_toys.empty:
            dsum_c = summarize_injection_grid(df_comb_toys)
            dsum_c["dataset"] = "combined"
            summary_frames.append(dsum_c)
            dsum_c.to_csv(os.path.join(outdir, "inj_extract_summary_combined.csv"), index=False)

        df_sum = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        if not df_sum.empty:
            df_sum.to_csv(os.path.join(outdir, "inj_extract_summary_all.csv"), index=False)

            xvar = "inj_nsigma" if "inj_nsigma" in df_sum.columns and np.isfinite(df_sum["inj_nsigma"]).any() else "strength"
            preferred_order = ["2015", "2016", "combined"]
            present = [str(x) for x in df_sum["dataset"].astype(str).unique()]
            ds_order = [d for d in preferred_order if d in present] + sorted(d for d in present if d not in preferred_order)

            for ds_key in ds_order:
                sub = df_sum[df_sum["dataset"].astype(str) == ds_key].copy()
                plot_linearity(sub, xvar=xvar, title=f"{ds_key}: linearity", outpath=os.path.join(outdir, f"linearity_{ds_key}.png"))
                plot_bias_vs_injected_strength(sub, xvar=xvar, title=f"{ds_key}: bias", outpath=os.path.join(outdir, f"bias_{ds_key}.png"))
                plot_pull_width(sub, xvar=xvar, title=f"{ds_key}: pull width", outpath=os.path.join(outdir, f"pull_width_{ds_key}.png"))
                plot_coverage(sub, xvar=xvar, title=f"{ds_key}: coverage", outpath=os.path.join(outdir, f"coverage_{ds_key}.png"))
                plot_injection_heatmap(sub, value_col="pull_mean", dataset_filter=ds_key, title=f"{ds_key}: mean pull heatmap", outpath=os.path.join(outdir, f"heatmap_pull_mean_{ds_key}.png"))
                plot_injection_heatmap(sub, value_col="pull_std", dataset_filter=ds_key, title=f"{ds_key}: pull width heatmap", outpath=os.path.join(outdir, f"heatmap_pull_width_{ds_key}.png"))

        print(f"\nSummary rows (all datasets + combined): {len(df_sum)}")
        if not df_sum.empty:
            print(df_sum.head(20).to_string())

    else:
        ds = datasets[dataset]
        print(f"Running injection study for {ds.label}")
        print(f"Masses: {mass_list}")
        print(f"Strengths: {strength_list}")

        df = run_injection_extraction_toys(
            ds,
            cfg,
            masses=mass_list,
            strengths=[float(x) for x in strength_list],
            n_toys=int(n_toys),
            strengths_mode=strengths_mode,
        )

        df_sum = summarize_injection_grid(df)
        out_sum = os.path.join(outdir, f"inj_extract_summary_{ds.key}.csv")
        df_sum.to_csv(out_sum, index=False)

        xvar = "inj_nsigma" if "inj_nsigma" in df_sum.columns and np.isfinite(df_sum["inj_nsigma"]).any() else "strength"
        plot_linearity(df_sum, xvar=xvar, title=f"{ds.label}: linearity", outpath=os.path.join(outdir, f"linearity_{ds.key}.png"))
        plot_bias_vs_injected_strength(df_sum, xvar=xvar, title=f"{ds.label}: bias", outpath=os.path.join(outdir, f"bias_{ds.key}.png"))
        plot_pull_width(df_sum, xvar=xvar, title=f"{ds.label}: pull width", outpath=os.path.join(outdir, f"pull_width_{ds.key}.png"))
        plot_coverage(df_sum, xvar=xvar, title=f"{ds.label}: coverage", outpath=os.path.join(outdir, f"coverage_{ds.key}.png"))
        plot_injection_heatmap(df_sum, value_col="pull_mean", title=f"{ds.label}: mean pull heatmap", outpath=os.path.join(outdir, f"heatmap_pull_mean_{ds.key}.png"))
        plot_injection_heatmap(df_sum, value_col="pull_std", title=f"{ds.label}: pull width heatmap", outpath=os.path.join(outdir, f"heatmap_pull_width_{ds.key}.png"))

        print(f"\nToy-level rows: {len(df)}")
        print(f"Summary rows: {len(df_sum)}")
        print(f"Wrote summary table: {out_sum}")
        print(df_sum.head(20).to_string())



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
@click.option(
    "--account",
    help="SLURM account/project to charge",
)
def slurm_gen(config, n_jobs, output, job_name, partition, time, memory, conda_env, account):
    """Generate SLURM array job script."""
    from .slurm import generate_slurm_script

    extra = [f"--account={account}"] if account else None

    job_script, submit_script = generate_slurm_script(
        config_path=config,
        n_jobs=n_jobs,
        output_path=output,
        job_name=job_name,
        partition=partition,
        time_limit=time,
        memory=memory,
        conda_env=conda_env,
        extra_sbatch=extra,
    )
    print(f"\nTo submit all {n_jobs} jobs, run:")
    print(f"  bash {submit_script}")




@main.command("slurm-gen-inject")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--datasets",
    required=True,
    help="Comma-separated dataset keys for injection jobs (e.g. 2015,2016,combined)",
)
@click.option(
    "--masses",
    required=True,
    help="Comma-separated masses (GeV)",
)
@click.option(
    "--strengths",
    required=True,
    help="Comma-separated injection strengths",
)
@click.option(
    "--n-toys",
    type=int,
    default=10000,
    show_default=True,
    help="Pseudoexperiments per job",
)
@click.option(
    "--output",
    "-o",
    default="submit_injection.slurm",
    help="Output SLURM script path",
)
@click.option(
    "--job-name",
    default="hps-gpr-inj",
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
@click.option(
    "--account",
    help="SLURM account/project to charge",
)
def slurm_gen_inject(config, datasets, masses, strengths, n_toys, output, job_name, partition, time, memory, conda_env, account):
    """Generate SLURM scripts for per-(dataset,mass,strength) injection jobs."""
    from .config import load_config
    from .slurm import generate_injection_slurm_scripts

    cfg = load_config(config)

    dataset_list = [d.strip() for d in str(datasets).split(",") if d.strip()]
    mass_list = [float(m.strip()) for m in str(masses).split(",") if m.strip()]
    strength_list = [float(s.strip()) for s in str(strengths).split(",") if s.strip()]

    if not dataset_list:
        raise click.BadParameter("No datasets provided", param_hint="--datasets")
    if not mass_list:
        raise click.BadParameter("No masses provided", param_hint="--masses")
    if not strength_list:
        raise click.BadParameter("No strengths provided", param_hint="--strengths")

    extra = [f"--account={account}"] if account else None

    ds_ranges = {
        "2015": tuple(cfg.range_2015),
        "2016": tuple(cfg.range_2016),
        "2021": tuple(cfg.range_2021),
    }

    job_script, submit_script, n_jobs = generate_injection_slurm_scripts(
        config_path=config,
        output_path=output,
        datasets=dataset_list,
        masses=mass_list,
        strengths=strength_list,
        n_toys=int(n_toys),
        output_root=cfg.output_dir,
        job_name=job_name,
        partition=partition,
        time_limit=time,
        memory=memory,
        conda_env=conda_env,
        extra_sbatch=extra,
        mass_ranges_by_dataset=ds_ranges,
    )
    print(f"\nPrepared {n_jobs} injection jobs.")
    print("To submit all jobs, run:")
    print(f"  bash {submit_script}")



@main.command("inject-plot")
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Root directory containing injection job outputs (injection_jobs/ or injection_flat/)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Directory to write merged injection tables and summary plots",
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Optional dataset filter (repeatable: 2015, 2016, combined)",
)
@click.option(
    "--write-merged-toys/--no-write-merged-toys",
    default=False,
    show_default=True,
    help="Write merged toy-level CSVs (can be very large)",
)
def inject_plot(input_dir, output_dir, dataset, write_merged_toys):
    """Merge distributed injection CSVs and produce publication-ready summary plots."""
    import glob
    import pandas as pd

    from .injection import summarize_injection_grid, combine_injection_toy_tables, _combined_mass_support_summary, format_combined_mass_support_summary
    from .plotting import (
        ensure_dir,
        plot_linearity,
        plot_bias_vs_injected_strength,
        plot_pull_width,
        plot_coverage,
        plot_injection_heatmap,
        plot_pull_histogram_by_mass,
        plot_pull_vs_mass,
    )

    ds_filter = {str(d).strip() for d in (dataset or []) if str(d).strip()}
    outdir = output_dir or os.path.join(input_dir, "injection_summary")
    ensure_dir(outdir)

    toy_paths = sorted(set(
        glob.glob(os.path.join(input_dir, "**", "injection_extraction", "inj_extract_toys_*.csv"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**", "inj_extract_toys_*.csv"), recursive=True)
    ))
    if not toy_paths:
        print(f"No toy-level injection CSVs found under {input_dir}")
        sys.exit(1)

    by_dataset = {}
    for fp in toy_paths:
        base = os.path.basename(fp)
        ds_token = base.replace("inj_extract_toys_", "").replace(".csv", "").strip()
        ds = ds_token.split("__", 1)[0]
        if ds_filter and ds not in ds_filter:
            continue
        try:
            dfi = pd.read_csv(fp)
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")
            continue
        if dfi.empty:
            continue
        if "dataset" not in dfi.columns:
            dfi["dataset"] = ds
        by_dataset.setdefault(ds, []).append(dfi)

    if not by_dataset:
        print("No valid injection toy tables loaded after filtering.")
        sys.exit(1)

    all_summaries = []
    toy_merged = {}
    mass_policy = "intersection"
    min_n_contrib = 2

    for ds, frames in sorted(by_dataset.items()):
        dft = pd.concat(frames, ignore_index=True)
        dft = dft.sort_values([c for c in ["mass_GeV", "strength", "toy"] if c in dft.columns]).reset_index(drop=True)

        dedup_cols = [c for c in ["dataset", "mass_GeV", "strength", "toy"] if c in dft.columns]
        if dedup_cols:
            dft = dft.drop_duplicates(subset=dedup_cols, keep="last")

        toy_merged[str(ds)] = dft.copy()

        if write_merged_toys:
            toys_out = os.path.join(outdir, f"inj_extract_toys_{ds}.csv")
            dft.to_csv(toys_out, index=False)
            print(f"Wrote {toys_out}")

        dsum = summarize_injection_grid(dft)
        dsum["dataset"] = str(ds)
        sum_out = os.path.join(outdir, f"inj_extract_summary_{ds}.csv")
        dsum.to_csv(sum_out, index=False)
        all_summaries.append(dsum)
        print(f"Wrote {sum_out}")

    non_combined = {k: pd.concat(v, ignore_index=True) for k, v in by_dataset.items() if k != "combined"}
    if len(non_combined) >= 2:
        support = _combined_mass_support_summary(
            non_combined,
            mass_policy=mass_policy,
            min_n_contrib=min_n_contrib,
        )
        print(format_combined_mass_support_summary(support))

        df_comb = combine_injection_toy_tables(
            non_combined,
            mass_policy=mass_policy,
            min_n_contrib=min_n_contrib,
        )
        if not df_comb.empty:
            if write_merged_toys:
                toys_out = os.path.join(outdir, "inj_extract_toys_combined.csv")
                df_comb.to_csv(toys_out, index=False)
                print(f"Wrote {toys_out}")
            dsum_c = summarize_injection_grid(df_comb)
            dsum_c["dataset"] = "combined"
            sum_out_c = os.path.join(outdir, "inj_extract_summary_combined.csv")
            dsum_c.to_csv(sum_out_c, index=False)
            all_summaries = [s for s in all_summaries if not ("dataset" in s.columns and (s["dataset"].astype(str) == "combined").all())]
            all_summaries.append(dsum_c)
            print(f"Wrote {sum_out_c}")

    df_sum = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    if df_sum.empty:
        print("No summary rows produced.")
        sys.exit(1)

    all_out = os.path.join(outdir, "inj_extract_summary_all.csv")
    df_sum.to_csv(all_out, index=False)
    print(f"Wrote {all_out}")

    xvar = "inj_nsigma" if "inj_nsigma" in df_sum.columns and np.isfinite(df_sum["inj_nsigma"]).any() else "strength"

    # Cross-dataset overlays
    plot_linearity(df_sum, xvar=xvar, title="Injection linearity (all datasets)", outpath=os.path.join(outdir, "linearity_all.png"))
    plot_bias_vs_injected_strength(df_sum, xvar=xvar, title="Injection bias (all datasets)", outpath=os.path.join(outdir, "bias_all.png"))
    plot_pull_width(df_sum, xvar=xvar, title="Pull width (all datasets)", outpath=os.path.join(outdir, "pull_width_all.png"))
    plot_coverage(df_sum, xvar=xvar, title="Coverage (all datasets)", outpath=os.path.join(outdir, "coverage_all.png"))

    preferred_order = ["2015", "2016", "combined"]
    present = [str(x) for x in df_sum["dataset"].astype(str).unique()]
    ds_order = [d for d in preferred_order if d in present] + sorted(d for d in present if d not in preferred_order)

    for required_ds in preferred_order:
        if required_ds not in present:
            print(f"Warning: no summary rows found for required dataset '{required_ds}'")

    for ds_key in ds_order:
        sub = df_sum[df_sum["dataset"].astype(str) == ds_key].copy()
        dft = toy_merged.get(ds_key, pd.DataFrame())
        plot_linearity(sub, xvar=xvar, title=f"{ds_key}: linearity", outpath=os.path.join(outdir, f"linearity_{ds_key}.png"))
        plot_bias_vs_injected_strength(sub, xvar=xvar, title=f"{ds_key}: bias", outpath=os.path.join(outdir, f"bias_{ds_key}.png"))
        plot_pull_width(sub, xvar=xvar, title=f"{ds_key}: pull width", outpath=os.path.join(outdir, f"pull_width_{ds_key}.png"))
        plot_coverage(sub, xvar=xvar, title=f"{ds_key}: coverage", outpath=os.path.join(outdir, f"coverage_{ds_key}.png"))
        plot_injection_heatmap(sub, value_col="pull_mean", dataset_filter=ds_key, title=f"{ds_key}: mean pull heatmap", outpath=os.path.join(outdir, f"heatmap_pull_mean_{ds_key}.png"))
        plot_injection_heatmap(sub, value_col="pull_std", dataset_filter=ds_key, title=f"{ds_key}: pull width heatmap", outpath=os.path.join(outdir, f"heatmap_pull_width_{ds_key}.png"))

        if not dft.empty:
            plot_pull_vs_mass(dft, dataset_key=ds_key, title=f"{ds_key}: pull mean/width vs mass", outpath=os.path.join(outdir, f"pull_vs_mass_{ds_key}.png"))
            hist_dir = os.path.join(outdir, f"pull_hist_{ds_key}")
            ensure_dir(hist_dir)
            paths = plot_pull_histogram_by_mass(dft, dataset_key=ds_key, group_by_strength=True, pvalue_method="ks", outdir=hist_dir)
            print(f"Wrote {len(paths)} pull-histogram plots for {ds_key} to {hist_dir}")

    print(f"\nSummary rows: {len(df_sum)}")
    print(df_sum.head(20).to_string())

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
    """Combine results from parallel SLURM jobs and generate summary plot suites."""
    import pandas as pd

    from .slurm import combine_results
    from .plotting import (
        ensure_dir,
        plot_ul_bands,
        plot_observed_ul_only,
        plot_ul_pvalues,
        plot_ul_pvalue_components,
        plot_analytic_p0,
        plot_Z_local_global,
        plot_injection_heatmap,
        plot_linearity,
        plot_pull_width,
    )

    df_single, df_comb, bands_a, bands_eps2, bands_comb = combine_results(output_dir, prefix)

    if df_single is not None:
        print(f"\nCombined single results: {len(df_single)} rows")
    if df_comb is not None:
        print(f"Combined results: {len(df_comb)} rows")

    for name, path in (bands_a or {}).items():
        try:
            df = pd.read_csv(path)
            png = path[:-4] + ".png"
            plot_ul_bands(df, use_eps2=False, title=f"Expected signal-yield UL bands ({name})", outpath=png)
            print(f"Wrote {png}")
        except Exception as e:
            print(f"Warning: could not plot A bands for {name}: {e}")

    for name, path in (bands_eps2 or {}).items():
        try:
            df = pd.read_csv(path)
            png = path[:-4] + ".png"
            plot_ul_bands(df, use_eps2=True, title=f"Expected $\epsilon^2$ UL bands ({name})", outpath=png)
            print(f"Wrote {png}")
        except Exception as e:
            print(f"Warning: could not plot eps2 bands for {name}: {e}")

    for name, path in (bands_comb or {}).items():
        try:
            df = pd.read_csv(path)
            png = path[:-4] + ".png"
            plot_ul_bands(df, use_eps2=True, title=f"Expected combined $\epsilon^2$ UL bands ({name})", outpath=png)
            print(f"Wrote {png}")
        except Exception as e:
            print(f"Warning: could not plot combined bands for {name}: {e}")

    # Publication-style summary suites from merged UL-band CSVs.
    # Priority: explicit combination bands -> eps2 bands -> generic UL-band tables.
    summary_inputs = dict((bands_comb or {}))
    if not summary_inputs:
        summary_inputs = dict((bands_eps2 or {}))
    if not summary_inputs:
        summary_inputs = dict((bands_a or {}))

    for name, path in summary_inputs.items():
        try:
            df = pd.read_csv(path).sort_values("mass_GeV").reset_index(drop=True)
            tag = str(name).replace("__", "_").strip("_")
            suite_dir = os.path.join(output_dir, f"summary_combined_{tag}")
            ensure_dir(suite_dir)

            alpha_vals = df["cls_alpha"].to_numpy(float) if "cls_alpha" in df.columns else np.array([0.05])
            alpha_vals = alpha_vals[np.isfinite(alpha_vals)]
            alpha_used = float(alpha_vals[0]) if alpha_vals.size else 0.05
            cl_pct = int(round(100.0 * (1.0 - alpha_used)))

            plot_ul_bands(
                df,
                use_eps2=True,
                title=f"Expected/observed {cl_pct}% CL upper limits on $\epsilon^2$ ({tag})",
                outpath=os.path.join(suite_dir, "ul_bands_eps2_obsexp.png"),
            )
            if "A_obs" in df.columns or "ul_A_obs" in df.columns:
                plot_ul_bands(
                    df,
                    use_eps2=False,
                    title=f"Expected/observed {cl_pct}% CL upper limits on signal yield ({tag})",
                    outpath=os.path.join(suite_dir, "ul_bands_signal_yield_obsexp.png"),
                )

            plot_observed_ul_only(
                df,
                y="eps2",
                title=f"Observed {cl_pct}% CL upper limit on $\epsilon^2$ ({tag})",
                outpath=os.path.join(suite_dir, "ul_observed_only_eps2.png"),
            )
            if "A_obs" in df.columns or "ul_A_obs" in df.columns:
                plot_observed_ul_only(
                    df,
                    y="yield",
                    title=f"Observed {cl_pct}% CL upper limit on signal yield ({tag})",
                    outpath=os.path.join(suite_dir, "ul_observed_only_signal_yield.png"),
                )

            plot_ul_pvalues(
                df,
                title=f"UL toy-tail p-values ({tag})",
                outpath=os.path.join(suite_dir, "ul_pvalues.png"),
            )
            plot_ul_pvalue_components(
                df,
                title=f"UL toy-tail p-value components + local/global references ({tag})",
                outpath=os.path.join(suite_dir, "ul_pvalues_components_local_global_refs.png"),
            )

            if "p0_analytic" in df.columns:
                lee_width = float(df["bands_train_exclude_nsigma"].dropna().iloc[0]) if "bands_train_exclude_nsigma" in df.columns and df["bands_train_exclude_nsigma"].notna().any() else 1.96
                plot_analytic_p0(
                    df,
                    title=f"Analytic local/global p0 vs mass ({tag})",
                    outpath=os.path.join(suite_dir, "p0_analytic_local_global.png"),
                    apply_lee=True,
                    lee_method="sidak",
                    indep_width_sigma=lee_width,
                )
                plot_Z_local_global(
                    df,
                    title=f"Local/global Z vs mass ({tag})",
                    outpath=os.path.join(suite_dir, "Z_local_global.png"),
                    apply_lee=True,
                    lee_method="sidak",
                    indep_width_sigma=lee_width,
                )

            # Add per-dataset UL + local/global suites into combined summary folder
            for ds_name, ds_path in (bands_a or {}).items():
                try:
                    dfa = pd.read_csv(ds_path).sort_values("mass_GeV").reset_index(drop=True)
                    plot_ul_bands(
                        dfa,
                        use_eps2=False,
                        title=f"{ds_name}: expected/observed {cl_pct}% CL signal-yield UL",
                        outpath=os.path.join(suite_dir, f"{ds_name}_UL_sig_yield_bands.png"),
                    )
                except Exception as e:
                    print(f"Warning: dataset signal-yield UL plot failed for {ds_name}: {e}")
            for ds_name, ds_path in (bands_eps2 or {}).items():
                try:
                    dfe = pd.read_csv(ds_path).sort_values("mass_GeV").reset_index(drop=True)
                    plot_ul_bands(
                        dfe,
                        use_eps2=True,
                        title=f"{ds_name}: expected/observed {cl_pct}% CL epsilon^2 UL",
                        outpath=os.path.join(suite_dir, f"{ds_name}_UL_eps2_yield_bands.png"),
                    )
                except Exception as e:
                    print(f"Warning: dataset eps2 UL plot failed for {ds_name}: {e}")

            if df_single is not None and len(df_single):
                for ds_name, sub in df_single.groupby("dataset"):
                    try:
                        sub = sub.sort_values("mass_GeV").reset_index(drop=True)
                        if "p0_analytic" in sub.columns:
                            plot_analytic_p0(
                                sub,
                                title=f"{ds_name}: analytic local/global p0",
                                outpath=os.path.join(suite_dir, f"{ds_name}_p0_local_global.png"),
                                apply_lee=True,
                                lee_method="sidak",
                                indep_width_sigma=lee_width,
                            )
                            plot_Z_local_global(
                                sub,
                                title=f"{ds_name}: local/global Z",
                                outpath=os.path.join(suite_dir, f"{ds_name}_Z_local_global.png"),
                                apply_lee=True,
                                lee_method="sidak",
                                indep_width_sigma=lee_width,
                            )
                    except Exception as e:
                        print(f"Warning: dataset local/global summary failed for {ds_name}: {e}")

            inj_all = os.path.join(output_dir, "injection_extraction", "inj_extract_summary_all.csv")
            if os.path.exists(inj_all):
                try:
                    dfi = pd.read_csv(inj_all)
                    xvar = "inj_nsigma" if "inj_nsigma" in dfi.columns and dfi["inj_nsigma"].notna().any() else "strength"
                    for ds_key in sorted(dfi["dataset"].astype(str).unique()):
                        sub = dfi[dfi["dataset"].astype(str) == ds_key].copy()
                        plot_injection_heatmap(sub, value_col="pull_mean", dataset_filter=ds_key, title=f"{ds_key}: mean pull heatmap", outpath=os.path.join(suite_dir, f"{ds_key}_heatmap_pull_mean.png"))
                        plot_injection_heatmap(sub, value_col="pull_std", dataset_filter=ds_key, title=f"{ds_key}: pull width heatmap", outpath=os.path.join(suite_dir, f"{ds_key}_heatmap_pull_width.png"))
                        plot_linearity(sub, xvar=xvar, title=f"{ds_key}: linearity", outpath=os.path.join(suite_dir, f"{ds_key}_linearity.png"))
                        plot_pull_width(sub, xvar=xvar, title=f"{ds_key}: pull width", outpath=os.path.join(suite_dir, f"{ds_key}_pull_width.png"))
                except Exception as e:
                    print(f"Warning: could not generate injection summary products: {e}")

            print(f"Wrote summary plot suite: {suite_dir}")
        except Exception as e:
            print(f"Warning: could not generate summary suite for {name}: {e}")




@main.command("re-run-2016-bands")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--mass",
    "-mass",
    required=True,
    multiple=True,
    type=float,
    help="Mass(es) in MeV to re-run (repeat option, e.g. -mass 37 -mass 48)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory containing task_#### subdirectories (defaults to config output_dir)",
)
@click.option(
    "--n-tasks",
    type=int,
    help="Total number of original SLURM tasks (auto-inferred from task_#### folders when omitted)",
)
def re_run_2016_bands(config, mass, output_dir, n_tasks):
    """Compatibility wrapper for re-running selected masses (MeV)."""
    ctx = click.get_current_context()
    ctx.invoke(re_run, config=config, masses=mass, output_dir=output_dir, n_tasks=n_tasks)

@main.command("re-run")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "--masses",
    "-m",
    required=True,
    multiple=True,
    type=float,
    help="Mass(es) in MeV to re-run (repeat option, e.g. -m 37 -m 48)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory containing task_#### subdirectories (defaults to config output_dir)",
)
@click.option(
    "--n-tasks",
    type=int,
    help="Total number of original SLURM tasks (auto-inferred from task_#### folders when omitted)",
)
def re_run(config, masses, output_dir, n_tasks):
    """Re-run one or more masses by re-executing the owning task IDs."""
    from .config import load_config
    from .dataset import make_datasets
    from .slurm import infer_n_tasks_from_output_dir, get_task_ids_for_masses

    cfg = load_config(config)
    run_outdir = output_dir or cfg.output_dir

    if not masses:
        print("No masses were provided. Use --masses/-m with MeV values (e.g. -m 37 -m 48).")
        sys.exit(1)

    datasets = make_datasets(cfg)
    if not datasets:
        print("No datasets enabled. Check configuration.")
        sys.exit(1)

    if n_tasks is None:
        n_tasks = infer_n_tasks_from_output_dir(run_outdir)
        if n_tasks is None:
            print(
                "Could not infer --n-tasks from output directory. "
                "Either provide --n-tasks or ensure task_#### folders exist."
            )
            sys.exit(1)

    masses_gev = [float(m) / 1000.0 for m in masses]
    try:
        task_ids = get_task_ids_for_masses(datasets, cfg.mass_step_gev, int(n_tasks), masses_gev)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not task_ids:
        print("No tasks selected for re-run.")
        return

    print(f"Re-running tasks in {run_outdir}: {task_ids}")
    for tid in task_ids:
        print(f"\n[re-run] task_{tid:04d}")
        # Re-use the existing scan command path so output files are overwritten in-place.
        scan.callback(
            config=config,
            output_dir=run_outdir,
            mass_min=None,
            mass_max=None,
            array_task=int(tid),
            n_tasks=int(n_tasks),
        )


if __name__ == "__main__":
    main()
