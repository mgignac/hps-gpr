from pathlib import Path

from hps_gpr.slurm import generate_extraction_display_slurm_scripts


def test_generate_extraction_display_slurm_scripts_writes_expected_commands(tmp_path):
    job = tmp_path / "submit_extract_display.slurm"

    job_script, submit_script, n_jobs = generate_extraction_display_slurm_scripts(
        config_path="study_configs/config_2015_extraction_display_v15p8.yaml",
        output_path=str(job),
        dataset="combined",
        dataset_keys=["2015", "2016", "2021"],
        masses=[0.040, 0.080],
        strengths=[3.0, 5.0],
        output_root="outputs/extraction_display_batch",
        mass_range=(0.03, 0.25),
    )

    assert n_jobs == 4
    job_text = Path(job_script).read_text()
    submit_text = Path(submit_script).read_text()
    assert "hps-gpr extract-display" in job_text
    assert '--masses "${EXTRACT_MASS}"' in job_text
    assert '--strengths "${EXTRACT_STRENGTH}"' in job_text
    assert '--datasets "${EXTRACT_DATASET_KEYS}"' in job_text
    assert 'EXTRACT_DATASET="combined"' in submit_text
    assert 'EXTRACT_DATASET_KEYS="2015,2016,2021"' in submit_text
    assert "EXTRACT_MASS=0.08" in submit_text
