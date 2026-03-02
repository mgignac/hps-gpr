#!/bin/bash
# Submit 96 individual SLURM jobs for hps-gpr scan
N_TASKS=96
JOB_SCRIPT="/sdf/home/e/epeets/src/hps-gpr/submit_2015_2016_combined_bands_10k.slurm"

mkdir -p logs

for TASK_ID in $(seq 0 $(( N_TASKS - 1 ))); do
    sbatch --export=ALL,TASK_ID=${TASK_ID},N_TASKS=${N_TASKS} "${JOB_SCRIPT}"
done

echo "Submitted ${N_TASKS} jobs."
