#!/bin/bash
#SBATCH --job-name=hps2015_2016_combined_bands_10k
#SBATCH --partition=milano
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/15_15_combined/%j.out
#SBATCH --error=logs/15_16_combined/%j.err
#SBATCH --account=hps:hps-prod

mkdir -p logs

# TASK_ID and N_TASKS are passed via --export at submission time
hps-gpr scan \
    --config config_2015_2016_combined_10k.yaml \
    --array-task ${TASK_ID} \
    --n-tasks ${N_TASKS}
