#!/bin/bash
#SBATCH --job-name=hps2015_10k
#SBATCH --array=0-119
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=/sdf/scratch/hps/epeets/output/logs/%A_%a.out
#SBATCH --error=/sdf/scratch/hps/epeets/output/logs/%A_%a.err

# Create logs directory
mkdir -p logs


# Run the scan for this array task
hps-gpr scan \
    --config config_2015_10k.yaml \
    --array-task $SLURM_ARRAY_TASK_ID \
    --n-tasks 120
