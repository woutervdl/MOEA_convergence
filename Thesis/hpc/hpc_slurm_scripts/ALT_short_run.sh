#!/bin/bash
#SBATCH --job-name="ALT_SHORT_moea_run_1_min_test"
#SBATCH --time=00:10:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-17 #(3 problems × 3 algorithms × 2 seeds)

# This script should be submitted with sbatch --ntasks=X run.sh X
CORES=$1
NFE=3000

# Calculate indices
PROBLEM_IDX=$(( $SLURM_ARRAY_TASK_ID / 6 ))  # 3 algorithms × 2 seeds = 6 per problem
ALGORITHM_IDX=$(( ($SLURM_ARRAY_TASK_ID % 6) / 2 ))  # 2 seeds per algorithm

# Pass raw SLURM_ARRAY_TASK_ID to hpc_run.py
srun --export=ALL apptainer exec \
    --env PYTHONPATH=/opt/MOEA_convergence \
    --env LC_ALL=C \
    --env LANG=C \
    /scratch/wmvanderlinden/MOEA_convergence/moea.sif \
    python /opt/MOEA_convergence/Thesis/hpc/hpc_run.py \
    $PROBLEM_IDX \
    $ALGORITHM_IDX \
    $CORES \
    $NFE \
    $SLURM_ARRAY_TASK_ID