#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=compute-p1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-8 #(3 problems × 3 algorithms × 1 seeds)

# This script should be submitted with sbatch --job-name="X" --ntasks=Y run.sh Y Z (where Y is the core count and Z is the seed number)
CORES=$1
SEED=$2
NFE=50000

N_PROBLEMS=3
N_ALGORITHMS=3
N_SEEDS=1

ALGOS_PER_PROBLEM=$N_ALGORITHMS
SEEDS_PER_ALGO=$N_SEEDS
COMBOS_PER_PROBLEM=$(( $N_ALGORITHMS * $N_SEEDS ))

# Calculate indices from SLURM_ARRAY_TASK_ID
PROBLEM_IDX=$(( $SLURM_ARRAY_TASK_ID / $COMBOS_PER_PROBLEM )) 
TEMP_IDX_IN_PROBLEM=$(( $SLURM_ARRAY_TASK_ID % $COMBOS_PER_PROBLEM )) 
ALGORITHM_IDX=$(( $TEMP_IDX_IN_PROBLEM / $SEEDS_PER_ALGO ))  

echo "Task ID: $SLURM_ARRAY_TASK_ID => Problem: $PROBLEM_IDX, Algo: $ALGORITHM_IDX, Seed: $SEED, Cores: $CORES"

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
    $SEED 