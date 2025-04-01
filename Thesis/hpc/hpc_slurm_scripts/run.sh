#!/bin/bash
#SBATCH --job-name="moea"
#SBATCH --time=04:00:00
#SBATCH --partition=compute-p1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-239 #(3 problems × 4 algorithms × 20 seeds)

# This script should be submitted with sbatch --ntasks=X run.sh X
CORES=$1
NFE=10000  # Set desired NFE value

# Validate core count
if [[ ! "4 6 8" =~ (^|[[:space:]])$CORES($|[[:space:]]) ]]; then
    echo "Invalid core count: $CORES. Use 4, 6, or 8."
    exit 1
fi

# Define arrays
PROBLEMS=("DTLZ2" "DTLZ3" "JUSTICE")
ALGORITHMS=("eps_nsgaii" "sse_nsgaii" "generational_borg" "borg")

# Calculate indices based on SLURM_ARRAY_TASK_ID (e.g., array size = problems × algorithms × seeds)
PROBLEM_IDX=$(( $SLURM_ARRAY_TASK_ID / 30 ))     # Total combinations per core count divided by number of seeds per config (30 = algorithms × seeds)
ALGORITHM_IDX=$(( ($SLURM_ARRAY_TASK_ID % 30) / 10 ))   # Modulo by number of seeds per config (10), then divide by number of seeds per algorithm (10)
SEED_IDX=$(( $SLURM_ARRAY_TASK_ID % 10 ))       # Modulo by number of seeds per config (10)

# Generate unique random seed based on SLURM_ARRAY_TASK_ID and indices (reproducible across runs)
SEED=$((10000 + SEED_IDX * 17))

# Echo start information
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running problem ${PROBLEMS[$PROBLEM]} with ${ALGORITHMS[$ALGORITHM]} on ${CORES} cores (seed $SEED)"

# Run with specified cores
python hpc_run.py $PROBLEM $ALGORITHM $CORES $SEED $NFE