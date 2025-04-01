import sys
from Thesis.hpc.run_single_experiment import *
import random
import numpy as np

if __name__ == "__main__":
    # Parse command line arguments
    problem_idx = int(sys.argv[1])  # 0=DTLZ2, 1=DTLZ3, 2=JUSTICE
    algorithm_idx = int(sys.argv[2])  # 0=eps_nsgaii, 1=sse_nsgaii, 2=generational borg, 3=borg
    cores = int(sys.argv[3])
    
    # Generate a random seed based on SLURM_ARRAY_TASK_ID
    slurm_task_id = int(sys.argv[4])  # SLURM_ARRAY_TASK_ID passed as argument
    nfe = int(sys.argv[5])
    
    problems = ['DTLZ2', 'DTLZ3', 'JUSTICE']
    algorithms = ['eps_nsgaii', 'sse_nsgaii', 'generational_borg', 'borg']
    
    # Generate a reproducible random seed based on task ID and problem/algorithm indices
    seed = 12345 + slurm_task_id + problem_idx * 100 + algorithm_idx * 10
    print(f"Using seed: {seed}")
    
    # Run single configuration
    run_single_experiment(
        problem_name=problems[problem_idx],
        algorithm=algorithms[algorithm_idx],
        nfe=nfe,
        cores=cores,
        seed=seed
    )