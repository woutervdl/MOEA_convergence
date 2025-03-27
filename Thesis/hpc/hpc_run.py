import sys
import run_experiments
import random
import numpy as np

if __name__ == "__main__":
    # Parse command line arguments
    problem_idx = int(sys.argv[1])  # 0=DTLZ2, 1=DTLZ3, 2=JUSTICE
    algorithm_idx = int(sys.argv[2])  # 0=eps_nsgaii, 1=sse_nsgaii, 2=generational borg, 3=borg
    cores = int(sys.argv[3])
    seed = int(sys.argv[4])
    
    problems = ['DTLZ2', 'DTLZ3', 'JUSTICE']
    algorithms = ['eps_nsgaii', 'sse_nsgaii', 'generational_borg', 'borg']
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Run single configuration
    run_experiments.run_single_experiment(
        problem_name=problems[problem_idx],
        algorithm=algorithms[algorithm_idx],
        cores=cores,
        seed=seed
    )
