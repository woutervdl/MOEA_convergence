import sys
from Thesis.hpc.run_single_experiment import *

if __name__ == "__main__":
    problem_idx = int(sys.argv[1])
    algorithm_idx = int(sys.argv[2])
    cores = int(sys.argv[3])
    nfe = int(sys.argv[4])
    seed = int(sys.argv[5]) 

    problems = ['DTLZ2', 'DTLZ3', 'JUSTICE']
    algorithms = ['eps_nsgaii', 'borg', 'generational_borg']
    
    print(f"Running {problems[problem_idx]} with {algorithms[algorithm_idx]} on {cores} cores (seed {seed})")
    
    run_single_experiment(
        problem_name=problems[problem_idx],
        algorithm=algorithms[algorithm_idx],
        cores=cores,
        nfe=nfe,
        seed=seed
    )