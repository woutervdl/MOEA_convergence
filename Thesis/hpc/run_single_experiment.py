from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import *
import os
import random
import pandas as pd

def run_single_experiment(problem_name, algorithm, cores, nfe, seed):
    """Run a single experiment configuration for HPC"""
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    # Create results directory
    result_dir = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
    os.makedirs(result_dir, exist_ok=True)

    # Get model
    model = {
        'DTLZ2': get_dtlz2_problem(4),
        'DTLZ3': get_dtlz3_problem(4),
        'JUSTICE': get_justice_model()
    }[problem_name]

    # Run optimization
    results, convergences, metrics, runtimes = run_optimisation_experiment(
        model=model,
        algorithms=[algorithm],
        nfe=nfe,
        seeds=[seed],
        core_count=cores
    )

    # Save results
    save_single_result(
        results=results,
        convergences=convergences,
        metrics=metrics,
        runtimes=runtimes,
        problem_name=problem_name,
        algorithm=algorithm,
        cores=cores,
        seed=seed
    )

def save_single_result(results, convergences, metrics, runtimes, problem_name, algorithm, cores, seed):
    """Save results for a single experiment"""
    # Save result
    result_path = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
    results[0].to_csv(os.path.join(result_path, f"seed{seed}_results.csv"), index=False)
    
    # Save convergence
    convergences[0].to_csv(os.path.join(result_path, f"seed{seed}_convergence.csv"), index=False)
    
    # Save metrics
    metrics[algorithm][0].to_csv(os.path.join(result_path, f"seed{seed}_metrics.csv"), index=False)
    
    # Save runtime
    runtime_df = pd.DataFrame([{
        'problem': problem_name,
        'algorithm': algorithm,
        'seed': seed,
        'cores': cores,
        'runtime': runtimes[0]
    }])
    runtime_df.to_csv(os.path.join(result_path, f"runtime_{cores}cores_{seed}.csv"), index=False)

if __name__ == "__main__":
    # Original run_experiments function can be removed or kept for local testing
    pass
