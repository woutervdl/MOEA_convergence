from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import *
import os
import random
import numpy as np
import h5py

# def run_single_experiment(problem_name, algorithm, cores, nfe, seed):
#     """Run a single experiment configuration for HPC"""
#     ema_logging.log_to_stderr(ema_logging.INFO)

#     # Set random seeds for reproducibility
#     random.seed(seed)
#     np.random.seed(seed)
    
#     # Create results directory
#     result_dir = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
#     os.makedirs(result_dir, exist_ok=True)

#     # Get model
#     model = {
#         'DTLZ2': get_dtlz2_problem(4),
#         'DTLZ3': get_dtlz3_problem(4),
#         'JUSTICE': get_justice_model()
#     }[problem_name]

#     # Run optimization
#     results, convergences, metrics, runtimes = run_optimisation_experiment(
#         model=model,
#         algorithms=[algorithm],
#         nfe=nfe,
#         seeds=[seed],
#         core_count=cores
#     )

#     # Save results
#     save_single_result(
#         results=results,
#         convergences=convergences,
#         metrics=metrics,
#         runtimes=runtimes,
#         problem_name=problem_name,
#         algorithm=algorithm,
#         cores=cores,
#         seed=seed
#     )

# def save_single_result(results, convergences, metrics, runtimes, problem_name, algorithm, cores, seed):
#     """Save results for a single experiment"""
#     # Save result
#     result_path = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
#     results[0].to_csv(os.path.join(result_path, f"seed{seed}_results.csv"), index=False)
    
#     # Save convergence
#     convergences[0].to_csv(os.path.join(result_path, f"seed{seed}_convergence.csv"), index=False)
    
#     # Save metrics
#     metrics[algorithm][0].to_csv(os.path.join(result_path, f"seed{seed}_metrics.csv"), index=False)
    
#     # Save runtime
#     runtime_df = pd.DataFrame([{
#         'problem': problem_name,
#         'algorithm': algorithm,
#         'seed': seed,
#         'cores': cores,
#         'runtime': runtimes[0]
#     }])
#     runtime_df.to_csv(os.path.join(result_path, f"runtime_{cores}cores_{seed}.csv"), index=False)

# if __name__ == "__main__":
#     # Original run_experiments function can be removed or kept for local testing
#     pass

SCRATCH_BASE = "/scratch/wmvanderlinden/MOEA_convergence"

# Configure once at module level
ema_logging.log_to_stderr(ema_logging.INFO)
MODEL_MAP = {
    'DTLZ2': (get_dtlz2_problem, 4),
    'DTLZ3': (get_dtlz3_problem, 4),
    'JUSTICE': (get_justice_model, None)
}

def run_single_experiment(problem_name, algorithm, cores, nfe, seed):
    """Optimized single experiment runner for HPC"""
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create result directory structure
    result_dir = os.path.join(SCRATCH_BASE,"results", problem_name, f"{cores}cores", algorithm, f"seed{seed}")
    #result_dir = os.path.join("./results", problem_name, f"{cores}cores", algorithm, f"seed{seed}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Model loading with pre-configured parameters
    model_func, n_obj = MODEL_MAP[problem_name]
    model = model_func(n_obj) if n_obj else model_func()
    
    # Run optimization with memory monitoring
    results, convergences, metrics, runtimes = run_optimisation_experiment(
        model=model,
        algorithms=[algorithm],
        nfe=nfe,
        seeds=[seed],
        core_count=cores
    )
    
    # Save results
    save_single_result(
        results[0], 
        convergences[0], 
        metrics[algorithm][0], 
        runtimes[0],
        result_dir,
        problem_name,
        algorithm,
        cores,
        seed
    )

def save_single_result(results, convergence, metrics, runtime, path, 
                               problem_name, algorithm, cores, seed):
    """Optimized result saving with HDF5"""
    with h5py.File(os.path.join(path, f"results_{problem_name}_{algorithm}_{cores}cores_seed{seed}.h5"), "w") as hf:
        # Store results
        results_group = hf.create_group("results")
        for col in results.columns:
            results_group.create_dataset(col, data=results[col].values)
        
        # Store convergence
        conv_group = hf.create_group("convergence")
        for col in convergence.columns:
            conv_group.create_dataset(col, data=convergence[col].values)
        
        # Store metrics
        metrics_group = hf.create_group("metrics")
        for col in metrics.columns:
            metrics_group.create_dataset(col, data=metrics[col].values)
        
        # Store runtime
        hf.attrs["runtime"] = runtime
        hf.attrs["problem"] = problem_name
        hf.attrs["algorithm"] = algorithm
        hf.attrs["cores"] = cores
        hf.attrs["seed"] = seed

if __name__ == "__main__":
    #run_single_experiment('DTLZ2', 'borg', 4, 10000, 42)
    #run_single_experiment('DTLZ2', 'borg', 4, 10000, 17)
    #run_single_experiment('DTLZ2', 'eps_nsgaii', 4, 10000, 42)
    #run_single_experiment('DTLZ2', 'eps_nsgaii', 4, 10000, 17)
    #run_single_experiment('DTLZ2', 'borg', 8, 10000, 42)
    #run_single_experiment('DTLZ2', 'borg', 8, 10000, 17)
    #run_single_experiment('DTLZ2', 'eps_nsgaii', 8, 10000, 42)
    #run_single_experiment('DTLZ2', 'eps_nsgaii', 8, 10000, 17)
    run_single_experiment('JUSTICE', 'eps_nsgaii', 6, 200, 40) # test
    #pass