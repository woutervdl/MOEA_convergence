from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import *
import os
import random
import numpy as np
import h5py
import tempfile
import shutil

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

    # Define FINAL result directory on shared FS
    final_result_dir = os.path.join(SCRATCH_BASE,"results", problem_name, f"{cores}cores", algorithm, f"seed{seed}")
    os.makedirs(final_result_dir, exist_ok=True)
    
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
        final_result_dir,
        problem_name,
        algorithm,
        cores,
        seed
    )

def save_single_result(results, convergence, metrics, runtime, final_path,
                       problem_name, algorithm, cores, seed):
    """Optimized result saving with HDF5 (Write Locally First)"""

    h5_filename = f"results_{problem_name}_{algorithm}_{cores}cores_seed{seed}.h5"
    final_h5_filepath = os.path.join(final_path, h5_filename)

    # Create a temporary directory on the node-local filesystem (e.g., /tmp)
    try:
        with tempfile.TemporaryDirectory(prefix="h5save_", dir="/tmp") as temp_dir:
            temp_h5_filepath = os.path.join(temp_dir, h5_filename)

            # Write the HDF5 file in the node-local temporary directory
            try:
                with h5py.File(temp_h5_filepath, "w") as hf:
                    # Store results
                    results_group = hf.create_group("results")
                    for col in results.columns:
                         # Ensure data is numpy array for h5py
                         data = results[col].values if hasattr(results[col], 'values') else results[col]
                         results_group.create_dataset(col, data=np.asarray(data))

                    # Store convergence
                    conv_group = hf.create_group("convergence")
                    for col in convergence.columns:
                         data = convergence[col].values if hasattr(convergence[col], 'values') else convergence[col]
                         conv_group.create_dataset(col, data=np.asarray(data))

                    # Store metrics
                    metrics_group = hf.create_group("metrics")
                    for col in metrics.columns:
                         data = metrics[col].values if hasattr(metrics[col], 'values') else metrics[col]
                         metrics_group.create_dataset(col, data=np.asarray(data))

                    # Store attributes
                    hf.attrs["runtime"] = runtime
                    hf.attrs["problem"] = problem_name
                    hf.attrs["algorithm"] = algorithm
                    hf.attrs["cores"] = cores
                    hf.attrs["seed"] = seed

            except Exception as write_err:
                 raise # Re-raise the error if writing failed

            # If writing to temp file succeeded, move it to the final destination
            try:
                shutil.move(temp_h5_filepath, final_h5_filepath)
            except Exception as move_err:
                raise # Re-raise the error if moving failed

    except Exception as temp_dir_err:
        raise # Re-raise the error

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