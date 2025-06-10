from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import *
import os
import random
import numpy as np
import h5py
import tempfile

# Set base directory to scratch for HPC runs
SCRATCH_BASE = "/scratch/wmvanderlinden/MOEA_convergence"

# Configure problem settings once at module level
ema_logging.log_to_stderr(ema_logging.INFO)
MODEL_MAP = {
    'DTLZ2': (get_dtlz2_problem, 4),
    'DTLZ3': (get_dtlz3_problem, 4),
    'JUSTICE': (get_justice_model, None)
}

def run_single_experiment(problem_name, algorithm, cores, nfe, seed):
    np.random.seed(seed); random.seed(seed) # Setting rng for reproducibility

    # Path for the HDF5 file
    final_result_dir = os.path.join(SCRATCH_BASE, "hdf5_results", problem_name, f"{cores}cores", algorithm, f"seed{seed}")
    os.makedirs(final_result_dir, exist_ok=True)

    # Selecting the model based on the problem name
    model_func, n_obj = MODEL_MAP[problem_name]
    model = model_func(n_obj) if n_obj else model_func()

    # Run the optimisation experiment
    results_list, convergences_list, runtimes_list = run_optimisation_experiment(
        model=model,
        algorithms=[algorithm],
        nfe=nfe,
        seeds=[seed],   
        core_count=cores,
        problem_name_for_path=problem_name 
    )

    # Save the results to HDF5
    save_single_result(
        results_list[0],      
        convergences_list[0],  
        runtimes_list[0],   
        final_result_dir,
        problem_name, 
        algorithm, 
        cores, 
        seed
    )
    print(f"HPC run for {problem_name}, {algorithm}, {cores} cores, seed {seed} finished and saved")

def save_single_result(
        final_archive_df, 
        epsilon_progress_df, 
        runtime_val, 
        final_path,
        problem_name, 
        algorithm, 
        cores, 
        seed_val):
    # Set up the final path and filename
    h5_filename = f"final_state_{problem_name}_{algorithm}_{cores}cores_seed{seed_val}.h5"
    final_h5_filepath = os.path.join(final_path, h5_filename)

    try:
        with tempfile.TemporaryDirectory(prefix="h5save_", dir="/tmp") as temp_dir: # Use node-local /tmp
            temp_h5_filepath = os.path.join(temp_dir, h5_filename)
            try:
                with h5py.File(temp_h5_filepath, "w") as hf:
                    # Store final_archive_df
                    final_archive_group = hf.create_group("final_archive")
                    for col in final_archive_df.columns:
                        data = final_archive_df[col].values if hasattr(final_archive_df[col], 'values') else final_archive_df[col]
                        final_archive_group.create_dataset(col, data=np.asarray(data))
                    
                    # Store epsilon_progress_df
                    if epsilon_progress_df is not None and not epsilon_progress_df.empty:
                        ep_group = hf.create_group("epsilon_progress")
                        for col in epsilon_progress_df.columns:
                            data = epsilon_progress_df[col].values if hasattr(epsilon_progress_df[col], 'values') else epsilon_progress_df[col]
                            ep_group.create_dataset(col, data=np.asarray(data))

                    # Store attributes
                    hf.attrs["runtime"] = runtime_val
                    hf.attrs["problem"] = problem_name
                    hf.attrs["algorithm"] = algorithm
                    hf.attrs["cores"] = cores
                    hf.attrs["seed"] = seed_val
            except Exception as write_err: print(f"HDF5 Write Error: {write_err}"); raise
            shutil.move(temp_h5_filepath, final_h5_filepath) # Move the temp HDF5 file to the final location
            print(f"Final state HDF5 saved to {final_h5_filepath}")
    except Exception as temp_err: print(f"Temp Dir/Move Error: {temp_err}"); raise