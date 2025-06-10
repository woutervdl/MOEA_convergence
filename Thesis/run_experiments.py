from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import run_optimisation_experiment
import os
import numpy as np 
import h5py    
import tempfile  
import shutil   
import pandas as pd

# Set up local directories for saving results
LOCAL_ARCHIVES_OUTPUT_DIR = "./hdf5_results" 
os.makedirs(LOCAL_ARCHIVES_OUTPUT_DIR, exist_ok=True)

def run_experiments():
    """
    Runs optimisation experiments for multiple problems and algorithms, saving results to HDF5 files.
    Purpose is to quickly locally test the optimisation code.
    """
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Defining the problems, seeds, MOEAs and core counts
    problems = [
        ('DTLZ2', get_dtlz2_problem(4)),
        ('DTLZ3', get_dtlz3_problem(4)), 
        #('JUSTICE', get_justice_model())  
    ]
    algorithms = ['eps_nsgaii', 'generational_borg', 'borg']
    core_count_list = [6] 
    nfe = 70000
    # num_seeds = 5 
    # random.seed(12345)
    # seed_values = random.sample(range(10000, 1000000), num_seeds)
    seed_values = [12345, 23403, 39349, 60930, 93489] 
    print(f"Using seed values: {seed_values}")

    # Loop through each problem, core count, MOEA and seed and run the optimisation experiments
    for problem_name, model in problems:
        print(f"Running experiments for {problem_name}")
        problem_dir = os.path.join(LOCAL_ARCHIVES_OUTPUT_DIR, problem_name)
        os.makedirs(problem_dir, exist_ok=True)

        for cores in core_count_list: 
            print(f"Running experiments for {problem_name} with {cores} cores")
            core_dir = os.path.join(problem_dir, f"{cores}_cores") 
            os.makedirs(core_dir, exist_ok=True)

            final_archives_list, epsilon_progress_list, runtimes_list = run_optimisation_experiment(
                model, algorithms, nfe, seed_values, cores, problem_name
            )
            
            result_index = 0 # To index into the returned lists
            for algorithm_name_iter in algorithms: 
                algorithm_dir = os.path.join(core_dir, algorithm_name_iter)
                os.makedirs(algorithm_dir, exist_ok=True) 

                for seed_value_iter in seed_values:
                    if result_index < len(final_archives_list):
                        current_final_archive = final_archives_list[result_index]
                        current_epsilon_progress = epsilon_progress_list[result_index]
                        current_runtime = runtimes_list[result_index]

                        save_results(
                            current_final_archive,
                            current_epsilon_progress,
                            current_runtime,
                            algorithm_dir, 
                            problem_name,
                            algorithm_name_iter,
                            cores,
                            seed_value_iter
                        )
                    else:
                        print(f"Warning: Missing results for {algorithm_name_iter}, seed {seed_value_iter}")
                    result_index += 1
        
        print(f"Completed experiments for {problem_name}")

def save_results(
        final_archive_df,      
        epsilon_progress_df,    
        runtime_val,         
        final_path,        
        problem_name, 
        algorithm_name_str,    
        cores_val,            
        seed_val_int):
    """
    Saves the final archive and epsilon progress DataFrames to an HDF5 file.
    """          

    h5_filename = f"final_state_{problem_name}_{algorithm_name_str}_{cores_val}cores_seed{seed_val_int}.h5"
    final_h5_filepath = os.path.join(final_path, h5_filename)

    try:
        with tempfile.TemporaryDirectory(prefix="h5save_", dir="/tmp") as temp_dir:
            temp_h5_filepath = os.path.join(temp_dir, h5_filename)
            try:
                with h5py.File(temp_h5_filepath, "w") as hf:
                    # Store final_archive_df
                    final_archive_group = hf.create_group("final_archive")
                    # Check if it's a DataFrame and not empty
                    if isinstance(final_archive_df, pd.DataFrame) and not final_archive_df.empty:
                        for col in final_archive_df.columns:
                            data = final_archive_df[col].values if hasattr(final_archive_df[col], 'values') else final_archive_df[col]
                            final_archive_group.create_dataset(col, data=np.asarray(data))
                    elif final_archive_df is not None: # It exists but isn't a usable DataFrame
                        print(f"Warning: final_archive_df for {h5_filename} is not a valid DataFrame or is empty.")
                    
                    # Store epsilon_progress_df
                    if epsilon_progress_df is not None and isinstance(epsilon_progress_df, pd.DataFrame) and not epsilon_progress_df.empty:
                        ep_group = hf.create_group("epsilon_progress")
                        for col in epsilon_progress_df.columns:
                            data = epsilon_progress_df[col].values if hasattr(epsilon_progress_df[col], 'values') else epsilon_progress_df[col]
                            ep_group.create_dataset(col, data=np.asarray(data))
                    elif epsilon_progress_df is not None:
                         print(f"Warning: epsilon_progress_df for {h5_filename} is not a valid DataFrame or is empty.")


                    # Store attributes
                    hf.attrs["runtime"] = runtime_val if runtime_val is not None else -1.0 # Handle None runtime
                    hf.attrs["problem"] = problem_name
                    hf.attrs["algorithm"] = algorithm_name_str
                    hf.attrs["cores"] = cores_val
                    hf.attrs["seed"] = seed_val_int
            except Exception as write_err: 
                print(f"HDF5 Write Error for {final_h5_filepath}: {write_err}")
                raise
            shutil.move(temp_h5_filepath, final_h5_filepath)
            print(f"  Final state HDF5 saved to: {final_h5_filepath}")
    except Exception as temp_err: 
        print(f"Temp Dir/Move Error for {final_h5_filepath}: {temp_err}")
        raise


if __name__ == "__main__":
    run_experiments()
