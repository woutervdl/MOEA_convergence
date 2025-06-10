import pandas as pd
import numpy as np
import os
import gc
from concurrent.futures import ProcessPoolExecutor 
import h5py 
from ema_workbench import ema_logging
from ema_workbench.em_framework.optimization import (
    GenerationalDistanceMetric, EpsilonIndicatorMetric, SpacingMetric, HypervolumeMetric, ArchiveLogger, to_problem
)
from Thesis.util.spread import Spread
from ema_workbench.em_framework.outcomes import ScalarOutcome
from JUSTICE_fork.solvers.convergence.hypervolume import calculate_hypervolume_from_archives
from Thesis.util.model_definitions import get_justice_model, get_dtlz2_problem, get_dtlz3_problem
from numba import jit, prange

# Define base directories and paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ARCHIVES_BASE = os.path.join(BASE_DIR, "archives")
HDF5_BASE = os.path.join(BASE_DIR, "hdf5_results")
GLOBAL_JUSTICE_REF_SET_CSV_PATH = os.path.join(BASE_DIR, "global_JUSTICE_ref_set.csv")
CORE_COUNTS_TO_ANALYZE = [16, 32, 48]  

# Define the problems and their configurations
PROBLEMS_CONFIG = {
    "DTLZ2": {"model_func": get_dtlz2_problem, "n_obj": 4, "epsilons": [0.05] * 4},
    "DTLZ3": {"model_func": get_dtlz3_problem, "n_obj": 4, "epsilons": [0.05] * 4},
    "JUSTICE": {"model_func": get_justice_model, "n_obj": None, "epsilons": [0.01, 0.25, 10, 10]},
}

# Define the algorithms and seeds to analyse
ALGORITHMS_TO_ANALYZE = ['eps_nsgaii', 'borg', 'generational_borg'] 
SEEDS_TO_ANALYZE = [12345, 23403, 39349, 60930, 93489]

# Cache for models and reference sets
CACHED_MODELS_AND_REFS_STORE = {}

# Define the objective directions for JUSTICE to ensure correct metric calculations
objective_directions = [True, True, False, False]  # True = minimise, False = maximise

def transform_objectives(dataframe_input: pd.DataFrame, directions: list) -> pd.DataFrame:
    """
    Transforms the objective columns of a DataFrame according to the specified directions.
    Maximization objectives are negated to convert them to minimization problems.
    Levers (non-objective columns) are preserved.

    Args:
        dataframe_input (pd.DataFrame): The input DataFrame. Assumes objectives
                                         are the last 'num_objectives' columns.
        directions (list): A list of booleans indicating optimization direction
                           for each objective. True for minimize, False for maximize.

    Returns:
        pd.DataFrame: A new DataFrame with transformed objective values and
                      preserved lever values, column names, and index.
    """
    num_objectives = len(directions)

    if not isinstance(dataframe_input, pd.DataFrame):
        raise TypeError(
            "Input 'dataframe_input' to transform_objectives must be a Pandas DataFrame."
        )

    # Work on a copy of the DataFrame's values to avoid modifying the original data accidentally.
    data_values_matrix = dataframe_input.values.copy()

    # Iterate through each row of the NumPy array
    for i in range(data_values_matrix.shape[0]):
        # Iterate through each objective. Objectives are the last 'num_objectives' columns.
        for j in range(num_objectives):
            # Calculate the column index for the current objective
            # dataframe_input.shape[1] is the total number of columns in the original DataFrame
            obj_col_idx_in_numpy_array = dataframe_input.shape[1] - num_objectives + j
            
            value = data_values_matrix[i, obj_col_idx_in_numpy_array]
            is_minimize_direction = directions[j]  # True if minimise, False if maximise
            
            # Ensure the value is numeric before transformation
            numeric_value = float(value)
            
            if not is_minimize_direction:  # If False (objective is to be maximised)
                # Negate to convert to a minimisation problem
                data_values_matrix[i, obj_col_idx_in_numpy_array] = -numeric_value
            else:  # If True (objective is to be maximised)
                # Ensure it's a float, but value remains the same for minimisation
                data_values_matrix[i, obj_col_idx_in_numpy_array] = numeric_value
        
    # Create a new DataFrame from the modified NumPy array,
    # using the original DataFrame's column names and index.
    transformed_df = pd.DataFrame(
        data_values_matrix,
        columns=dataframe_input.columns,
        index=dataframe_input.index.copy() # Preserve the original index
    )
    
    return transformed_df

# Numba JIT-compiled function for efficiency calculation (originally written for parallel HPC execution, used here since it was written anyway)
@jit(nopython=True, parallel=True)
def efficiency_kernel(hv_values, nfe_values): 
    efficiencies = np.zeros_like(hv_values)
    if len(hv_values) < 2: return efficiencies
    for i in prange(1, len(hv_values)):
        if nfe_values[i] > nfe_values[i-1]:
            delta_hv = hv_values[i] - hv_values[i-1]
            delta_nfe = nfe_values[i] - nfe_values[i-1]
            efficiencies[i] = delta_hv / delta_nfe if delta_nfe > 0 else 0.0
        else: efficiencies[i] = 0.0
    return efficiencies

def preload_models_and_reference_sets():
    """
    Loads/generates models, problem objects, and FULL reference sets (levers + objectives)
    for each problem once in the main process.
    """
    print("Preloading models and full reference sets...")
    global CACHED_MODELS_AND_REFS_STORE
    for prob_name, model_config in PROBLEMS_CONFIG.items():
        if prob_name in CACHED_MODELS_AND_REFS_STORE:
            continue

        print(f"  Processing {prob_name} for preloading...")
        # Create the model instance and problem object
        model_inst = model_config["model_func"](model_config["n_obj"]) if model_config["n_obj"] else model_config["model_func"]()
        problem_inst = to_problem(model_inst, searchover="levers")
        
        ref_df_full = pd.DataFrame()
        if prob_name == 'JUSTICE':
            # Load the global JUSTICE reference set from CSV
            if os.path.exists(GLOBAL_JUSTICE_REF_SET_CSV_PATH):
                ref_df_full = pd.read_csv(GLOBAL_JUSTICE_REF_SET_CSV_PATH)
                print(f"    Loaded JUSTICE global reference set (full). Shape: {ref_df_full.shape}")
            else:
                print(f"    ERROR: Global JUSTICE reference file not found: {GLOBAL_JUSTICE_REF_SET_CSV_PATH}")
        elif prob_name in ['DTLZ2', 'DTLZ3']:
            # Generate the reference set for DTLZ problems
            if hasattr(model_inst, 'generate_true_pareto_solutions'):
                ref_df_full = model_inst.generate_true_pareto_solutions(n_points=1000)
                if ref_df_full is not None and not ref_df_full.empty:
                    print(f"    Generated {prob_name} true reference set (full). Shape: {ref_df_full.shape}")
                else:
                    print(f"    ERROR: generate_true_pareto_solutions for {prob_name} returned empty.")
            else:
                print(f"    WARNING: Model for {prob_name} does not have 'generate_true_pareto_solutions' method.")
        
        if not ref_df_full.empty:
            # Ensure the reference set has the expected columns (levers + outcomes)
            expected_cols = [l.name for l in model_inst.levers] + [o.name for o in model_inst.outcomes]
            missing_cols = [col for col in expected_cols if col not in ref_df_full.columns]
            if missing_cols:
                print(f"    ERROR: Reference set for {prob_name} is missing columns: {missing_cols}. Expected: {expected_cols}. Available: {list(ref_df_full.columns)}")
                ref_df_full = pd.DataFrame()
            else:
                 print(f"    Reference set for {prob_name} validated against problem definition.")

        # Store the model, problem, and reference set in the cache
        CACHED_MODELS_AND_REFS_STORE[prob_name] = {
            'model': model_inst,
            'problem': problem_inst,
            'ref_set_full': ref_df_full 
        }

def calculate_metrics_for_single_run(problem_name_arg, algorithm_arg, cores_arg, seed_arg,
                                     problem_info_for_worker): 
    """
    Calculates various performance metrics for a single optimisation run,
    saves results to HDF5, and returns None.
    """    

    model_obj = problem_info_for_worker['model']
    problem_obj = problem_info_for_worker['problem']
    full_ref_set_for_run = problem_info_for_worker['ref_set_full'] 

    print(f"  Processing: {problem_name_arg}, Algo: {algorithm_arg}, Cores: {cores_arg}, Seed: {seed_arg}")

    # Define paths for archives and HDF5 files
    archive_tar_gz_dir = os.path.join(ARCHIVES_BASE, problem_name_arg, f"{cores_arg}cores", algorithm_arg, f"seed{seed_arg}")
    archive_tar_gz_file = os.path.join(archive_tar_gz_dir, "archive.tar.gz")
    hdf5_dir = os.path.join(HDF5_BASE, problem_name_arg, f"{cores_arg}cores", algorithm_arg, f"seed{seed_arg}")
    hdf5_file_name = f"final_state_{problem_name_arg}_{algorithm_arg}_{cores_arg}cores_seed{seed_arg}.h5"
    hdf5_file_path = os.path.join(hdf5_dir, hdf5_file_name)

    # Ensure the archive directory exists
    if not os.path.exists(archive_tar_gz_file):
        print(f"    SKIP: Archive .tar.gz file not found: {archive_tar_gz_file}")
        return pd.DataFrame()
    try: 
        # Load the archives from the tar.gz file
        archives_history = ArchiveLogger.load_archives(archive_tar_gz_file)
    except Exception as e: 
        print(f"    ERROR loading {archive_tar_gz_file}: {e}")
        return pd.DataFrame()
    if not archives_history: 
        print(f"    SKIP: No archives loaded (empty history) from {archive_tar_gz_file}")
        return pd.DataFrame()
    archive_items = list(archives_history.items())

    epsilon_progress_df_for_run, total_runtime_for_run = pd.DataFrame(), np.nan
    if os.path.exists(hdf5_file_path): 
        try:
            with h5py.File(hdf5_file_path, "r") as hf:
                # Load epsilon progress and runtime if available
                if "epsilon_progress" in hf:
                    ep_group = hf["epsilon_progress"]
                    epsilon_progress_df_for_run = pd.DataFrame({key: ep_group[key][()] for key in ep_group.keys()})
                    if 'nfe' in epsilon_progress_df_for_run.columns:
                        epsilon_progress_df_for_run['nfe'] = epsilon_progress_df_for_run['nfe'].astype(int)
                if "runtime" in hf.attrs: total_runtime_for_run = hf.attrs["runtime"]
        except Exception as e: 
            print(f"    WARNING: Error loading HDF5 {hdf5_file_path}: {e}")
    else: 
        print(f"    WARNING: HDF5 file not found: {hdf5_file_path}. EpsilonProgress/runtime missing.")

    # Initialise metrics
    gd_metric, ei_metric, sp_metric, sm_metric, std_hv_metric = None, None, None, None, None
    
    # Check if the reference set is available and not empty
    if full_ref_set_for_run is None or full_ref_set_for_run.empty:
        print(f"    WARNING (Worker): FULL Reference set for {problem_name_arg} is empty. Ref-based metrics will be NaN.")
    elif problem_name_arg == 'JUSTICE':
        # Apply transformation to the reference set only for JUSTICE (not needed for DTLZ)
        transformed_reference_set = transform_objectives(full_ref_set_for_run, objective_directions)
        gd_metric = GenerationalDistanceMetric(transformed_reference_set, problem_obj)
        ei_metric = EpsilonIndicatorMetric(transformed_reference_set, problem_obj)
    else:
        gd_metric = GenerationalDistanceMetric(full_ref_set_for_run, problem_obj)
        ei_metric = EpsilonIndicatorMetric(full_ref_set_for_run, problem_obj)
        std_hv_metric = HypervolumeMetric(full_ref_set_for_run, problem_obj)

    # Initialise other metrics (not dependent on reference set)
    sm_metric = SpacingMetric(problem_obj)
    sp_metric = Spread(problem_obj) 

    historical_metrics = []
    outcome_names_list = [o.name for o in model_obj.outcomes]
    direction_optimization_list = ["max" if o.kind == ScalarOutcome.MAXIMIZE else "min" for o in model_obj.outcomes]
    custom_hv_hist = {}

    # Calculate custom hypervolume for JUSTICE
    if problem_name_arg == 'JUSTICE':
        if not (full_ref_set_for_run is None or full_ref_set_for_run.empty) and os.path.exists(GLOBAL_JUSTICE_REF_SET_CSV_PATH):
            try:
                custom_hv_df = calculate_hypervolume_from_archives(
                    outcome_names_list, direction_optimization_list, archive_tar_gz_dir, "archive.tar.gz",
                    "./temp_hv_outputs", False, True,
                    os.path.dirname(GLOBAL_JUSTICE_REF_SET_CSV_PATH), os.path.basename(GLOBAL_JUSTICE_REF_SET_CSV_PATH)
                ) 
                if not custom_hv_df.empty:
                    custom_hv_df['nfe'] = custom_hv_df['nfe'].astype(int)
                    for _, row in custom_hv_df.iterrows(): custom_hv_hist[row['nfe']] = row['hypervolume']
            except Exception as e: 
                print(f"    ERROR Custom HV JUSTICE ({archive_tar_gz_dir}): {e}")
        else: 
            print(f"    SKIP Custom HV JUSTICE: Ref objectives or global CSV file missing.")

    # Loop through the archives and calculate metrics
    for nfe_str, archive_df_snapshot_full in archive_items:
        transformed_archive_df_snapshot_full = transform_objectives(archive_df_snapshot_full, objective_directions)
        nfe_int, row = int(nfe_str), {"nfe": int(nfe_str), "archive_size": len(archive_df_snapshot_full)}
        
        # Use custom hypervolume and transformed objectives for JUSTICE
        if problem_name_arg == 'JUSTICE': 
            row["hypervolume"] = custom_hv_hist.get(nfe_int, np.nan)
            row["generational_distance"] = gd_metric.calculate(transformed_archive_df_snapshot_full.iloc[:,1:]) if gd_metric else np.nan #.iloc[:,1:] is used because the index column was also saved
            row["epsilon_indicator"] = ei_metric.calculate(transformed_archive_df_snapshot_full.iloc[:,1:]) if ei_metric else np.nan
            row["spacing"] = sm_metric.calculate(transformed_archive_df_snapshot_full.iloc[:,1:]) if sm_metric else np.nan
            row["spread"] = sp_metric.calculate(transformed_archive_df_snapshot_full.iloc[:,1:]) if sp_metric else np.nan
        # Use standard hypervolume and non-transformed objectives for DTLZ
        elif std_hv_metric: 
            try: 
                row["hypervolume"] = std_hv_metric.calculate(archive_df_snapshot_full.iloc[:,1:]) #.iloc[:,1:] is used because the index column was also saved
            except Exception: 
                row["hypervolume"] = np.nan
            row["generational_distance"] = gd_metric.calculate(archive_df_snapshot_full.iloc[:,1:]) if gd_metric else np.nan #.iloc[:,1:] is used because the index column was also saved
            row["epsilon_indicator"] = ei_metric.calculate(archive_df_snapshot_full.iloc[:,1:]) if ei_metric else np.nan
            row["spacing"] = sm_metric.calculate(archive_df_snapshot_full.iloc[:,1:]) if sm_metric else np.nan
            row["spread"] = sp_metric.calculate(archive_df_snapshot_full.iloc[:,1:]) if sp_metric else np.nan
        else: 
            row["hypervolume"] = np.nan

        historical_metrics.append(row)

    if not historical_metrics: 
        return pd.DataFrame()
    metrics_df = pd.DataFrame(historical_metrics).sort_values(by="nfe")
    # Merge with epsilon progress if available
    if not epsilon_progress_df_for_run.empty and 'nfe' in epsilon_progress_df_for_run.columns:
        metrics_df = pd.merge(metrics_df, epsilon_progress_df_for_run, on="nfe", how="left", suffixes=('', '_ep'))
    # Calculate time efficiency based on hypervolume and NFE
    if "hypervolume" in metrics_df.columns and metrics_df["hypervolume"].notna().any():
        hv, nfe_vals = metrics_df["hypervolume"].ffill().fillna(0).to_numpy(), metrics_df["nfe"].to_numpy()
        if len(hv) > 1: 
            metrics_df["time_efficiency"] = efficiency_kernel(hv, nfe_vals)
        else: 
            metrics_df["time_efficiency"] = 0.0
        metrics_df["time_efficiency"].fillna(0, inplace=True)
    else: 
        metrics_df["time_efficiency"] = np.nan
    metrics_df["total_runtime"] = total_runtime_for_run
    # Aggregate time efficiency based on the last hypervolume and total runtime
    if pd.notna(total_runtime_for_run) and total_runtime_for_run > 0 and "hypervolume" in metrics_df.columns and not metrics_df.empty:
        final_hv_series = metrics_df["hypervolume"].dropna()
        final_hv = final_hv_series.iloc[-1] if not final_hv_series.empty else 0.0
        metrics_df["time_efficiency_aggregated"] = final_hv / total_runtime_for_run #NUTTELOOS
    else: 
        metrics_df["time_efficiency_aggregated"] = np.nan
    metrics_df["problem"], metrics_df["algorithm"], metrics_df["cores"], metrics_df["seed"] = problem_name_arg, algorithm_arg, cores_arg, seed_arg
    gc.collect()

    # Create output directory for results
    output_dir = os.path.join(
        "../results",
        problem_name_arg,
        f"{cores_arg}cores",
        algorithm_arg,
        f"seed{seed_arg}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results_filename = f"results_{problem_name_arg}_{algorithm_arg}_{cores_arg}cores_seed{seed_arg}.h5"
    results_path = os.path.join(output_dir, results_filename)

    # Save the results to HDF5
    with h5py.File(results_path, 'w') as hf:
        # Store all metrics
        metrics_group = hf.create_group("metrics")
        for col in metrics_df.columns:
            if col in ['problem', 'algorithm', 'cores', 'seed']:
                continue  
            data = metrics_df[col].values
            if pd.api.types.is_integer_dtype(metrics_df[col]):
                metrics_group.create_dataset(col, data=data.astype(int))
            else:
                metrics_group.create_dataset(col, data=data.astype(float))
        
        # Store epsilon progress if exists
        if not epsilon_progress_df_for_run.empty:
            convergence_group = hf.create_group("convergence")
            for col in epsilon_progress_df_for_run.columns:
                convergence_group.create_dataset(col, data=epsilon_progress_df_for_run[col].values.astype(float))
        
        # Store runtime and metadata as attributes
        if pd.notna(total_runtime_for_run):
            hf.attrs['runtime'] = float(total_runtime_for_run)
        hf.attrs['problem'] = problem_name_arg
        hf.attrs['algorithm'] = algorithm_arg
        hf.attrs['cores'] = int(cores_arg)
        hf.attrs['seed'] = int(seed_arg)

    print(f"Saved HDF5 file: {results_path}")
    gc.collect()
    return None 

def worker_process_task(task_args_tuple):
    """
    Worker function to process a single task.
    """
    calculate_metrics_for_single_run(*task_args_tuple) 
    return None

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    # Preload models and reference sets once in the main process
    preload_models_and_reference_sets() 

    tasks = []
    # Prepare tasks for each problem, algorithm, core count, and seed
    for prob_name, _ in PROBLEMS_CONFIG.items():
        problem_info = CACHED_MODELS_AND_REFS_STORE.get(prob_name)
        if not problem_info or problem_info['ref_set_full'].empty:
            print(f"WARNING (Task Prep): Model/Problem or FULL Reference set for {prob_name} is empty or not preloaded. Skipping tasks for this problem.")
            continue

        for cores in CORE_COUNTS_TO_ANALYZE:
            for algo in ALGORITHMS_TO_ANALYZE:
                for seed in SEEDS_TO_ANALYZE:
                    tasks.append((prob_name, algo, cores, seed, problem_info))
    
    num_workers = 6
    print(f"\nStarting analysis for {len(tasks)} tasks using {num_workers} workers...")
    if not tasks: 
        print("No tasks to process. Check config and preloading.")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_process_task, tasks))