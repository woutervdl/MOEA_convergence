import pandas as pd
import numpy as np
import os
import gc
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
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

#### HPC
# ARCHIVES_BASE = "../hpc/full_archives_tar_gz" 
# HDF5_BASE = "../hpc/hdf5_results"
# CORE_COUNTS = [16, 32, 48]
####

#### Local
ARCHIVES_BASE = "../local_archives"
HDF5_BASE = "../hdf5_results"
CORE_COUNTS = [6]
#####

GLOBAL_JUSTICE_REF_SET_CSV_PATH = "../hpc/global_ref_set.csv"

# Experiment Matrix
PROBLEMS_CONFIG = {
    "JUSTICE": {"model_func": get_justice_model, "n_obj": None, "epsilons": [0.01, 0.25, 10, 10]},
    "DTLZ2": {"model_func": get_dtlz2_problem, "n_obj": 4, "epsilons": [0.05] * 4},
    "DTLZ3": {"model_func": get_dtlz3_problem, "n_obj": 4, "epsilons": [0.05] * 4}
}
ALGORITHMS = ['eps_nsgaii', 'sse_nsgaii', 'borg', 'generational_borg']
SEEDS = [12345, 23403, 39349, 60930, 93489]

# Output path for the final aggregated metrics CSV
LOCAL_AGGREGATED_METRICS_OUTPUT_FILE = "./all_calculated_metrics_summary.csv"

# Global dictionaries to store pre-loaded reference sets (objectives only)
REF_OBJECTIVES_STORE = {}

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

def preload_all_reference_sets():
    """ Loads/generates all necessary reference objective sets once. """
    print("Preloading all reference objective sets...")
    # JUSTICE
    if os.path.exists(GLOBAL_JUSTICE_REF_SET_CSV_PATH):
        full_global_df = pd.read_csv(GLOBAL_JUSTICE_REF_SET_CSV_PATH)
        m_justice = get_justice_model(); #p_justice = to_problem(m_justice, searchover='levers')
        obj_names_justice = [o.name for o in m_justice.outcomes]
        try:
            REF_OBJECTIVES_STORE['JUSTICE'] = full_global_df[obj_names_justice].copy()
            print(f"  Loaded global JUSTICE reference objectives. Shape: {REF_OBJECTIVES_STORE['JUSTICE'].shape}")
        except KeyError: # Handle error if columns missing
            print(f"  ERROR: Objective columns {obj_names_justice} not in {GLOBAL_JUSTICE_REF_SET_CSV_PATH}.")
            REF_OBJECTIVES_STORE['JUSTICE'] = pd.DataFrame()
    else:
        print(f"  ERROR: Global JUSTICE reference file not found: {GLOBAL_JUSTICE_REF_SET_CSV_PATH}")
        REF_OBJECTIVES_STORE['JUSTICE'] = pd.DataFrame()

    # DTLZ2
    m_dtlz2 = get_dtlz2_problem(n_objectives=4); p_dtlz2 = to_problem(m_dtlz2, searchover='levers')
    obj_names_dtlz2 = [o.name for o in m_dtlz2.outcomes]
    true_full_dtlz2 = m_dtlz2.generate_true_pareto_solutions(n_points=1000) 
    REF_OBJECTIVES_STORE['DTLZ2'] = true_full_dtlz2[obj_names_dtlz2].copy()
    print(f"  Generated DTLZ2 true reference objectives. Shape: {REF_OBJECTIVES_STORE['DTLZ2'].shape}")

    # DTLZ3
    m_dtlz3 = get_dtlz3_problem(n_objectives=4); p_dtlz3 = to_problem(m_dtlz3, searchover='levers')
    obj_names_dtlz3 = [o.name for o in m_dtlz3.outcomes]
    true_full_dtlz3 = m_dtlz3.generate_true_pareto_solutions(n_points=1000)
    REF_OBJECTIVES_STORE['DTLZ3'] = true_full_dtlz3[obj_names_dtlz3].copy()
    print(f"  Generated DTLZ3 true reference objectives. Shape: {REF_OBJECTIVES_STORE['DTLZ3'].shape}")

def calculate_metrics_for_single_run(problem_name_arg, algorithm_arg, cores_arg, seed_arg, model_obj, problem_obj):
    """
    Loads data for a single experimental run and computes all metrics over NFE.
    """
    print(f"  Processing: {problem_name_arg}, Algo: {algorithm_arg}, Cores: {cores_arg}, Seed: {seed_arg}")

    archive_tar_gz_dir = os.path.join(ARCHIVES_BASE, problem_name_arg,
                                    f"{cores_arg}cores", algorithm_arg, f"seed{seed_arg}")
    archive_tar_gz_file = os.path.join(archive_tar_gz_dir, "archive.tar.gz")

    hdf5_dir = os.path.join(HDF5_BASE, problem_name_arg,
                            f"{cores_arg}cores", algorithm_arg, f"seed{seed_arg}")
    hdf5_file_name = f"final_state_{problem_name_arg}_{algorithm_arg}_{cores_arg}cores_seed{seed_arg}.h5"
    hdf5_file_path = os.path.join(hdf5_dir, hdf5_file_name)

    if not os.path.exists(archive_tar_gz_file):
        print(f"    ERROR: Archive .tar.gz file not found: {archive_tar_gz_file}")
        return pd.DataFrame() # Return empty if no archive history
    
    archives_history = ArchiveLogger.load_archives(archive_tar_gz_file)
    if not archives_history:
        print(f"    ERROR: No archives loaded from {archive_tar_gz_file}")
        return pd.DataFrame()
    
    archive_items = list(archives_history.items())

    epsilon_progress_df_for_run = pd.DataFrame()
    total_runtime_for_run = np.nan
    if os.path.exists(hdf5_file_path):
        try:
            with h5py.File(hdf5_file_path, "r") as hf:
                if "epsilon_progress" in hf:
                    ep_group = hf["epsilon_progress"]
                    data_dict = {key: ep_group[key][()] for key in ep_group}
                    epsilon_progress_df_for_run = pd.DataFrame(data_dict)
                    if 'nfe' in epsilon_progress_df_for_run.columns:
                        epsilon_progress_df_for_run['nfe'] = epsilon_progress_df_for_run['nfe'].astype(int)
                if "runtime" in hf.attrs:
                    total_runtime_for_run = hf.attrs["runtime"]
        except Exception as e:
            print(f"    Error loading HDF5 file {hdf5_file_path}: {e}")
    else:
        print(f"    WARNING: HDF5 file not found: {hdf5_file_path}. EpsilonProgress and runtime will be missing.")

    current_ref_objectives = REF_OBJECTIVES_STORE.get(problem_name_arg)
    if current_ref_objectives is None or current_ref_objectives.empty:
        print(f"    ERROR: Reference objectives for {problem_name_arg} not loaded or empty. Cannot calculate some metrics.")
        gd_metric, ei_metric, sp_metric, sm_metric, standard_hv_metric_emaw = None, None, None, None, None
    else:
        gd_metric = GenerationalDistanceMetric(current_ref_objectives, problem_obj, d=1)
        ei_metric = EpsilonIndicatorMetric(current_ref_objectives, problem_obj)
        sp_metric = Spread(problem_obj, reference_set=current_ref_objectives)
        sm_metric = SpacingMetric(problem_obj)
        standard_hv_metric_emaw = HypervolumeMetric(current_ref_objectives, problem_obj)

    historical_metrics_list = []
    outcome_names_list = [o.name for o in model_obj.outcomes]
    direction_optimization_list = ["max" if o.kind == ScalarOutcome.MAXIMIZE else "min" for o in model_obj.outcomes]

    custom_hv_history_dict = {}
    if problem_name_arg == 'JUSTICE':
        if not os.path.exists(GLOBAL_JUSTICE_REF_SET_CSV_PATH):
            print(f"    ERROR Custom HV (JUSTICE): Global reference set CSV not found at {GLOBAL_JUSTICE_REF_SET_CSV_PATH}. Custom HV will be NaN.")
        else:
            custom_hv_output_df = calculate_hypervolume_from_archives(
                list_of_objectives=outcome_names_list,
                direction_of_optimization=direction_optimization_list,
                input_data_path=archive_tar_gz_dir, 
                file_name="archive.tar.gz",         
                output_data_path="./temp_hv_outputs_convergence_analysis",
                saving=False,              
                global_reference_set=True,        
                global_reference_set_path=os.path.dirname(GLOBAL_JUSTICE_REF_SET_CSV_PATH),
                global_reference_set_file=os.path.basename(GLOBAL_JUSTICE_REF_SET_CSV_PATH)
            )
            
            if not custom_hv_output_df.empty and 'nfe' in custom_hv_output_df.columns and 'hypervolume' in custom_hv_output_df.columns:
                custom_hv_output_df['nfe'] = custom_hv_output_df['nfe'].astype(int)
                for nfe_val, hv_val in zip(custom_hv_output_df['nfe'], custom_hv_output_df['hypervolume']):
                    custom_hv_history_dict[nfe_val] = hv_val
                print(f"    Custom HV history calculated for JUSTICE run ({problem_name_arg}, {algorithm_arg}, seed{seed_arg}, cores{cores_arg}).")
            else:
                print(f"    Custom HV for JUSTICE run returned empty or malformed DataFrame.")

    for nfe_str, archive_df_snapshot in archive_items:
        nfe_int = int(nfe_str)
        # EMAW metrics can usually take the full archive_df_snapshot if problem_obj is correct
        
        current_metrics_row = {"nfe": nfe_int}
        current_metrics_row["archive_size"] = len(archive_df_snapshot)

        # Hypervolume
        if problem_name_arg == 'JUSTICE':
            current_metrics_row["hypervolume"] = custom_hv_history_dict.get(nfe_int, np.nan)
        elif standard_hv_metric_emaw: # DTLZ uses standard EMAW HV
            try:
                current_metrics_row["hypervolume"] = standard_hv_metric_emaw.calculate(archive_df_snapshot)
            except: current_metrics_row["hypervolume"] = np.nan
        else: current_metrics_row["hypervolume"] = np.nan

        # Other EMAW Metrics
        if gd_metric: current_metrics_row["generational_distance"] = gd_metric.calculate(archive_df_snapshot)
        else: current_metrics_row["generational_distance"] = np.nan
        
        if ei_metric: current_metrics_row["epsilon_indicator"] = ei_metric.calculate(archive_df_snapshot)
        else: current_metrics_row["epsilon_indicator"] = np.nan
        
        if sp_metric: current_metrics_row["spread"] = sp_metric.calculate(archive_df_snapshot)
        else: current_metrics_row["spread"] = np.nan
        
        if sm_metric: current_metrics_row["spacing"] = sm_metric.calculate(archive_df_snapshot)
        else: current_metrics_row["spacing"] = np.nan
        
        historical_metrics_list.append(current_metrics_row)

    if not historical_metrics_list:
        print(f"    No historical metrics generated for run.")
        return pd.DataFrame()

    metrics_over_nfe_df = pd.DataFrame(historical_metrics_list)
    metrics_over_nfe_df.sort_values(by="nfe", inplace=True)

    # Merge with Epsilon Progress data 
    if not epsilon_progress_df_for_run.empty:
        metrics_over_nfe_df = pd.merge(metrics_over_nfe_df, epsilon_progress_df_for_run, on="nfe", how="left", suffixes=('', '_ep'))

    # Calculate NFE-based Time Efficiency
    if "hypervolume" in metrics_over_nfe_df.columns and metrics_over_nfe_df["hypervolume"].notna().any():
        hv_for_eff = metrics_over_nfe_df["hypervolume"].fillna(method='ffill').fillna(0).to_numpy()
        nfe_for_eff = metrics_over_nfe_df["nfe"].to_numpy()
        if len(hv_for_eff) > 1 and len(nfe_for_eff) > 1 and np.any(np.diff(nfe_for_eff) > 0):
            metrics_over_nfe_df["time_efficiency_nfe"] = efficiency_kernel(hv_for_eff, nfe_for_eff)
            if np.any(metrics_over_nfe_df["time_efficiency_nfe"] > 0):
                try:
                    first_nz_idx = metrics_over_nfe_df[metrics_over_nfe_df["time_efficiency_nfe"] > 0].index[0]
                    first_nz_val = metrics_over_nfe_df.loc[first_nz_idx, "time_efficiency_nfe"]
                    metrics_over_nfe_df.loc[metrics_over_nfe_df.index < first_nz_idx, "time_efficiency_nfe"] = first_nz_val
                    metrics_over_nfe_df["time_efficiency_nfe"].fillna(method='bfill', inplace=True); metrics_over_nfe_df["time_efficiency_nfe"].fillna(0, inplace=True)
                except IndexError: metrics_over_nfe_df["time_efficiency_nfe"] = 0.0
            else: metrics_over_nfe_df["time_efficiency_nfe"] = 0.0
        else: metrics_over_nfe_df["time_efficiency_nfe"] = 0.0
    else: metrics_over_nfe_df["time_efficiency_nfe"] = np.nan
    
    # Add total runtime for overall time efficiency (Final HV / Total Runtime)
    metrics_over_nfe_df["total_runtime"] = total_runtime_for_run
    if total_runtime_for_run > 0 and "hypervolume" in metrics_over_nfe_df.columns:
        final_hv = metrics_over_nfe_df["hypervolume"].iloc[-1] if not metrics_over_nfe_df["hypervolume"].empty and pd.notna(metrics_over_nfe_df["hypervolume"].iloc[-1]) else 0
        metrics_over_nfe_df["time_efficiency_total"] = final_hv / total_runtime_for_run
    else:
        metrics_over_nfe_df["time_efficiency_total"] = np.nan


    # Add metadata
    metrics_over_nfe_df["problem"] = problem_name_arg
    metrics_over_nfe_df["algorithm"] = algorithm_arg
    metrics_over_nfe_df["hpc_cores"] = cores_arg
    metrics_over_nfe_df["seed"] = seed_arg
    
    gc.collect()
    return metrics_over_nfe_df

def worker_process_task(task_args_tuple):
    """
    Helper function to unpack arguments from the task tuple and
    call calculate_metrics_for_single_run.
    This is to help with pickling for multiprocessing.
    """
    return calculate_metrics_for_single_run(*task_args_tuple)

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    preload_all_reference_sets() # Load/generate reference data once

    all_runs_metrics_dfs = []
    
    # Prepare arguments for parallel processing
    tasks_for_parallel_processing = []
    for prob_name, prob_config in PROBLEMS_CONFIG.items():
        model_inst = prob_config["model_func"](prob_config["n_obj"]) if prob_config["n_obj"] else prob_config["model_func"]()
        problem_inst = to_problem(model_inst, searchover="levers")
        for hpc_cores in CORE_COUNTS: # Corrected from your code snippet, should be CORE_COUNTS
            for algo in ALGORITHMS:
                for seed_val in SEEDS:
                    tasks_for_parallel_processing.append(
                        (prob_name, algo, hpc_cores, seed_val, model_inst, problem_inst)
                    )
    
    num_local_workers = 6 
    print(f"\nStarting local convergence analysis using {num_local_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_local_workers) as executor:
        # Use the new top-level helper function instead of the lambda
        results_from_pool = list(executor.map(worker_process_task, tasks_for_parallel_processing))
    
    all_runs_metrics_dfs = [df for df in results_from_pool if not df.empty] # Collect non-empty DataFrames

    if all_runs_metrics_dfs:
        final_aggregated_df = pd.concat(all_runs_metrics_dfs, ignore_index=True)
        final_aggregated_df.to_csv(LOCAL_AGGREGATED_METRICS_OUTPUT_FILE, index=False)
        print(f"\nSuccessfully completed all analyses. Aggregated metrics saved to: {LOCAL_AGGREGATED_METRICS_OUTPUT_FILE}")
    else:
        print("\nNo metrics were generated from any run. Check for errors in loading data or paths.")