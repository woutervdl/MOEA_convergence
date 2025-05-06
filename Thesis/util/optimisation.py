from Thesis.algorithms.borgMOEA import BorgMOEA
from Thesis.algorithms.sse_nsga_ii import SteadyStateEpsNSGAII
from Thesis.util.spread import Spread
from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                     EpsilonProgress,
                                                     ArchiveLogger,
                                                     EpsilonIndicatorMetric,
                                                     GenerationalDistanceMetric,
                                                     SpacingMetric,
                                                     EpsNSGAII,
                                                     GenerationalBorg,
                                                     to_problem,
                                                     epsilon_nondominated)
from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench.em_framework.outcomes import ScalarOutcome
from ema_workbench.em_framework.points import Scenario 
import pandas as pd
import os
import time
from JUSTICE_fork.solvers.convergence.hypervolume import calculate_hypervolume_from_archives
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import gc
from numba import jit,prange
import numpy as np
import tempfile
import shutil

############### HPC paths
SCRATCH_BASE = "/scratch/wmvanderlinden/MOEA_convergence" 
ARCHIVES_PATH = os.path.join(SCRATCH_BASE, "full_archives_tar_gz") 
os.makedirs(ARCHIVES_PATH, exist_ok=True)
TEMP_LOGGER_BASE_HPC = os.path.join(SCRATCH_BASE, "temp_logger")
###############

######################## Local paths
# BASE_PATH_LOCAL = "." # Uses current directory (.) as the base
# ARCHIVES_PATH_LOCAL = os.path.join(BASE_PATH_LOCAL, "local_archives")
# TEMP_LOGGER_BASE_LOCAL = os.path.join(BASE_PATH_LOCAL, "local_temp_logger")
########################

def optimise_problem(evaluator, model, algorithm_name, nfe, seed, problem_name_for_path, cores_for_path):
    """ HPC version: Runs optimisation and saves .tar.gz to a structured path. """
    
    # Setting epsilon values
    if model.name == 'JUSTICE':
        epsilons = [
            0.01,  # Welfare
            0.25,  # Years above threshold
            #0.01, # Fraction of ensemble members above threshold 
            10,    # Welfare loss damage 
            10     # Welfare loss abatement
        ]
    else:
        epsilons = [0.05] * len(model.outcomes)

    ############# HPC runs
    run_archive_dir = os.path.join(
        ARCHIVES_PATH,
        problem_name_for_path,       
        f"{cores_for_path}cores",      
        algorithm_name,                   
        f"seed{seed}"                      
    )
    os.makedirs(run_archive_dir, exist_ok=True)
    temp_logger_parent_dir = TEMP_LOGGER_BASE_HPC
    os.makedirs(temp_logger_parent_dir, exist_ok=True)
    #############

    ############## Local runs
    # run_archive_dir = os.path.join(
    #     ARCHIVES_PATH_LOCAL,
    #     problem_name_for_path,       
    #     f"{cores_for_path}cores",      
    #     algorithm_name,                   
    #     f"seed{seed}"                      
    # )
    # os.makedirs(run_archive_dir, exist_ok=True)
    # os.makedirs(run_archive_dir, exist_ok=True)
    # temp_logger_parent_dir = TEMP_LOGGER_BASE_LOCAL
    # os.makedirs(temp_logger_parent_dir, exist_ok=True)
    ##############    
    
    final_archive_name = "archive.tar.gz"
    final_archive_path = os.path.join(run_archive_dir, final_archive_name)

    # Use a unique temporary directory on the compute node's local scratch for ArchiveLogger
    temp_log_base_dir = tempfile.mkdtemp(
        prefix=f"{problem_name_for_path}_{algorithm_name}_cores{cores_for_path}_seed{seed}_pid{os.getpid()}_",
        dir=temp_logger_parent_dir
    )

    convergence_metrics = [
        ArchiveLogger(
            temp_log_base_dir, # Log to temp dir first
            [l.name for l in model.levers],
            [o.name for o in model.outcomes],
            base_filename=final_archive_name 
        ),
        EpsilonProgress(),
    ]

    if algorithm_name == 'eps_nsgaii': algorithm = EpsNSGAII
    elif algorithm_name == 'borg': algorithm = BorgMOEA
    elif algorithm_name == 'generational_borg': algorithm = GenerationalBorg
    elif algorithm_name == 'sse_nsgaii': algorithm = SteadyStateEpsNSGAII

    start_time = time.time()

    opt_params = {
        'nfe': nfe, 
        'searchover': "levers", 
        'epsilons': epsilons,
        'convergence': convergence_metrics,
        'algorithm': algorithm, 
        'seed': seed, 
        'population_size': 100
    }

    if model.name == 'JUSTICE':
        opt_params['reference'] = Scenario("reference", ssp_rcp_scenario=model.scenario_string)

    result_final_archive, convergence = evaluator.optimize(**opt_params)
    runtime = time.time() - start_time

    # Move the archive to final destination
    source_archive_path = os.path.join(temp_log_base_dir, final_archive_name)
    shutil.move(source_archive_path, final_archive_path)
    shutil.rmtree(temp_log_base_dir)

    return result_final_archive, convergence, runtime 

def run_optimisation_experiment(model, algorithms, nfe, seeds, core_count, problem_name_for_path):
    """
    Run optimisation with multiple algorithms and seeds
    
    Parameters:
    -----------
    model : Model
        The problem to optimise
    algorithms : list
        List of algorithm names
    nfe : int
        Number of function evaluations
    seeds : int
        Number of seeds to use
    core_count : int
        Number of cores to use for multiprocessing
    problem_name_for_path : str
        Name of the problem for path structure
        
    Returns:
    --------
    tuple
        (final_archives_list, epsilon_progress_list, runtimes_list): final archives, epsilon progress and runtimes
    """
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    final_archives_list = []
    epsilon_progress_list = []
    runtimes_list = []

    with MultiprocessingEvaluator(model, n_processes=core_count) as evaluator:
        for algorithm_name in algorithms:
            for seed_value in seeds:
                print(f"HPC Starting: {problem_name_for_path}, {algorithm_name}, {core_count} cores, seed {seed_value}, NFE {nfe}")
                
                result, convergence, runtime = optimise_problem(
                    evaluator, model, algorithm_name, nfe, seed_value,
                    problem_name_for_path=problem_name_for_path,
                    cores_for_path=core_count             
                )
                final_archives_list.append(result)
                epsilon_progress_list.append(convergence)
                runtimes_list.append(runtime)
    
    return final_archives_list, epsilon_progress_list, runtimes_list