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

# def optimise_problem(evaluator, model, algorithm_name, nfe, seed):
#     """
#     Optimise a problem using the specified MOEA
    
#     Parameters:
#     -----------
#     evaluator : MultiprocessingEvaluator
#         The evaluator to use for optimisation
#     model : Model
#         The problem to optimise
#     algorithm_name : str
#         The algorithm to use ('eps_nsgaii', 'borg', or 'generational_borg')
#     nfe : int
#         Number of function evaluations
#     seed : int
#         Random seed for reproducibility
    
#     Returns:
#     --------
#     tuple
#         (results, convergence): optimisation results and convergence metrics
#     """
    
#     # Set epsilon values
#     if model.name == 'JUSTICE':
#         epsilons = [
#             0.01, # welfare
#             0.25, # years above threshold
#             #0.01, # fraction of ensemble members above threshold
#             10, # welfare loss damage
#             10 # welfare loss abatement
#         ]
#     else:
#         epsilons = [0.05] * len(model.outcomes)
    
#     os.makedirs("archives", exist_ok=True)
    
#     # Setup convergence metrics
#     convergence_metrics = [
#         ArchiveLogger(
#             "./archives",
#             [l.name for l in model.levers],
#             [o.name for o in model.outcomes],
#             base_filename=f"{algorithm_name}_seed{seed}.tar.gz",
#         ),
#         EpsilonProgress(),
#     ]

#     # Select the appropriate MOEA
#     if algorithm_name == 'eps_nsgaii':
#         algorithm = EpsNSGAII
#     elif algorithm_name == 'borg':
#         algorithm = BorgMOEA
#     elif algorithm_name == 'generational_borg':
#         algorithm = GenerationalBorg
#     elif algorithm_name == 'sse_nsgaii':
#         algorithm = SteadyStateEpsNSGAII

#     # Log time for each run
#     start_time = time.time()

#     opt_params = {
#         'nfe':nfe,
#         'searchover':"levers",
#         'epsilons':epsilons,
#         'convergence':convergence_metrics,
#         'algorithm':algorithm,
#         'seed':seed,
#         'population_size':100
#     }

#     if model.name == 'JUSTICE':
#         opt_params['reference'] = Scenario("reference", ssp_rcp_scenario=model.scenario_string)
    
#     # Run optimisation
#     result, convergence = evaluator.optimize(**opt_params)

#     # Calculate time taken
#     runtime = time.time() - start_time
    
#     return result, convergence, runtime

# def analyse_convergence(results, model, algorithm_names, seeds):
#     """
#     Analyse convergence metrics from optimisation runs
    
#     Parameters:
#     -----------
#     results : list
#         List of optimisation results
#     model : Model
#         The problem to optimise
#     algorithm_names : list
#         List of algorithm names
#     seeds : list
#         List of seed values used
    
#     Returns:
#     --------
#     dict
#         Dictionary of metrics by algorithm and seed
#     """
#     # Create problem from model
#     problem = to_problem(model, searchover="levers")

#     # Set epsilon values based on the model
#     if model.name == 'JUSTICE':
#         epsilons = [
#             0.01, # welfare
#             0.25, # years above threshold
#             #0.01, # fraction of ensemble members above threshold
#             10, # welfare loss damage
#             10 # welfare loss abatement
#         ]
#     else:
#         epsilons = [0.05] * len(model.outcomes)
    
#     # Create reference set from all results
#     reference_set = epsilon_nondominated(results, epsilons, problem)
    
#     # Get outcome names and directions for custom hypervolume
#     outcome_names = [o.name for o in model.outcomes]
#     direction_of_optimization = []
#     for outcome in model.outcomes:
#         if outcome.kind == ScalarOutcome.MAXIMIZE:
#             direction_of_optimization.append("max")
#         else:
#             direction_of_optimization.append("min")
    
#     # Initialize standard metrics for other calculations
#     gd = GenerationalDistanceMetric(reference_set, problem, d=1)
#     ei = EpsilonIndicatorMetric(reference_set, problem)
#     sp = Spread(problem)
#     sm = SpacingMetric(problem)
    
#     # For fallback if custom hypervolume fails
#     standard_hv = HypervolumeMetric(reference_set, problem)
    
#     # Load archives and calculate metrics
#     metrics_by_algorithm = {}
    
#     # Create a multiprocessing pool for hypervolume calculation
#     with multiprocessing.Pool() as pool:
#         for algorithm in algorithm_names:
#             print(f'Analysing convergence for {algorithm}')
#             metrics_by_seed = []
            
#             for i, seed_value in enumerate(seeds):
#                 print(f'Processing seed {seed_value}')
                
#                 # Archive file path
#                 archive_path = "./archives"
#                 archive_file = f"{algorithm}_seed{seed_value}.tar.gz"
                
#                 # Load archives
#                 archives = ArchiveLogger.load_archives(f"{archive_path}/{archive_file}")
                
#                 # Calculate metrics for each archive
#                 metrics = []
#                 previous_hv = None
#                 previous_nfe = None
#                 time_efficiencies = []
                
#                 try:
#                     # Try to use custom hypervolume calculation
#                     # Inject the pool into the global namespace of the hypervolume module
#                     import JUSTICE_fork.solvers.convergence.hypervolume as hv_module
#                     hv_module.pool = pool
                    
#                     hv_results = calculate_hypervolume_from_archives(
#                         list_of_objectives=outcome_names,
#                         direction_of_optimization=direction_of_optimization,
#                         input_data_path=archive_path,
#                         file_name=archive_file,
#                         output_data_path="./results",
#                         saving=False,
#                     )
                    
#                     # Create a dictionary mapping NFE to hypervolume
#                     hv_dict = dict(zip(hv_results['nfe'].astype(int), hv_results['hypervolume']))
                    
#                     # Use custom hypervolume results
#                     use_custom_hv = True
#                     print(f"Using custom hypervolume for {algorithm} seed {seed_value}")
                    
#                 except Exception as e:
#                     print(f"Error using custom hypervolume: {str(e)}")
#                     print(f"Falling back to standard hypervolume for {algorithm} seed {seed_value}")
#                     use_custom_hv = False
                
#                 # Process each archive
#                 for nfe, archive in archives.items():
#                     nfe_int = int(nfe)
                    
#                     # Remove index column from archive file
#                     archive_no_index = archive.iloc[:, 1:]
                    
#                     # Get hypervolume value
#                     if use_custom_hv and nfe_int in hv_dict:
#                         current_hv = hv_dict[nfe_int]
#                     else:
#                         # Fallback to standard hypervolume
#                         current_hv = standard_hv.calculate(archive_no_index)
                    
#                     # Calculate time efficiency
#                     time_efficiency = 0.0
#                     if previous_hv is not None and previous_nfe is not None and nfe_int > previous_nfe:
#                         hv_change = current_hv - previous_hv
#                         nfe_change = nfe_int - previous_nfe
#                         time_efficiency = hv_change / nfe_change if nfe_change > 0 else 0.0
                    
#                     time_efficiencies.append(time_efficiency)
                    
#                     # Calculate all metrics
#                     scores = {
#                         "generational_distance": gd.calculate(archive_no_index),
#                         "hypervolume": current_hv,
#                         "epsilon_indicator": ei.calculate(archive_no_index),
#                         "archive_size": len(archive_no_index),
#                         "spread": sp.calculate(archive_no_index),
#                         "spacing": sm.calculate(archive_no_index),
#                         "time_efficiency": time_efficiency,
#                         "nfe": nfe_int,
#                     }
#                     metrics.append(scores)
                    
#                     # Store current values for next iteration
#                     previous_hv = current_hv
#                     previous_nfe = nfe_int
                
#                 # Fix the first point's time_efficiency
#                 if len(metrics) > 1:
#                     # Find the first non-zero time efficiency value
#                     non_zero_efficiencies = [te for te in time_efficiencies if te > 0]
#                     if non_zero_efficiencies:
#                         # Use the first non-zero value
#                         first_valid_efficiency = non_zero_efficiencies[0]
#                         metrics[0]["time_efficiency"] = first_valid_efficiency
                
#                 # Convert to DataFrame and sort by nfe
#                 metrics_df = pd.DataFrame.from_dict(metrics)
#                 metrics_df.sort_values(by="nfe", inplace=True)
#                 metrics_by_seed.append(metrics_df)
            
#             metrics_by_algorithm[algorithm] = metrics_by_seed
    
#     return metrics_by_algorithm

# def run_optimisation_experiment(model, algorithms, nfe, seeds, core_count):
#     """
#     Run optimisation experiment with multiple algorithms and seeds
    
#     Parameters:
#     -----------
#     model : Model
#         The problem to optimize
#     algorithms : list
#         List of algorithm names
#     nfe : int
#         Number of function evaluations
#     seeds : int
#         Number of seeds to use
#     core_count : int
#         Number of cores to use for multiprocessing
    
#     Returns:
#     --------
#     tuple
#         (results, convergences, metrics, runtime): optimisation results, convergence data, metrics and runtime
#     """
    
#     ema_logging.log_to_stderr(ema_logging.INFO)
    
#     results = []
#     convergences = []
#     runtimes = []
    
#     with MultiprocessingEvaluator(model, n_processes=core_count) as evaluator:
#         for algorithm in algorithms:
#             for i, seed_value in enumerate(seeds):
#                 result, convergence, runtime = optimise_problem(
#                     evaluator, model, algorithm, nfe, seed_value
#                 )
#                 results.append(result)
#                 convergences.append(convergence)
#                 runtimes.append(runtime)
    
#     # Analyze convergence
#     metrics = analyse_convergence(results, model, algorithms, seeds)
    
#     return results, convergences, metrics, runtimes

# Uncomment for HPC runs
SCRATCH_BASE = "/scratch/wmvanderlinden/MOEA_convergence"
ARCHIVES_PATH = os.path.join(SCRATCH_BASE, "archives")
os.makedirs(ARCHIVES_PATH, exist_ok=True)

def optimise_problem(evaluator, model, algorithm_name, nfe, seed):
    """
    Optimise a problem using the specified MOEA
    
    Parameters:
    -----------
    evaluator : MultiprocessingEvaluator
        The evaluator to use for optimisation
    model : Model
        The problem to optimise
    algorithm_name : str
        The algorithm to use ('eps_nsgaii', 'borg', or 'generational_borg')
    nfe : int
        Number of function evaluations
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (results, convergence): optimisation results and convergence metrics
    """
    # Set epsilon values
    if model.name == 'JUSTICE':
        epsilons = [
            0.01,  # welfare
            0.25,  # years above threshold
            #0.01, # fraction of ensemble members above threshold
            10,    # welfare loss damage
            10     # welfare loss abatement
        ]
    else:
        epsilons = [0.05] * len(model.outcomes)
    
    # Uncomment for local runs
    os.makedirs("archives", exist_ok=True)
    
    # Setup convergence metrics for HPC
    convergence_metrics = [
        ArchiveLogger(
            ARCHIVES_PATH,
            [l.name for l in model.levers],
            [o.name for o in model.outcomes],
            base_filename=f"{algorithm_name}_seed{seed}.tar.gz",
        ),
        EpsilonProgress(),
    ]

    # Setup convergence metrics for local runs
    # convergence_metrics = [
    #     ArchiveLogger(
    #         "./archives",
    #         [l.name for l in model.levers],
    #         [o.name for o in model.outcomes],
    #         base_filename=f"{algorithm_name}_seed{seed}.tar.gz",
    #     ),
    #     EpsilonProgress(),
    # ]
    
    # Select the appropriate MOEA
    if algorithm_name == 'eps_nsgaii':
        algorithm = EpsNSGAII
    elif algorithm_name == 'borg':
        algorithm = BorgMOEA
    elif algorithm_name == 'generational_borg':
        algorithm = GenerationalBorg
    elif algorithm_name == 'sse_nsgaii':
        algorithm = SteadyStateEpsNSGAII
    
    # Log time for each run
    start_time = time.time()
    
    opt_params = {
        'nfe': nfe,
        'searchover': "levers",
        'epsilons': epsilons,
        'convergence': convergence_metrics,
        'algorithm': algorithm,
        'seed': seed,
        'population_size': 2
    }
    
    if model.name == 'JUSTICE':
        opt_params['reference'] = Scenario("reference", ssp_rcp_scenario=model.scenario_string)
    
    # Run optimisation
    result, convergence = evaluator.optimize(**opt_params)
    
    # Calculate time taken
    runtime = time.time() - start_time
    
    return result, convergence, runtime

def adaptive_batch_size(n_archives):
    if n_archives > 10000: return 100
    elif n_archives > 1000: return 50
    else: return 10

@jit(nopython=True, parallel=True)
def efficiency_kernel(hv_values, nfe_values):
    efficiencies = np.zeros_like(hv_values)
    for i in prange(1, len(hv_values)): 
        if nfe_values[i] > nfe_values[i-1]:
            delta_hv = hv_values[i] - hv_values[i-1]
            delta_nfe = nfe_values[i] - nfe_values[i-1]
            efficiencies[i] = delta_hv / delta_nfe if delta_nfe > 0 else 0.0
    return efficiencies

def process_seed(algorithm, seed_value, archives_path, metrics_data, problem_metrics):
    """
    Process a single seed for an algorithm
    
    Parameters:
    -----------
    algorithm : str
        Algorithm name
    seed_value : int
        Seed value
    archives_path : str
        Path to archives
    metrics_data : dict
        Dictionary with metric data
    problem_metrics : tuple
        Tuple containing (problem, reference_set, gd, ei, sp, sm, standard_hv)
    
    Returns:
    --------
    DataFrame
        DataFrame with metrics for this seed
    """
    problem, reference_set, gd, ei, sp, sm, standard_hv = problem_metrics
    
    archive_file = f"{algorithm}_seed{seed_value}.tar.gz"
    print(f'Processing {algorithm} seed {seed_value}')
    
    # Load archives (upper for local, lower for HPC)
    #archives = ArchiveLogger.load_archives(f"{archives_path}/{archive_file}")
    archives = ArchiveLogger.load_archives(os.path.join(ARCHIVES_PATH, archive_file))
    
    # Convert archives to list for batching
    archive_items = list(archives.items())
    batch_size = adaptive_batch_size(len(archive_items))

    # Calculate metrics for each archive
    metrics = [None] * len(archive_items)
    current_idx = 0

    try:
        # Try to use custom hypervolume calculation
        hv_results = calculate_hypervolume_from_archives(
            list_of_objectives=metrics_data['outcome_names'],
            direction_of_optimization=metrics_data['direction_of_optimization'],
            input_data_path=archives_path,
            file_name=archive_file,
            output_data_path="./results",
            saving=False,
            #pool=None #DIT NOG EEN KEER FIKSEN
        )
        
        # Create a dictionary mapping NFE to hypervolume
        hv_dict = dict(zip(hv_results['nfe'].astype(int), hv_results['hypervolume']))
        use_custom_hv = True
        print(f"Using custom hypervolume for {algorithm} seed {seed_value}")
    except Exception as e:
        print(f"Error using custom hypervolume: {str(e)}")
        print(f"Falling back to standard hypervolume for {algorithm} seed {seed_value}")
        use_custom_hv = False

    # Process archives in batches
    for i in range(0, len(archive_items), batch_size):
        batch = archive_items[i:i+batch_size]
        
        # Process each archive in the current batch
        for nfe, archive in batch:
            nfe_int = int(nfe)
            archive_no_index = archive.iloc[:, 1:]
        
            # Get hypervolume value
            if use_custom_hv and nfe_int in hv_dict:
                current_hv = hv_dict[nfe_int]
            else:
                # # Fallback to standard hypervolume
                # current_hv = standard_hv.calculate(archive_no_index)
                print('Hypervolume error')
            
            # Calculate all metrics
            with ThreadPoolExecutor() as executor:
                futures = {
                    'gd': executor.submit(gd.calculate, archive_no_index),
                    'ei': executor.submit(ei.calculate, archive_no_index),
                    'spread': executor.submit(sp.calculate, archive_no_index),
                    'spacing': executor.submit(sm.calculate, archive_no_index),
                    #'hypervolume': executor.submit(standard_hv.calculate, archive_no_index)
                }
                scores = {
                    "generational_distance": futures['gd'].result(),
                    "epsilon_indicator": futures['ei'].result(),
                    "spread": futures['spread'].result(),
                    "spacing": futures['spacing'].result(),
                    "hypervolume": current_hv,
                    #"hypervolume": futures['hypervolume'].result(),
                    "archive_size": len(archive_no_index),
                    "nfe": nfe_int,
                }
            
            metrics[current_idx] = scores
            current_idx += 1

    # Vectorized time efficiency calculation
    if len(metrics) > 1:
        import numpy as np
        nfe_values = np.array([m["nfe"] for m in metrics])
        hv_values = np.array([m["hypervolume"] for m in metrics])
        
        # Calculate efficiencies using Numba kernel ONLY
        efficiencies = efficiency_kernel(hv_values, nfe_values)
        
        # Set first point's efficiency to first non-zero value
        if np.any(efficiencies > 0):
            first_nonzero_idx = np.nonzero(efficiencies)[0][0]
            efficiencies[:first_nonzero_idx] = efficiencies[first_nonzero_idx]
        
        # Update metrics with calculated efficiencies
        for i, m in enumerate(metrics):
            m["time_efficiency"] = efficiencies[i]

    # Convert to DataFrame and sort by nfe
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.sort_values(by="nfe", inplace=True)

    del archives
    del archive_items
    del batch
    gc.collect()

    return metrics_df

def analyse_convergence(results, model, algorithm_names, seeds, core_count=None):
    """
    Analyse convergence metrics from optimisation runs
    
    Parameters:
    -----------
    results : list
        List of optimisation results
    model : Model
        The problem to optimise
    algorithm_names : list
        List of algorithm names
    seeds : list
        List of seed values used
    core_count : int, optional
        Number of cores to use for parallel processing
        
    Returns:
    --------
    dict
        Dictionary of metrics by algorithm and seed
    """
    # Create problem from model
    problem = to_problem(model, searchover="levers")
    
    # Set epsilon values based on the model
    if model.name == 'JUSTICE':
        epsilons = [
            0.01,  # welfare
            0.25,  # years above threshold
            #0.01, # fraction of ensemble members above threshold
            10,    # welfare loss damage
            10     # welfare loss abatement
        ]
    else:
        epsilons = [0.05] * len(model.outcomes)
    
    # Create reference set from all results
    reference_set = epsilon_nondominated(results, epsilons, problem)
    
    # Get outcome names and directions for custom hypervolume
    outcome_names = [o.name for o in model.outcomes]
    direction_of_optimization = []
    for outcome in model.outcomes:
        if outcome.kind == ScalarOutcome.MAXIMIZE:
            direction_of_optimization.append("max")
        else:
            direction_of_optimization.append("min")
    
    # Initialize standard metrics for other calculations
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    sp = Spread(problem)
    sm = SpacingMetric(problem)
    # For fallback if custom hypervolume fails
    standard_hv = HypervolumeMetric(reference_set, problem)
    
    # Prepare metrics data dictionary
    metrics_data = {
        'outcome_names': outcome_names,
        'direction_of_optimization': direction_of_optimization
    }
    
    # Package problem metrics for passing to worker processes
    problem_metrics = (problem, reference_set, gd, ei, sp, sm, standard_hv)
    
    # Load archives and calculate metrics
    metrics_by_algorithm = {}
    archive_path = "./archives"
    
    # Use ProcessPoolExecutor for parallel processing of seeds
    # If core_count is not specified, use a reasonable default
    if core_count is None:
        core_count = min(len(seeds), 4)  # Conservative default
    
    for algorithm in algorithm_names:
        print(f'Analysing convergence for {algorithm}')
        
        # Process seeds in parallel
        with ProcessPoolExecutor(max_workers=core_count) as executor:
            # Create a partial function with fixed parameters
            process_func = partial(
                process_seed, 
                algorithm, 
                archives_path=archive_path,
                metrics_data=metrics_data,
                problem_metrics=problem_metrics
            )
            
            # Submit all tasks and gather results
            futures = [executor.submit(process_func, seed_value) for seed_value in seeds]
            metrics_by_seed = [future.result() for future in futures]
        
        metrics_by_algorithm[algorithm] = metrics_by_seed
    
    return metrics_by_algorithm

def run_optimisation_experiment(model, algorithms, nfe, seeds, core_count):
    """
    Run optimisation experiment with multiple algorithms and seeds
    
    Parameters:
    -----------
    model : Model
        The problem to optimize
    algorithms : list
        List of algorithm names
    nfe : int
        Number of function evaluations
    seeds : int
        Number of seeds to use
    core_count : int
        Number of cores to use for multiprocessing
        
    Returns:
    --------
    tuple
        (results, convergences, metrics, runtime): optimisation results, convergence data, metrics and runtime
    """
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    results = []
    convergences = []
    runtimes = []
    
    with MultiprocessingEvaluator(model, n_processes=core_count) as evaluator:
        for algorithm in algorithms:
            for seed_value in seeds:
                result, convergence, runtime = optimise_problem(
                    evaluator, model, algorithm, nfe, seed_value
                )
                
                results.append(result)
                convergences.append(convergence)
                runtimes.append(runtime)
    
    # Analyze convergence - pass the core_count to enable parallel processing
    metrics = analyse_convergence(results, model, algorithms, seeds, core_count)
    
    return results, convergences, metrics, runtimes