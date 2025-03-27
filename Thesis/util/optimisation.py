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
import multiprocessing
import os
import time
from JUSTICE_fork.solvers.convergence.hypervolume import calculate_hypervolume_from_archives


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
            0.01, # welfare
            0.25, # years above threshold
            #0.01, # fraction of ensemble members above threshold
            10, # welfare loss damage
            10 # welfare loss abatement
        ]
    else:
        epsilons = [0.05] * len(model.outcomes)
    
    os.makedirs("archives", exist_ok=True)
    
    # Setup convergence metrics
    convergence_metrics = [
        ArchiveLogger(
            "./archives",
            [l.name for l in model.levers],
            [o.name for o in model.outcomes],
            base_filename=f"{algorithm_name}_seed{seed}.tar.gz",
        ),
        EpsilonProgress(),
    ]

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
        'nfe':nfe,
        'searchover':"levers",
        'epsilons':epsilons,
        'convergence':convergence_metrics,
        'algorithm':algorithm,
        'seed':seed,
        'population_size':100
    }

    if model.name == 'JUSTICE':
        opt_params['reference'] = Scenario("reference", ssp_rcp_scenario=model.scenario_string)
    
    # Run optimisation
    result, convergence = evaluator.optimize(**opt_params)

    # Calculate time taken
    runtime = time.time() - start_time
    
    return result, convergence, runtime

def analyse_convergence(results, model, algorithm_names, seeds):
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
            0.01, # welfare
            0.25, # years above threshold
            #0.01, # fraction of ensemble members above threshold
            10, # welfare loss damage
            10 # welfare loss abatement
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
    
    # Load archives and calculate metrics
    metrics_by_algorithm = {}
    
    # Create a multiprocessing pool for hypervolume calculation
    with multiprocessing.Pool() as pool:
        for algorithm in algorithm_names:
            print(f'Analysing convergence for {algorithm}')
            metrics_by_seed = []
            
            for i, seed_value in enumerate(seeds):
                print(f'Processing seed {seed_value}')
                
                # Archive file path
                archive_path = "./archives"
                archive_file = f"{algorithm}_seed{seed_value}.tar.gz"
                
                # Load archives
                archives = ArchiveLogger.load_archives(f"{archive_path}/{archive_file}")
                
                # Calculate metrics for each archive
                metrics = []
                previous_hv = None
                previous_nfe = None
                time_efficiencies = []
                
                try:
                    # Try to use custom hypervolume calculation
                    # Inject the pool into the global namespace of the hypervolume module
                    import JUSTICE_fork.solvers.convergence.hypervolume as hv_module
                    hv_module.pool = pool
                    
                    hv_results = calculate_hypervolume_from_archives(
                        list_of_objectives=outcome_names,
                        direction_of_optimization=direction_of_optimization,
                        input_data_path=archive_path,
                        file_name=archive_file,
                        output_data_path="./results",
                        saving=False,
                    )
                    
                    # Create a dictionary mapping NFE to hypervolume
                    hv_dict = dict(zip(hv_results['nfe'].astype(int), hv_results['hypervolume']))
                    
                    # Use custom hypervolume results
                    use_custom_hv = True
                    print(f"Using custom hypervolume for {algorithm} seed {seed_value}")
                    
                except Exception as e:
                    print(f"Error using custom hypervolume: {str(e)}")
                    print(f"Falling back to standard hypervolume for {algorithm} seed {seed_value}")
                    use_custom_hv = False
                
                # Process each archive
                for nfe, archive in archives.items():
                    nfe_int = int(nfe)
                    
                    # Remove index column from archive file
                    archive_no_index = archive.iloc[:, 1:]
                    
                    # Get hypervolume value
                    if use_custom_hv and nfe_int in hv_dict:
                        current_hv = hv_dict[nfe_int]
                    else:
                        # Fallback to standard hypervolume
                        current_hv = standard_hv.calculate(archive_no_index)
                    
                    # Calculate time efficiency
                    time_efficiency = 0.0
                    if previous_hv is not None and previous_nfe is not None and nfe_int > previous_nfe:
                        hv_change = current_hv - previous_hv
                        nfe_change = nfe_int - previous_nfe
                        time_efficiency = hv_change / nfe_change if nfe_change > 0 else 0.0
                    
                    time_efficiencies.append(time_efficiency)
                    
                    # Calculate all metrics
                    scores = {
                        "generational_distance": gd.calculate(archive_no_index),
                        "hypervolume": current_hv,
                        "epsilon_indicator": ei.calculate(archive_no_index),
                        "archive_size": len(archive_no_index),
                        "spread": sp.calculate(archive_no_index),
                        "spacing": sm.calculate(archive_no_index),
                        "time_efficiency": time_efficiency,
                        "nfe": nfe_int,
                    }
                    metrics.append(scores)
                    
                    # Store current values for next iteration
                    previous_hv = current_hv
                    previous_nfe = nfe_int
                
                # Fix the first point's time_efficiency
                if len(metrics) > 1:
                    # Find the first non-zero time efficiency value
                    non_zero_efficiencies = [te for te in time_efficiencies if te > 0]
                    if non_zero_efficiencies:
                        # Use the first non-zero value
                        first_valid_efficiency = non_zero_efficiencies[0]
                        metrics[0]["time_efficiency"] = first_valid_efficiency
                
                # Convert to DataFrame and sort by nfe
                metrics_df = pd.DataFrame.from_dict(metrics)
                metrics_df.sort_values(by="nfe", inplace=True)
                metrics_by_seed.append(metrics_df)
            
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
            for i, seed_value in enumerate(seeds):
                result, convergence, runtime = optimise_problem(
                    evaluator, model, algorithm, nfe, seed_value
                )
                results.append(result)
                convergences.append(convergence)
                runtimes.append(runtime)
    
    # Analyze convergence
    metrics = analyse_convergence(results, model, algorithms, seeds)
    
    return results, convergences, metrics, runtimes