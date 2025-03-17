from Thesis.algorithms.borgMOEA import BorgMOEA
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
import pandas as pd
import os
import time

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

    # Log time for each run
    start_time = time.time()
    
    # Run optimisation
    result, convergence = evaluator.optimize(
        nfe=nfe,
        searchover="levers",
        epsilons=epsilons,
        convergence=convergence_metrics,
        algorithm=algorithm,
        seed=seed
    )

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
    seeds : int
        Number of seeds used
    
    Returns:
    --------
    dict
        Dictionary of metrics by algorithm and seed
    """
    # Create problem from model
    problem = to_problem(model, searchover="levers")
    
    # Create reference set from all results
    reference_set = epsilon_nondominated(results, [0.05] * len(model.outcomes), problem)
    
    # Initialise metrics
    hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    sp = Spread(problem)
    sm = SpacingMetric(problem)
    
    # Load archives and calculate metrics
    metrics_by_algorithm = {}
    
    for algorithm in algorithm_names:
        print(f'Analysing convergence for {algorithm}')
        metrics_by_seed = []
        
        for i, seed_value in enumerate(seeds):
            print(f'Processing seed {seed_value}')

            # Load archives
            archives = ArchiveLogger.load_archives(f"./archives/{algorithm}_seed{seed_value}.tar.gz")
            
            # Calculate metrics for each archive
            metrics = []
            previous_hv = None
            previous_nfe = None
            time_efficiencies = []

            for nfe, archive in archives.items():
                
                # Remove index column from archive file
                archive_no_index = archive.iloc[:, 1:]

                # Calculating time efficiency
                current_hv = hv.calculate(archive_no_index)
                time_efficiency = 0.0
                if previous_hv is not None and int(nfe)>previous_nfe:
                    hv_change = current_hv - previous_hv
                    nfe_change = int(nfe) - previous_nfe
                    time_efficiency = hv_change / nfe_change
                
                time_efficiencies.append(time_efficiency)
                
                # Calculate metrics
                scores = {
                    "generational_distance": gd.calculate(archive_no_index),
                    "hypervolume": current_hv,
                    "epsilon_indicator": ei.calculate(archive_no_index),
                    "archive_size": len(archive_no_index),
                    "spread": sp.calculate(archive_no_index),
                    "spacing": sm.calculate(archive_no_index),
                    "time_efficiency": time_efficiency,
                    "nfe": int(nfe),
                }
                metrics.append(scores)

                # Store current values for next iteration
                previous_hv = current_hv
                previous_nfe = nfe
            
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