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
    
    # Run optimisation
    result, convergence = evaluator.optimize(
        nfe=nfe,
        searchover="levers",
        epsilons=epsilons,
        convergence=convergence_metrics,
        algorithm=algorithm,
        seed=seed
    )
    
    return result, convergence

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
        metrics_by_seed = []
        
        for seed in range(seeds):
            # Load archives
            archives = ArchiveLogger.load_archives(f"./archives/{algorithm}_seed{seed}.tar.gz")
            
            # Calculate metrics for each archive
            metrics = []
            for nfe, archive in archives.items():
                
                # Remove index column from archive file
                archive_no_index = archive.iloc[:, 1:]
                
                scores = {
                    "generational_distance": gd.calculate(archive_no_index),
                    "hypervolume": hv.calculate(archive_no_index),
                    "epsilon_indicator": ei.calculate(archive_no_index),
                    "archive_size": len(archive_no_index),
                    "spread": sp.calculate(archive_no_index),
                    "spacing": sm.calculate(archive_no_index),
                    "nfe": int(nfe),
                }
                metrics.append(scores)
            
            # Convert to DataFrame and sort by nfe
            metrics_df = pd.DataFrame.from_dict(metrics)
            metrics_df.sort_values(by="nfe", inplace=True)
            metrics_by_seed.append(metrics_df)
        
        metrics_by_algorithm[algorithm] = metrics_by_seed
    
    return metrics_by_algorithm

def run_optimisation_experiment(model, algorithms, nfe, seeds):
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
    
    Returns:
    --------
    tuple
        (results, convergences, metrics): optimisation results, convergence data, and metrics
    """
    
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    results = []
    convergences = []
    
    with MultiprocessingEvaluator(model) as evaluator:
        for algorithm in algorithms:
            for seed in range(seeds):
                result, convergence = optimise_problem(
                    evaluator, model, algorithm, nfe, seed
                )
                results.append(result)
                convergences.append(convergence)
    
    # Analyze convergence
    metrics = analyse_convergence(results, model, algorithms, seeds)
    
    return results, convergences, metrics