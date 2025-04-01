from ema_workbench import ema_logging
from Thesis.util.model_definitions import get_dtlz2_problem, get_dtlz3_problem, get_justice_model
from Thesis.util.optimisation import *
import os
import random

def run_experiments():
    """
    Run experiments for different problems and algorithms
    """
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Create main results directory
    os.makedirs("./results", exist_ok=True)

    # Define problems
    problems = [
        #('DTLZ2', get_dtlz2_problem(4)),  
        ('DTLZ3', get_dtlz3_problem(4)),
        #('JUSTICE', get_justice_model())
    ]

    # Define algorithms
    algorithms = ['eps_nsgaii','borg']#,'generational_borg']

    # Define the core counts to be tested
    core_count = [4,6]
    
    # Define experiment parameters
    nfe =2000 
    num_seeds = 4

    # Using a large range to minimize correlation chances
    random.seed(12345)  # Master seed for reproducibility
    seed_values = random.sample(range(10000, 1000000), num_seeds)
    print(f"Using seed values: {seed_values}")    

    # Run experiments for each problem
    for problem_name, model in problems:
        print(f"Running experiments for {problem_name}")
        
        # Create problem directory
        problem_dir = os.path.join("./results", problem_name)
        os.makedirs(problem_dir, exist_ok=True)

        # Run experiments for each core count
        for cores in core_count:
            print(f"Running experiments for {problem_name} with {cores} cores")

            # Create core directory
            core_dir = os.path.join(problem_dir, f"{cores}_cores")
            os.makedirs(core_dir, exist_ok=True)
        
            # Create algorithm directories
            for algorithm in algorithms:
                algorithm_dir = os.path.join(core_dir, algorithm)
                os.makedirs(algorithm_dir, exist_ok=True)
            
            results, convergences, metrics, runtimes = run_optimisation_experiment(
                model, algorithms, nfe, seed_values, cores
            )
            
            # Save results
            save_results(results, convergences, metrics, runtimes, problem_name, algorithms, seed_values, cores)
        
        print(f"Completed experiments for {problem_name}")

def save_results(results, convergences, metrics, runtimes, problem_name, algorithms, seeds, cores):
    """
    Save experiment results to files in organized folder structure
    
    Parameters:
    -----------
    results : list
        List of optimisation results
    convergences : list
        List of convergence data
    metrics : dict
        Dictionary of performance metrics
    runtimes : list
        List of runtimes
    problem_name : str
        Name of the problem
    algorithms : list
        List of algorithm names
    seeds : list
        List of seed values used
    cores : int
        Number of cores used
    """
    num_seeds = len(seeds)  # Get the number of seeds
    
    # Save results
    for i, result in enumerate(results):
        algorithm_idx = i // num_seeds
        seed_idx = i % num_seeds
        algorithm = algorithms[algorithm_idx]
        seed_value = seeds[seed_idx]  # Get the actual seed value
        
        # Create path with hierarchical structure
        result_path = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
        filename = os.path.join(result_path, f"seed{seed_idx}_results.csv")
        result.to_csv(filename, index=False)
    
    # Save convergence data
    for i, convergence in enumerate(convergences):
        algorithm_idx = i // num_seeds
        seed_idx = i % num_seeds
        algorithm = algorithms[algorithm_idx]
        seed_value = seeds[seed_idx]  # Get the actual seed value
        
        # Create path with hierarchical structure
        convergence_path = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
        filename = os.path.join(convergence_path, f"seed{seed_idx}_convergence.csv")
        convergence.to_csv(filename, index=False)
    
    # Save metrics
    for algorithm, algorithm_metrics in metrics.items():
        for seed_idx, seed_metrics in enumerate(algorithm_metrics):
            if seed_idx < num_seeds:  # Make sure we don't go out of bounds
                seed_value = seeds[seed_idx]  # Get the actual seed value
                
                # Create path with hierarchical structure
                metrics_path = os.path.join("./results", problem_name, f"{cores}_cores", algorithm)
                filename = os.path.join(metrics_path, f"seed{seed_idx}_metrics.csv")
                seed_metrics.to_csv(filename, index=False)

    # Save runtimes
    runtime_data = []
    for i, runtime in enumerate(runtimes):
        algorithm_idx = i // num_seeds
        seed_idx = i % num_seeds
        algorithm = algorithms[algorithm_idx]
        seed_value = seeds[seed_idx]  # Get the actual seed value
        
        runtime_data.append({
            'problem': problem_name,
            'algorithm': algorithm,
            'seed': seed_idx,
            'seed_value': seed_value,  # Store the actual seed value
            'cores': cores,
            'runtime': runtime
        })

    # Create runtime DataFrame and save
    runtime_df = pd.DataFrame(runtime_data)
    runtime_path = os.path.join("./results", problem_name)
    runtime_filename = os.path.join(runtime_path, f"runtimes_{cores}_cores.csv")
    runtime_df.to_csv(runtime_filename, index=False)


if __name__ == "__main__":
    run_experiments()