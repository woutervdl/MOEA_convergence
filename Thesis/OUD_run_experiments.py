from platypus import DTLZ2, EpsNSGAII, Real, TournamentSelector
from Thesis.algorithms.borgMOEA import BorgMOEA
from ema_workbench.em_framework.samplers import LHSSampler
from scipy import stats

#Defining the parameter ranges for the algorithms
param_ranges = {
    'eps_NSGA2': {
        'epsilons': (0.001, 0.1),
        'population_size': (30, 150),
        'tournament_size': [2, 3, 4, 5, 6, 7]
    },
    'Borg': {
        'epsilons': (0.001, 0.1),
        'population_size': (30, 150),
        'tournament_size': [2, 3, 4, 5, 6, 7],
        'recency_list_size': (20, 80),
        'max_mutation_index': (5, 20),
        'selection_ratio': (0.01, 0.05),
    }
}

seed_numbers = [1]  #Seed numbers for the experiments

#Defining the problem setup
def setup_problem():
    n_variables = 13
    n_objectives = 4
    problem = DTLZ2(n_objectives, n_variables)
    problem.types = [Real(0, 1) for _ in range(n_variables)]
    return problem

#Definining the run function for EpsNSGA2
def run_EpsNSGA2(problem, config, nfe):
    print('Starting EpsNSGAII')
    algorithm = EpsNSGAII(problem,
                        epsilons=[config['epsilons']]*problem.nobjs,
                        population_size=config['population_size'],
                        selector=TournamentSelector(config['tournament_size']),
                        )
    algorithm.run(nfe)
    print('Finished EpsNSGAII run')
    return algorithm

#Defining the run function for Borg
def run_Borg(problem, config, nfe):
    print('Starting Borg')
    algorithm = BorgMOEA(problem,
                       epsilons=[config['epsilons']]*problem.nobjs,
                       population_size=config['population_size'],
                       selector=TournamentSelector(config['tournament_size']),
                       recency_list_size=config['recency_list_size'],
                       max_mutation_index=config['max_mutation_index'],
                       selection_ratio=config['selection_ratio'],
                       )
    algorithm.run(nfe)
    print('Finished Borg run')
    return algorithm

#Defining a function to generate LHS samples for the algorithm parameters
def generate_LHS_samples(param_ranges, n_samples):
    samples = {}

    for algorithm, params in param_ranges.items():
        sampler = LHSSampler()
        algorithm_samples = []

        #Storing sampled values for each parameter separately
        sampled_params = {}

        for param, value_range in params.items():
            if isinstance(value_range, tuple):  #Continuous parameter
                distribution = stats.uniform(value_range[0], value_range[1] - value_range[0])
            elif isinstance(value_range, list):  #Categorical parameter
                distribution = stats.randint(0, len(value_range))  #Integer indices for categories
            else:
                raise ValueError(f"Invalid range type for {param}: {value_range}")

            #Sampling separately for each parameter
            sampled_params[param] = sampler.sample(distribution, n_samples)

        #Putting sampled values into dictionaries
        for i in range(n_samples):
            sample_config = {}
            for param, values in sampled_params.items():
                value_range = params[param]

                if isinstance(value_range, tuple):  
                    #Rounding to int if original range was integer-based
                    if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                        sample_config[param] = int(round(values[i]))  #Ensuring int
                    else:
                        sample_config[param] = values[i]  #Keep as float

                elif isinstance(value_range, list):  
                    sample_config[param] = value_range[int(values[i])]  #Converting float index to int

            algorithm_samples.append(sample_config)

        samples[algorithm] = algorithm_samples

    return samples