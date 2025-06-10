
from platypus import (EpsilonBoxArchive, nondominated_sort,
                      nondominated_truncate, AdaptiveTimeContinuation,
                      NSGAII, RandomGenerator, TournamentSelector, UM)

class SteadyStateNSGAII_for_EpsilonArchive(NSGAII):
    """
    A steady-state version of NSGAII's core logic, designed to be used
    with an EpsilonBoxArchive. It overrides the generational iteration
    with a steady-state one.
    """
    def __init__(self,
                 problem,
                 epsilons, 
                 population_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 variator=None,
                 **kwargs):
        # Initialise NSGAII but pass an EpsilonBoxArchive as its archive
        super().__init__(problem,
                         population_size=population_size,
                         generator=generator,
                         selector=selector,
                         variator=variator,
                         archive=EpsilonBoxArchive(epsilons), # Use EpsilonBoxArchive
                         **kwargs)

    def iterate(self):
        """
        Performs one steady-state iteration:
        1. Select parents
        2. Create offspring
        3. Evaluate new offspring.
        4. Integrate offspring into the population using NSGA-II's selection (rank & diversity)
        5. Add the evaluated offspring to the EpsilonBoxArchive
        """
        # 1. Select parents
        # If variator.arity is not defined or different from selector needs, adjust.
        # For typical genetic operators (SBX, PM), arity is 2.
        num_parents = self.variator.arity if hasattr(self.variator, 'arity') else 2
        parents = self.selector.select(num_parents, self.population)

        # 2. Create offspring
        offspring_list = self.variator.evolve(parents)

        # 3. Evaluate new offspring
        self.evaluate_all(offspring_list) # Updates self.nfe

        # 4. Integrate offspring and select next population (N + k -> N)
        # For each new offspring, add it to the pool and re-select
        # This maintains population size and uses NSGA-II's survival criteria
        current_population = list(self.population) # Make a mutable copy
        for offspring_individual in offspring_list:
            # Combine current population and the new offspring
            pool = current_population + [offspring_individual]

            # Apply standard NSGA-II survival selection: rank + diversity
            nondominated_sort(pool)
            # Truncate back to population size using rank/diversity
            current_population = nondominated_truncate(pool, self.population_size)
        
        self.population = current_population # Update the main population

        # 5. Update the Epsilon Archive with the (evaluated) offspring
        # self.archive is the EpsilonBoxArchive instance.
        # Adding offspring_list ensures new candidates are considered by the archive
        self.archive.extend(offspring_list)

# The wrapper class remains very similar, just using the new core
class SteadyStateEpsNSGAII(AdaptiveTimeContinuation):
    """
    This class provides the same interface and adaptive restart behavior
    as the original EpsNSGAII, but uses the steady-state core logic
    derived from NSGAII.
    """
    def __init__(self,
                 problem,
                 epsilons,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None,
                 # Parameters for AdaptiveTimeContinuation
                 window_size = 100,
                 max_window_size = 1000,
                 population_ratio = 4.0,
                 min_population_size = 10,
                 max_population_size = 10000,
                 mutator = None, # Default mutator (UM(1.0)) will be set by AdaptiveTimeContinuation if None
                 **kwargs):

        # Instantiate the NEW STEADY-STATE core algorithm
        core_algorithm = SteadyStateNSGAII_for_EpsilonArchive(
            problem,
            epsilons,
            population_size=population_size,
            generator=generator,
            selector=selector,
            variator=variator,
            **kwargs)

        # Set default mutator for AdaptiveTimeContinuation if none is provided
        if mutator is None:
            mutator = UM(probability=1.0/problem.nvars if problem.nvars > 0 else 0.1) # Example default

        # Initialise the AdaptiveTimeContinuation wrapper with the steady-state core
        super().__init__(algorithm=core_algorithm,
                         window_size=window_size,
                         max_window_size=max_window_size,
                         population_ratio=population_ratio,
                         min_population_size=min_population_size,
                         max_population_size=max_population_size,
                         mutator=mutator)