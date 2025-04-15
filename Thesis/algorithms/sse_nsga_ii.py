from platypus import AbstractGeneticAlgorithm, EpsilonBoxArchive, nondominated_sort, nondominated_truncate, AdaptiveTimeContinuation, default_variator
from platypus.operators import RandomGenerator, TournamentSelector, UM

class SteadyStateEpsNSGAII_Core(AbstractGeneticAlgorithm):
    """
    Core steady-state algorithm inspired by NSGA-II using EpsilonBoxArchive.
    Designed to be wrapped by AdaptiveTimeContinuation.
    """

    def __init__(self,
                 problem,
                 epsilons, 
                 population_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 variator=None,
                 **kwargs):
        """ Initialises the steady-state core algorithm. """
        super().__init__(problem, population_size, generator, **kwargs)
        self.selector = selector
        self.variator = variator
        self.archive = EpsilonBoxArchive(epsilons)
        # The result should reflect the archive's content
        self.result = self.archive

    def initialize(self):
        """ Initialises population, evaluates, sets default variator, populates archive. """
        super().initialize() # Creates and evaluates initial self.population

        # Set default variator if none provided
        if self.variator is None:
            self.variator = default_variator(self.problem)

        # Populate the archive with the initial population
        self.archive.extend(self.population)
        # Ensure self.result points to the archive list object
        self.result = self.archive

    def iterate(self):
        """ Performs one steady-state iteration. """
        # 1. Select parents
        parents = self.selector.select(self.variator.arity, self.population)

        # 2. Create offspring (typically 1 or 2 from variator)
        offspring_list = self.variator.evolve(parents)

        # 3. Evaluate new offspring
        self.evaluate_all(offspring_list) # Updates self.nfe

        # 4. Integrate offspring and select next population (N+k -> N)
        for offspring in offspring_list:
            # Combine current population and the new offspring
            pool = self.population + [offspring]

            # Apply standard NSGA-II survival selection: rank + diversity
            nondominated_sort(pool)
            # Truncate back to population size using rank/diversity
            self.population = nondominated_truncate(pool, self.population_size)

            # 5. Update the Epsilon Archive with the offspring
            #    The archive handles epsilon-dominance internally.
            self.archive.add(offspring)

        # 6. Ensure result points to the potentially updated archive
        #    (already points to the list object, content may have changed)
        self.result = self.archive

    def step(self):
        """ Defines one step of the algorithm for the runner. """
        if self.nfe == 0:
            self.initialize()
        else:
            self.iterate()
        # Result is always the archive, which is updated in initialize/iterate

class SteadyStateEpsNSGAII(AdaptiveTimeContinuation):
    """
    This class provides the same interface and adaptive restart behavior
    as the original EpsNSGAII, but uses the steady-state core logic.
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
                 mutator = UM(1.0), # Assuming UM is available
                 **kwargs):

        # Instantiate the STEADY-STATE core algorithm
        core_algorithm = SteadyStateEpsNSGAII_Core(problem,
                                                   epsilons,
                                                   population_size,
                                                   generator,
                                                   selector,
                                                   variator,
                                                   **kwargs)

        # Initialise the AdaptiveTimeContinuation wrapper with the steady-state core
        super().__init__(algorithm=core_algorithm,
                         window_size=window_size,
                         max_window_size=max_window_size,
                         population_ratio=population_ratio,
                         min_population_size=min_population_size,
                         max_population_size=max_population_size,
                         mutator=mutator)

