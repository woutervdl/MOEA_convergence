from platypus.algorithms import EpsNSGAII
from platypus.core import nondominated_sort

class SteadyStateEpsNSGAII(EpsNSGAII):
    """Steady-state version of ε-NSGA-II processing one solution per iteration"""
    
    def iterate(self):
        #Generate one offspring using existing variator configuration
        parents = self.selector.select(self.variator.arity, self.population)
        offspring = self.variator.evolve(parents) 
        
        #Evaluate 
        self.evaluate_all(offspring)
        
        #Epsilon-dominance filtering via archive
        surviving_offspring = []
        for solution in offspring:
            if not self.archive.epsilon_contains(solution):
                self.archive.add(solution)
                surviving_offspring.append(solution)
        
        #Merge and truncate population
        if surviving_offspring:
            combined = self.population + surviving_offspring
            nondominated_sort(combined)
            
            #Use truncation from archive instead of standard NSGA-II truncation
            self.population = self.archive.truncate(combined, self.population_size)
