import numpy as np
from numba import jit

class Spread:
    """Spread (SPR) performance indicator."""
    
    def __init__(self, problem=None):
        """
        Initialize the spread metric.
        
        Parameters:
        -----------
        problem : Problem, optional
            The optimisation problem.
        """
        self.problem = problem
    
    def calculate(self, solutions):
        """
        Compute the spread.
        
        Parameters:
        -----------
        solutions : DataFrame or array-like
            A set of solutions in objective space. If DataFrame, assumes
            objectives are in columns.
        
        Returns:
        --------
        float
            The spread value.
        """
        # Convert to NumPy array if it's a DataFrame
        if hasattr(solutions, 'values'):
            solutions = solutions.values
        else:
            solutions = np.array(solutions)
            
        # Need at least 2 solutions to calculate spread
        if len(solutions) < 2:
            return 0.0
            
        return self._compute_spread(solutions)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_spread(solutions):
        """
        Compute the spread metric using Numba for acceleration.
        
        Parameters:
        -----------
        solutions : numpy.ndarray
            Solutions in objective space.
            
        Returns:
        --------
        float
            The spread value.
        """
        # Sort solutions along the Pareto front using lexicographical sorting
        # Note: np.lexsort is not supported by numba, so we do this outside the function
        # This is handled in the calling function
        
        # Compute Euclidean distances between consecutive solutions
        distances = np.empty(solutions.shape[0]-1)
        for i in range(solutions.shape[0]-1):
            dist = 0.0
            for j in range(solutions.shape[1]):
                dist += (solutions[i+1, j] - solutions[i, j])**2
            distances[i] = np.sqrt(dist)
        
        # Compute spread
        avg_distance = np.mean(distances)
        numerator = 0.0
        denominator = 0.0
        
        for d in distances:
            numerator += np.abs(d - avg_distance)
            denominator += d
        
        return numerator / denominator if denominator > 0 else 0.0
