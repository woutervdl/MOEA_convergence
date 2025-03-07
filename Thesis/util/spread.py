import numpy as np

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
            
        # Sort solutions along the Pareto front using lexicographical sorting
        sorted_solutions = solutions[np.lexsort(np.fliplr(solutions).T)]
        
        # Compute Euclidean distances between consecutive solutions
        distances = np.linalg.norm(np.diff(sorted_solutions, axis=0), axis=1)
        
        # Compute spread
        avg_distance = np.mean(distances)
        numerator = np.sum(np.abs(distances - avg_distance))
        denominator = np.sum(distances)
        
        return numerator / denominator if denominator > 0 else 0.0