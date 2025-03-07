from ema_workbench import Model, RealParameter, ScalarOutcome
from platypus import DTLZ2, DTLZ3, Solution

class DTLZ2Model(Model):
    def __init__(self, name, n_objectives, n_position_variables=10):
        """
        Initialize the DTLZ2 model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_objectives : int
            Number of objectives
        n_position_variables : int, optional
            Number of position-related variables, defaulted at 10
        """
        # Calculate total variables using the formula: n + k - 1
        n_variables = n_position_variables + n_objectives - 1
        
        super().__init__(name, function=self.dtlz2_function)
        
        # Store parameters
        self.n_objectives = n_objectives
        self.n_position_variables = n_position_variables
        self.n_variables = n_variables
        
        # Create the Platypus problem
        self.problem = DTLZ2(n_objectives, n_variables)
        
        # Define levers and outcomes
        self.levers = [RealParameter(f'x{i}', 0, 1) for i in range(n_variables)]
        self.outcomes = [ScalarOutcome(f'f{i}', ScalarOutcome.MINIMIZE) 
                         for i in range(n_objectives)]
    
    def dtlz2_function(self, **kwargs):
        """
        Run the DTLZ2 model with the given inputs
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary with model inputs
            
        Returns:
        --------
        dict
            Dictionary with model outputs
        """
        # Extract variables from kwargs
        variables = [kwargs[f'x{i}'] for i in range(self.n_variables)]
        
        # Create a Solution object
        solution = Solution(self.problem)
        solution.variables = variables
        
        # Evaluate the solution
        self.problem.evaluate(solution)
        
        # Return the objectives as a dictionary
        return {f'f{i}': solution.objectives[i] for i in range(self.n_objectives)}

def get_dtlz2_problem(n_objectives, n_position_variables=10):
    """
    Create a DTLZ2 problem with the correct number of decision variables
    
    Parameters:
    -----------
    n_objectives : int
        Number of objectives
    n_position_variables : int, optional
        Number of position-related variables, defaulted at 10
        
    Returns:
    --------
    model : DTLZ2Model
        EMA Workbench model for the DTLZ2 problem
    """
    return DTLZ2Model("DTLZ2", n_objectives, n_position_variables)

class DTLZ3Model(Model):
    def __init__(self, name, n_objectives, n_position_variables=10):
        """
        Initialize the DTLZ3 model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_objectives : int
            Number of objectives
        n_position_variables : int, optional
            Number of position-related variables, defaulted at 10
        """
        # Calculate total variables using the formula: n + k - 1
        n_variables = n_position_variables + n_objectives - 1
        
        super().__init__(name, function=self.dtlz3_function)
        
        # Store parameters
        self.n_objectives = n_objectives
        self.n_position_variables = n_position_variables
        self.n_variables = n_variables
        
        # Create the Platypus problem
        self.problem = DTLZ3(n_objectives, n_variables)
        
        # Define levers and outcomes
        self.levers = [RealParameter(f'x{i}', 0, 1) for i in range(n_variables)]
        self.outcomes = [ScalarOutcome(f'f{i}', ScalarOutcome.MINIMIZE) 
                         for i in range(n_objectives)]
    
    def dtlz3_function(self, **kwargs):
        """
        Run the DTLZ3 model with the given inputs
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary with model inputs
            
        Returns:
        --------
        dict
            Dictionary with model outputs
        """
        # Extract variables from kwargs
        variables = [kwargs[f'x{i}'] for i in range(self.n_variables)]
        
        # Create a Solution object
        solution = Solution(self.problem)
        solution.variables = variables
        
        # Evaluate the solution
        self.problem.evaluate(solution)
        
        # Return the objectives as a dictionary
        return {f'f{i}': solution.objectives[i] for i in range(self.n_objectives)}

def get_dtlz3_problem(n_objectives, n_position_variables=10):
    """
    Create a DTLZ3 problem with the correct number of decision variables
    
    Parameters:
    -----------
    n_objectives : int
        Number of objectives
    n_position_variables : int, optional
        Number of position-related variables, defaulted at 10
        
    Returns:
    --------
    model : DTLZ3Model
        EMA Workbench model for the DTLZ3 problem
    """
    return DTLZ3Model("DTLZ3", n_objectives, n_position_variables)

def get_JUSTICE_problem():
    pass
