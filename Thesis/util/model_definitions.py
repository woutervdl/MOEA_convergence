from ema_workbench import Model, RealParameter, ScalarOutcome, CategoricalParameter, Constant
from platypus import DTLZ2, DTLZ3, Solution
from JUSTICE_fork.src.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction, Scenario
from JUSTICE_fork.solvers.emodps.rbf import RBF
from JUSTICE_fork.src.util.EMA_model_wrapper import model_wrapper_emodps as original_wrapper

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

# class JUSTICEModel(Model):
#     def __init__(self, name="JUSTICE", n_regions=57, n_timesteps=40):
#         """
#         Initialise the JUSTICE model
#         """
#         super().__init__(name, function=self.justice_function)
        
#         # Store parameters
#         self.n_regions = n_regions
#         self.n_timesteps = n_timesteps
        
#         # RBF parameters
#         self.n_inputs_rbf = 2
#         self.n_outputs_rbf = n_regions
        
#         # Create RBF to determine shape of parameters
#         rbf = RBF(
#             n_rbfs=(self.n_inputs_rbf + 2), 
#             n_inputs=self.n_inputs_rbf, 
#             n_outputs=self.n_outputs_rbf
#         )
#         centers_shape, radii_shape, weights_shape = rbf.get_shape()
        
#         # Define levers (RBF parameters)
#         levers_list = []
        
#         # Add centers
#         for i in range(centers_shape[0]):
#             levers_list.append(RealParameter(f"center {i}", -1, 1))
        
#         # Add radii
#         for i in range(radii_shape[0]):
#             levers_list.append(RealParameter(f"radii {i}", 0.1, 1))
        
#         # Add weights
#         for i in range(weights_shape[0]):
#             levers_list.append(RealParameter(f"weights {i}", 0, 1))
        
#         self.levers = levers_list
        
#         # Define constants (using integers directly)
#         self.constants = [
#             Constant("n_regions", n_regions),
#             Constant("n_timesteps", n_timesteps),
#             Constant("emission_control_start_timestep", 0),
#             Constant("n_inputs_rbf", self.n_inputs_rbf),
#             Constant("n_outputs_rbf", self.n_outputs_rbf),
#             Constant("ssp_rcp_scenario", 2),
#             Constant("social_welfare_function_type", 0),
#             Constant("economy_type", 0),
#             Constant("damage_function_type", 1),
#             Constant("abatement_type", 0)
#         ]
        
#         # Define outcomes
#         self.outcomes = [
#             ScalarOutcome('welfare', ScalarOutcome.MAXIMIZE),
#             ScalarOutcome('years_above_threshold', ScalarOutcome.MINIMIZE)
#         ]
    
#     def justice_function(self, **kwargs):
#         """
#         Custom wrapper to ensure parameters are properly passed to model_wrapper_emodps
#         """
#         # Create a modified copy of kwargs
#         modified_kwargs = kwargs.copy()
        
#         # Create direct instances of the required enums
#         # This bypasses the from_index method entirely
#         welfare_enum = WelfareFunction.UTILITARIAN
#         economy_enum = Economy.NEOCLASSICAL
#         damage_enum = DamageFunction.KALKUHL
#         abatement_enum = Abatement.ENERDATA
        
#         # Replace the parameters with the actual enum instances
#         modified_kwargs["social_welfare_function_type"] = welfare_enum
#         modified_kwargs["economy_type"] = economy_enum
#         modified_kwargs["damage_function_type"] = damage_enum
#         modified_kwargs["abatement_type"] = abatement_enum
        
#         # Call the model wrapper with our properly formatted parameters
#         welfare, years_above_threshold = model_wrapper_emodps(**modified_kwargs)
        
#         return {
#             'welfare': welfare,
#             'years_above_threshold': years_above_threshold
#         }

def fixed_model_wrapper_emodps(**kwargs):
    """
    A fixed version of model_wrapper_emodps that ensures proper enum handling
    """
    # Create a modified copy of kwargs
    modified_kwargs = kwargs.copy()
    
    # Directly set the enum values
    modified_kwargs["social_welfare_function_type"] = WelfareFunction.UTILITARIAN
    modified_kwargs["economy_type"] = Economy.NEOCLASSICAL
    modified_kwargs["damage_function_type"] = DamageFunction.KALKUHL
    modified_kwargs["abatement_type"] = Abatement.ENERDATA
    
    # Call the original wrapper with our fixed parameters
    return original_wrapper(**modified_kwargs)

class JUSTICEModel(Model):
    def __init__(self, name="JUSTICE", n_regions=57, n_timesteps=40):
        """
        Initialise the JUSTICE model
        """
        super().__init__(name, function=self.justice_function)
        
        # Store parameters
        self.n_regions = n_regions
        self.n_timesteps = n_timesteps
        
        # RBF parameters
        self.n_inputs_rbf = 2
        self.n_outputs_rbf = n_regions
        
        # Create RBF to determine shape of parameters
        rbf = RBF(
            n_rbfs=(self.n_inputs_rbf + 2), 
            n_inputs=self.n_inputs_rbf, 
            n_outputs=self.n_outputs_rbf
        )
        centers_shape, radii_shape, weights_shape = rbf.get_shape()
        
        # Define levers (RBF parameters)
        levers_list = []
        
        # Add centers
        for i in range(centers_shape[0]):
            levers_list.append(RealParameter(f"center {i}", -1, 1))
        
        # Add radii
        for i in range(radii_shape[0]):
            levers_list.append(RealParameter(f"radii {i}", 0.1, 1))
        
        # Add weights
        for i in range(weights_shape[0]):
            levers_list.append(RealParameter(f"weights {i}", 0, 1))
        
        self.levers = levers_list
        
        # Define constants - using integers directly
        self.constants = [
            Constant("n_regions", n_regions),
            Constant("n_timesteps", n_timesteps),
            Constant("emission_control_start_timestep", 0),
            Constant("n_inputs_rbf", self.n_inputs_rbf),
            Constant("n_outputs_rbf", self.n_outputs_rbf),
            Constant("ssp_rcp_scenario", 2)
        ]
        
        # Define outcomes
        self.outcomes = [
            ScalarOutcome('welfare', ScalarOutcome.MAXIMIZE),
            ScalarOutcome('years_above_threshold', ScalarOutcome.MINIMIZE)
        ]
    
    def justice_function(self, **kwargs):
        """
        Use fixed wrapper instead of the original
        """
        welfare, years_above_threshold = fixed_model_wrapper_emodps(**kwargs)
        return {
            'welfare': welfare,
            'years_above_threshold': years_above_threshold
        }


def get_justice_model(n_regions=57, n_timesteps=40):
    """
    Create a JUSTICE model with the specified parameters
    
    Parameters:
    -----------
    n_regions : int
        Number of regions in the model
    n_timesteps : int
        Number of timesteps to simulate
        
    Returns:
    --------
    model : JUSTICEModel
        EMA Workbench model for the JUSTICE problem
    """
    return JUSTICEModel("JUSTICE", n_regions, n_timesteps)