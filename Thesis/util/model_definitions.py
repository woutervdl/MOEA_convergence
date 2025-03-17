from ema_workbench import Model, RealParameter, ScalarOutcome, CategoricalParameter, Constant
from platypus import DTLZ2, DTLZ3, Solution
from JUSTICE_fork.src.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction, Scenario
from JUSTICE_fork.solvers.emodps.rbf import RBF
from JUSTICE_fork.src.util.EMA_model_wrapper import model_wrapper_emodps
from JUSTICE_fork.src.util.model_time import TimeHorizon
from JUSTICE_fork.src.util.data_loader import DataLoader

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
#         Initialize the JUSTICE model
#         """
#         super().__init__(name, function=self.justice_function)
        
#         # Store parameters
#         self.n_regions = n_regions
#         self.n_timesteps = n_timesteps
        
#         # RBF parameters - matching the working code
#         n_rbfs = 4
#         self.n_inputs_rbf = 2
#         self.n_outputs_rbf = n_regions
        
#         # Calculate centers and weights shape like the working code
#         centers_shape = n_rbfs * self.n_inputs_rbf
#         weights_shape = n_regions * n_rbfs
        
#         # Define levers (RBF parameters)
#         centers_levers = []
#         radii_levers = []
#         weights_levers = []
        
#         # Add centers and radii with the same ranges as working code
#         for i in range(centers_shape):
#             centers_levers.append(RealParameter(f"center {i}", -1.0, 1.0))
#             radii_levers.append(RealParameter(f"radii {i}", 0.0, 1.0))
        
#         # Add weights with the same ranges as working code
#         for i in range(weights_shape):
#             weights_levers.append(RealParameter(f"weights {i}", 0.0, 1.0))
        
#         # Set the levers attribute directly
#         self.levers = centers_levers + radii_levers + weights_levers
        
#         # Calculate emission control start timestep like the working code
#         # Assuming start_year=2015, timestep=1, emission_control_start_year=2025
#         emission_control_start_timestep = 10  # (2025-2015)/1
        
#         # Define constants - matching the working code
#         self.constants = [
#             Constant("n_regions", n_regions),
#             Constant("n_timesteps", n_timesteps),
#             Constant("emission_control_start_timestep", emission_control_start_timestep),
#             Constant("n_rbfs", n_rbfs),  # Added this from working code
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
#         Run the JUSTICE model with the given inputs
#         """
#         # Simply pass the kwargs directly to model_wrapper_emodps
#         welfare, years_above_threshold = model_wrapper_emodps(**kwargs)
        
#         return {
#             'welfare': welfare,
#             'years_above_threshold': years_above_threshold
#         }


# def get_justice_model(n_regions=57, n_timesteps=40):
#     """
#     Create a JUSTICE model with the specified parameters
    
#     Parameters:
#     -----------
#     n_regions : int
#         Number of regions in the model
#     n_timesteps : int
#         Number of timesteps to simulate
        
#     Returns:
#     --------
#     model : JUSTICEModel
#         EMA Workbench model for the JUSTICE problem
#     """
#     return JUSTICEModel("JUSTICE", n_regions, n_timesteps)

class JUSTICEModel(Model):
    def __init__(self, name="JUSTICE", n_regions=None, n_timesteps=None):
        """
        Initialize the JUSTICE model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_regions : int, optional
            Number of regions in the model (if None, will use DataLoader.REGION_LIST)
        n_timesteps : int, optional
            Number of timesteps to simulate (if None, will calculate from TimeHorizon)
        """
        # Initialize configuration parameters
        self.configure_model_parameters()
        
        # Initialize data loader and time horizon
        self.data_loader = DataLoader()
        self.time_horizon = TimeHorizon(
            start_year=self.start_year,
            end_year=self.end_year,
            data_timestep=self.data_timestep,
            timestep=self.timestep
        )
        
        # Set regions and timesteps
        self.n_regions = n_regions if n_regions is not None else len(self.data_loader.REGION_LIST)
        self.n_timesteps = n_timesteps if n_timesteps is not None else len(self.time_horizon.model_time_horizon)
        
        # Calculate emission control start timestep
        self.emission_control_start_timestep = self.time_horizon.year_to_timestep(
            year=self.emission_control_start_year, 
            timestep=self.timestep
        )
        
        # Initialize the model with the justice function
        super().__init__(name, function=self.justice_function)
        
        # Define model components
        self.define_levers()
        self.define_constants()
        self.define_outcomes()
    
    def configure_model_parameters(self):
        """Configure model parameters"""
        # Time parameters
        self.start_year = 2015
        self.end_year = 2300
        self.data_timestep = 5
        self.timestep = 1
        self.emission_control_start_year = 2025
        
        # RBF parameters
        self.n_rbfs = 4
        self.n_inputs_rbf = 2
        
        # Scenario and model type parameters
        self.scenario_index = 2  # SSP245
        self.welfare_function_type = 0  # UTILITARIAN
        self.economy_type = 0  # NEOCLASSICAL
        self.damage_function_type = 1  # KALKUHL
        self.abatement_type = 0  # ENERDATA
    
    def define_levers(self):
        """Define model levers (RBF parameters)"""
        # Calculate shapes for RBF parameters
        self.n_outputs_rbf = self.n_regions
        centers_shape = self.n_rbfs * self.n_inputs_rbf
        weights_shape = self.n_regions * self.n_rbfs
        
        # Initialize lever lists
        centers_levers = []
        radii_levers = []
        weights_levers = []
        
        # Create center and radii parameters
        for i in range(centers_shape):
            centers_levers.append(RealParameter(f"center_{i}", -1.0, 1.0))
            radii_levers.append(RealParameter(f"radii_{i}", 0.0, 1.0))
        
        # Create weight parameters
        for i in range(weights_shape):
            weights_levers.append(RealParameter(f"weights_{i}", 0.0, 1.0))
        
        # Set the levers attribute
        self.levers = centers_levers + radii_levers + weights_levers
    
    def define_constants(self):
        """Define model constants"""
        self.constants = [
            Constant("n_regions", self.n_regions),
            Constant("n_timesteps", self.n_timesteps),
            Constant("emission_control_start_timestep", self.emission_control_start_timestep),
            Constant("n_rbfs", self.n_rbfs),
            Constant("n_inputs_rbf", self.n_inputs_rbf),
            Constant("n_outputs_rbf", self.n_outputs_rbf),
            Constant("ssp_rcp_scenario", self.scenario_index),
            Constant("social_welfare_function_type", self.welfare_function_type),
            Constant("economy_type", self.economy_type),
            Constant("damage_function_type", self.damage_function_type),
            Constant("abatement_type", self.abatement_type)
        ]
    
    def define_outcomes(self):
        """Define model outcomes"""
        self.outcomes = [
            ScalarOutcome(
                "welfare",
                variable_name="welfare",
                kind=ScalarOutcome.MAXIMIZE
            ),
            ScalarOutcome(
                "years_above_threshold", 
                variable_name="years_above_threshold",
                kind=ScalarOutcome.MINIMIZE
            )
        ]
    
    def justice_function(self, **kwargs):
        """
        Run the JUSTICE model with the given inputs
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary with model inputs
            
        Returns:
        --------
        dict
            Dictionary with model outputs
        """
        # Simply pass the kwargs directly to model_wrapper_emodps
        welfare, years_above_threshold = model_wrapper_emodps(**kwargs)
        
        return {
            'welfare': welfare,
            'years_above_threshold': years_above_threshold
        }

def get_justice_model(n_regions=None, n_timesteps=None):
    """
    Create a JUSTICE model with the specified parameters
    
    Parameters:
    -----------
    n_regions : int, optional
        Number of regions in the model (if None, will use DataLoader.REGION_LIST)
    n_timesteps : int, optional
        Number of timesteps to simulate (if None, will calculate from TimeHorizon)
        
    Returns:
    --------
    model : JUSTICEModel
        EMA Workbench model for the JUSTICE problem
    """
    return JUSTICEModel("JUSTICE", n_regions, n_timesteps)