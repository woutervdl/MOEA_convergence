from ema_workbench import Model, RealParameter, ScalarOutcome, Constant
from platypus import DTLZ2, DTLZ3, Solution
from JUSTICE_fork.justice.util.enumerations import get_climate_scenario
from JUSTICE_fork.justice.util.EMA_model_wrapper import THESIS_model_wrapper_emodps
from JUSTICE_fork.justice.util.model_time import TimeHorizon
from JUSTICE_fork.justice.util.data_loader import DataLoader
import numpy as np
import pandas as pd

class DTLZ2Model(Model):
    def __init__(self, name, n_objectives, n_position_variables=10):
        """
        Initialise the DTLZ2 model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_objectives : int
            Number of objectives
        n_position_variables : int, optional
            Number of position-related variables, defaulted at 10
        """
        # Calculate total number of decision variables using the formula: n + k - 1
        n_variables = n_position_variables + n_objectives - 1
        
        # Initilalise the model with the DTLZ2 evaluation function
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
        variables = np.array([kwargs[f'x{i}'] for i in range(self.n_variables)])
        
        # Create a Solution object and set its variables
        solution = Solution(self.problem)
        solution.variables = variables
        
        # Evaluate the solution
        self.problem.evaluate(solution)
        
        # Return the objectives as a dictionary
        return {f'f{i}': solution.objectives[i] for i in range(self.n_objectives)}
    
    def generate_true_pareto_solutions(self, n_points=1000):
        """
        Generates decision variables AND corresponding objective values
        for points approximating the true Pareto front of DTLZ2.

        Variables x_M are fixed at 0.5. Variables x_I are sampled.
        Objectives are calculated directly.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns for all levers (x0..xn-1) and
            outcomes (f0..fM-1).
        """
        M = self.n_objectives
        n_vars = self.n_variables
        k = n_vars - (M - 1) # Number of x_M variables

        # 1. Generate x_I variables (x0 to x_{M-2}) uniformly in [0, 1]
        # These correspond to the angle parameters
        xI_values = np.random.rand(n_points, M - 1)

        # 2. Create x_M variables (x_{M-1} to x_{n-1}) fixed at 0.5
        xM_values = np.full((n_points, k), 0.5)

        # 3. Combine into full decision variable vectors
        x_values = np.hstack((xI_values, xM_values))

        # 4. Calculate corresponding objective values (using helper, g=0)
        f_values = calculate_dtlz_objectives(xI_values, M)

        # Set column names for levers and outcomes
        lever_names = [l.name for l in self.levers]
        outcome_names = [o.name for o in self.outcomes]

        # Create DataFrames for levers and outcomes
        df_levers = pd.DataFrame(x_values, columns=lever_names)
        df_outcomes = pd.DataFrame(f_values, columns=outcome_names)

        # Concatenate levers and outcomes into a single DataFrame
        pareto_solutions_df = pd.concat([df_levers, df_outcomes], axis=1)

        return pareto_solutions_df

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
        Initialise the DTLZ3 model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_objectives : int
            Number of objectives
        n_position_variables : int, optional
            Number of position-related variables, defaulted at 10
        """
        # Calculate total number of decision variables using the formula: n + k - 1
        n_variables = n_position_variables + n_objectives - 1
        
        # Initialise the model with the DTLZ3 evaluation function
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
        variables = np.array([kwargs[f'x{i}'] for i in range(self.n_variables)])
        
        # Create a Solution object
        solution = Solution(self.problem)
        solution.variables = variables
        
        # Evaluate the solution
        self.problem.evaluate(solution)
        
        # Return the objectives as a dictionary
        return {f'f{i}': solution.objectives[i] for i in range(self.n_objectives)}
    
    def generate_true_pareto_solutions(self, n_points=1000):
        """
        Generates decision variables AND corresponding objective values
        for points approximating the true global Pareto front of DTLZ3.

        Variables x_M are fixed at 0.5 (minimises g). Variables x_I are sampled.
        Objectives are calculated directly (assuming g=0).

        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns for all levers (x0..xn-1) and
            outcomes (f0..fM-1).
        """
        # The variable settings for the global front are the same as DTLZ2
        M = self.n_objectives
        n_vars = self.n_variables
        k = n_vars - (M - 1) # Number of x_M variables

        xI_values = np.random.rand(n_points, M - 1)
        xM_values = np.full((n_points, k), 0.5)
        x_values = np.hstack((xI_values, xM_values))
        f_values = calculate_dtlz_objectives(xI_values, M)

        lever_names = [l.name for l in self.levers]
        outcome_names = [o.name for o in self.outcomes]
        df_levers = pd.DataFrame(x_values, columns=lever_names)
        df_outcomes = pd.DataFrame(f_values, columns=outcome_names)
        pareto_solutions_df = pd.concat([df_levers, df_outcomes], axis=1)

        return pareto_solutions_df

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

def calculate_dtlz_objectives(xI_values, M):
    """
    Calculates DTLZ objectives from x_I variables assuming g=0.
    Used for generating Pareto front samples for DTLZ2/3.

    Parameters:
    -----------
    xI_values : np.ndarray
        Array of shape (n_points, M-1) with sampled x_I variables
    M : int
        Number of objectives

    Returns:
    --------
    f : np.ndarray
        Array of shape (n_points, M) with calculated objective values
    """
    n_points = xI_values.shape[0]
    f = np.ones((n_points, M))
    # Convert x_I (in [0,1]) to angles theta (in [0, pi/2])
    thetas = xI_values * (np.pi / 2.0)

    # Loop over objectives to calculate their values
    for m in range(M): # Objective index m=0..M-1
        idx = M - 1 - m # Corresponding theta/xI index
        if m == 0: # Last objective f_M
            f[:, m] = np.sin(thetas[:, 0])
        elif m == M - 1: # First objective f_1
            f[:, m] = np.prod(np.cos(thetas), axis=1)
        else: # Intermediate objectives f_2..f_{M-1}
            # Product of cosines up to theta_{idx-1} * sin(theta_{idx})
            f[:, m] = np.prod(np.cos(thetas[:, idx + 1:]), axis=1) * np.sin(thetas[:, idx])

    # Need to reverse the order of calculated objectives to match f1, f2, ... fM
    return f[:, ::-1]

class JUSTICEModel(Model):
    def __init__(self, name="JUSTICE", n_regions=None, n_timesteps=None):
        """
        Initialise the JUSTICE model
        
        Parameters:
        -----------
        name : str
            Name of the model
        n_regions : int, optional
            Number of regions in the model (if None, will use DataLoader.REGION_LIST)
        n_timesteps : int, optional
            Number of timesteps to simulate (if None, will calculate from TimeHorizon)
        """
        # Initialise configuration parameters
        self.configure_model_parameters()
        
        # Initialise data loader and time horizon
        self.data_loader = DataLoader()
        self.time_horizon = TimeHorizon(
            start_year=self.start_year,
            end_year=self.end_year,
            data_timestep=self.data_timestep,
            timestep=self.timestep
        )
        
        # Set max ensemble size
        self.max_ensemble_size = 40

        # Set regions and timesteps
        self.n_regions = n_regions if n_regions is not None else len(self.data_loader.REGION_LIST)
        self.n_timesteps = n_timesteps if n_timesteps is not None else len(self.time_horizon.model_time_horizon)
        
        # Calculate emission control start timestep
        self.emission_control_start_timestep = self.time_horizon.year_to_timestep(
            year=self.emission_control_start_year, 
            timestep=self.timestep
        )

        # Calculate the index for the year of interest for temperature
        self.temperature_year_of_interest_index = self.time_horizon.year_to_timestep(
            year=self.temperature_year_of_interest,
            timestep=self.timestep
        )
        
        # Initialise the model with the justice function
        super().__init__(name, function=self.justice_function)
        
        # Define model components
        self.define_levers()
        self.define_reference_scenario()
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
        self.temperature_year_of_interest = 2100
        
        # RBF parameters
        self.n_rbfs = 4
        self.n_inputs_rbf = 2
        
        # Scenario and model type parameters
        self.reference_scenario_index = 2 # SSP245
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
        
        # Initialise lever lists
        centers_levers = []
        radii_levers = []
        weights_levers = []
        
        # Create center and radii parameters
        for i in range(centers_shape):
            centers_levers.append(RealParameter(f"center_{i}", -1.0, 1.0))
            radii_levers.append(RealParameter(f"radii_{i}", 1e-4, 1.0))
        
        # Create weight parameters
        for i in range(weights_shape):
            weights_levers.append(RealParameter(f"weights_{i}", 1e-4, 1.0))
        
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
            Constant("social_welfare_function_type", self.welfare_function_type),
            Constant("economy_type", self.economy_type),
            Constant("damage_function_type", self.damage_function_type),
            Constant("abatement_type", self.abatement_type),
            Constant("temperature_year_of_interest_index", self.temperature_year_of_interest_index),
            Constant("max_ensemble_size", self.max_ensemble_size)
        ]    

    def define_reference_scenario(self):
        """Define the reference scenario"""
        self.scenario_string = get_climate_scenario(self.reference_scenario_index)

    def define_outcomes(self):
        """Define model outcomes"""
        self.outcomes = [
            ScalarOutcome(
                "welfare",
                variable_name="welfare",
                kind=ScalarOutcome.MINIMIZE
            ),
            ScalarOutcome(
                "years_above_threshold", 
                variable_name="years_above_threshold",
                kind=ScalarOutcome.MINIMIZE
            ),#,
            # ScalarOutcome(
            #     "fraction_above_threshold",
            #     variable_name="fraction_above_threshold",
            #     kind=ScalarOutcome.MINIMIZE,
            # ),
            ScalarOutcome(
                "welfare_loss_damage",
                variable_name="welfare_loss_damage",
                kind=ScalarOutcome.MAXIMIZE,
            ),
            ScalarOutcome(
                "welfare_loss_abatement",
                variable_name="welfare_loss_abatement",
                kind=ScalarOutcome.MAXIMIZE,
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
        # Adding the scenario index to the kwargs, passing it as constant didn't work
        kwargs["ssp_rcp_scenario"] = self.reference_scenario_index
        # Calling the model wrapper with all inputs
        welfare, years_above_threshold, welfare_loss_damage, welfare_loss_abatement = THESIS_model_wrapper_emodps(**kwargs)
        
        return {
            'welfare': welfare,
            'years_above_threshold': years_above_threshold,
            #'fraction_above_threshold': fraction_above_threshold,
            'welfare_loss_damage': welfare_loss_damage,
            'welfare_loss_abatement': welfare_loss_abatement
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