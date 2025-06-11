# README
## Parallel Performance of Multi-Objective Evolutionary Algorithms in Climate-Economy Modelling:
**Exploring the Scalability and Convergence Properties of MOEAs for Climate-Economy Decision Support**

## Research Context

This repository contains code for running, analysing, and visualising large-scale experiments on Multi-Objective Evolutionary Algorithms (MOEAs) in parallel high-performance computing (HPC) environments. It performs benchmarking on standard problems (DTLZ2, DTLZ3) and optimises a custom integrated assessment model (JUSTICE), with a focus on convergence dynamics and parallel performance.

- **Algorithmic Performance**: Comparative analysis of ε-NSGA-II, Borg, and Generational Borg
- **Scalability**: Parallel performance across different core counts (16, 32, 48 cores)
- **Convergence Dynamics**: Analysis of solution quality evolution
- **Problem Complexity**: Performance across standard benchmarks (DTLZ2/3) and real-world climate-economic models (JUSTICE)

---

## Repository Structure

```
MOEA_CONVERGENCE/
├── JUSTICE_Fork/                                   # The forked JUSTCIE repository
│   └── .....                                       # All JUSTICE files
├── Thesis/                                         # Core thesis framework
   ├── algorithms/                                  # Custom MOEA implementations
   │   ├── borgMOEA.py                              # Borg implementation
   │   └── sse_nsga_ii.py                           # Steady-state ε-NSGA-II variant (not used)
   ├── archives/                                    # Experiment archive storage
   │   ├── DTLZ2/                                   # DTLZ2 optimisation histories
   │   ├── DTLZ3/                                   # DTLZ3 optimisation histories  
   │   └── JUSTICE/                                 # JUSTICE IAM optimisation histories
   ├── figures/                                     # Generated visualisation outputs
   │   ├── DTLZ2/                                   # DTLZ2 analysis plots
   │   ├── DTLZ3/                                   # DTLZ3 analysis plots
   │   └── JUSTICE/                                 # JUSTICE analysis plots
   ├── hdf5_results/                                # Final run state HDF5 storage
   │   ├── DTLZ2/                                   # DTLZ2 final populations & metrics
   │   ├── DTLZ3/                                   # DTLZ3 final populations & metrics
   │   └── JUSTICE/                                 # JUSTICE final populations & metrics
   ├── hpc/                                         # HPC execution management
   │   ├── hpc_slurm_scripts/                       # SLURM job submission scripts
   │   │    ├── short_test.sh                       # Short run script to test code
   │   │    ├── single_seed_run_final_exclusive.sh  # Script used to run most experiments
   │   │    ├── single_seed_run_final.sh            # Script to run experiments on exclusive node
   │   │    └── thesis.def                          # Apptainer script to build container for HPC execution
   │   ├── hpc_run.py                               # Main HPC experiment launcher
   │   └── run_single_experiment.py                 # Single experiment runner
   ├── results/                                     # Processed analysis results
   │   ├── DTLZ2/                                   # DTLZ2 convergence metrics
   │   ├── DTLZ3/                                   # DTLZ3 convergence metrics
   │   └── JUSTICE/                                 # JUSTICE convergence metrics
   ├── util/                                        # Research utilities and analysis tools
   │   ├── convergence_analysis.py                  # Metric computation 
   │   ├── csv_visualisation.py                     # CSV-based plotting (not used anymore)
   │   ├── global_ref_set_generator.py              # Global reference set creation
   │   ├── hdf5_visualisation.py                    # HDF5-based visualisation
   │   ├── model_definitions.py                     # Problem formulations
   │   ├── optimisation.py                          # Parallel MOEA execution core
   │   └── spread.py                                # Custom spread metric implementation (not used)
   ├── analysis.ipynb                               # Quick analysis notebook
   ├── global_JUSTICE_ref_set.csv                   # Generated global reference set for metric calculation
   ├── run_experiments.py                           # Batch experiment launcher used in initial local tests
   ├── temp_archive_opener.ipynb                    # Quick archive inspection
   ├──  temp_hdf5_reader.ipynb                      # Quick HDF5 file inspection
   └── requirements.txt                             # Python dependencies
```

---

## Core Research Components

### 1. Problem Definitions (`util/model_definitions.py`)

**DTLZ2 Benchmark Problem**
- Standard 4-objective test problem with known analytical Pareto front
- Configurable decision variables (default: 10 position + 4 objectives - 1 = 13 variables)
- `generate_true_pareto_solutions()` method for reference set generation
- Smooth, convex Pareto front for algorithm validation

**DTLZ3 Benchmark Problem**  
- Modified DTLZ2 with multiple local optima
- Same Pareto front as DTLZ2 but significantly more challenging search space
- Tests algorithm robustness against premature convergence
- Global optimum requires setting distance variables (x_M) to 0.5

**JUSTICE Integrated Assessment Model**
- Custom climate-economic model with 4 conflicting objectives:
  1. `welfare`: Total discounted welfare (minimise)
  2. `years_above_threshold`: Temperature threshold exceedance (minimise)  
  3. `welfare_loss_damage`: Climate damage costs (maximise)
  4. `welfare_loss_abatement`: Emission reduction costs (maximise)
- RBF-based policy parameterisation
- Complex, non-analytical Pareto. Reference Pareto front generated by non-dominated sorting of all obtained solutions.

### 2. HPC Experiment Management (`hpc/`)

**Main Experiment Launcher (`hpc_run.py`)**

**Problem Indices:**
- 0: DTLZ2
- 1: DTLZ3  
- 2: JUSTICE

**Algorithm Indices:**
- 0: eps_nsgaii (ε-NSGA-II)
- 1: borg (Borg)
- 2: generational_borg (Generational Borg)

**Single Experiment Runner (`run_single_experiment.py`)**
- Configures problem-specific settings and random seeds
- Manages HDF5 output structure: `/scratch/wmvanderlinden/MOEA_convergence/hdf5_results/`
- Handles temporary file management for HPC scratch storage
- Stores final populations, epsilon progress, and runtime metadata

**Slurm Scripts (`hpc/slurm_scripts/`)**
- `short_test.sh`: Bash script to run with the apptainer file for short code tests.
- `single_seed_run_final.sh`: Bash script to run batch experiments with the apptainer file. No exlusive node access.
- `single_seed_run_final_exclusive.sh`: Bash script to run batch experiments with the apptainer file. Runs on an exclusive node.

### 3. Optimisation (`util/optimisation.py`)

**Algorithm Configurations:**
```python

epsilons_dtlz = [0.05, 0.05, 0.05, 0.05]  # DTLZ problems
epsilons_justice = [0.01, 0.25, 10, 10]   # JUSTICE objectives

population_size = 100  # All algorithms

# Reference scenario for JUSTICE
reference = Scenario("reference", ssp_rcp_scenario=2)  # SSP245
```

### 4. Convergence Analysis System (`util/convergence_analysis.py`)

**Computed Metrics:**
- **Hypervolume**: Volume dominated by population relative to reference set
- **Generational Distance**: Average distance to true Pareto front
- **Epsilon Indicator**: Worst distance to true Pareto front
- **Spacing**: Distribution uniformity of solutions
- **NFE Efficiency**: Δ(Hypervolume)/Δ(NFE) - custom time-normalised metric

**Processing Workflow:**
1. **Archive Loading**: Reads `.tar.gz` archives from optimisation history
2. **Reference Set Integration**: Uses analytical fronts (DTLZ) or global empirical sets (JUSTICE)
3. **Objective Transformation**: Handles mixed minimise/maximise objectives for JUSTICE
4. **Parallel Processing**: Distributes metric calculation across worker processes
5. **HDF5 Storage**: Structured storage in `/results//cores//seed/`

### 5. Reference Set Generation (`util/global_ref_set_generator.py`)

**JUSTICE Global Reference Set:**
- Aggregates final populations from all successful runs (3 algorithms × 3 core counts × 5 seeds)
- Applies ε-nondominated sorting with JUSTICE-specific epsilons: `[0.01, 0.25, 10, 10]`
- Outputs `global_JUSTICE_ref_set.csv` for performance metric calculation

**Processing Steps:**
1. Iterate through all HDF5 result files in `hdf5_results/JUSTICE/`
2. Extract final archives from `final_archive` groups
3. Combine into unified DataFrame with all objectives and levers
4. Apply epsilon-nondominated filtering
5. Save as CSV for use as reference set

### 6. Visualisation System (`util/hdf5_visualisation.py`)

**Plot Generation Functions:**

**`plot_metrics_by_cores()`**: Convergence curve comparison
- Multi-panel layout: algorithms (columns) × core counts (rows)
- Consistent axis scaling across all subplots
- Seed-differentiated colouring for reproducibility analysis
- Supports all computed metrics with appropriate axis labels

**`plot_runtime_comparison()`**: Computational performance analysis  
- Mean runtime ± standard deviation across seeds
- Direct comparison of parallel efficiency
- Error bars indicate run-to-run variability

**`plot_speedup_comparison()`**: Parallel scaling analysis
- Speedup calculation relative to minimum core count (16 cores)
- Individual seed speedups (optional) + mean trends
- Ideal speedup reference line for comparison
- Identifies parallel efficiency bottlenecks

**`plot_final_metric_comparison()`**: Final solution quality
- Endpoint comparison across algorithms and core counts
- Statistical significance testing through error bars
- Direct assessment of convergence achievement

**`plot_hypervolume_over_time()`**: Time-normalised convergence
- Hypervolume improvement over estimated wall-clock time
- NFE-to-time conversion using total runtime scaling
- Multi-core performance visualisation
- Real-world time-to-solution analysis
