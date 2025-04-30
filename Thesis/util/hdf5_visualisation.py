import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import h5py
import pandas as pd

# Default list of seeds used in the experiments
DEFAULT_SEEDS = [12345, 93489, 23403, 39349, 60930]
BASE_RESULTS_DIR = 'hpc/results'

def load_hdf5_data(problem_name, moeas, core_counts, group_name):
    """Fixed HDF5 loader matching your HPC file structure exactly"""
    data = []
    base_dir = os.path.join("hpc/results", problem_name)
    
    for cores in core_counts:
        cores_dir = os.path.join(base_dir, f"{cores}cores")
        if not os.path.exists(cores_dir):
            print(f"Missing cores directory: {cores_dir}")
            continue
            
        for moea in moeas:
            moea_dir = os.path.join(cores_dir, moea)
            if not os.path.exists(moea_dir):
                print(f"Missing algorithm directory: {moea_dir}")
                continue
                
            # Find all seed directories (seedXXXX format from run.sh)
            seed_dirs = [d for d in os.listdir(moea_dir) 
                        if d.startswith("seed") and os.path.isdir(os.path.join(moea_dir, d))]
            
            for seed_dir in seed_dirs:
                try:
                    # Extract numeric seed from directory name (seed12345 → 12345)
                    seed = int(seed_dir[4:])  # Remove "seed" prefix
                    hdf5_file = f"results_{problem_name}_{moea}_{cores}cores_seed{seed}.h5"
                    hdf5_path = os.path.join(moea_dir, seed_dir, hdf5_file)
                    
                    #print(f"Checking: {hdf5_path}")
                    
                    if not os.path.exists(hdf5_path):
                        print(f"Missing HDF5 file: {hdf5_path}")
                        continue
                        
                    with h5py.File(hdf5_path, 'r') as hf:
                        if group_name not in hf:
                            print(f"Missing group {group_name} in {hdf5_path}")
                            continue
                            
                        group = hf[group_name]
                        entry = {
                            'algorithm': moea,
                            'cores': cores,
                            'seed': seed,
                            **{k: group[k][()] for k in group.keys()}  # Load all datasets
                        }
                        data.append(entry)
                        #print(f"Loaded {len(group.keys())} datasets from {hdf5_path}")
                        
                except Exception as e:
                    print(f"Error processing {seed_dir}: {str(e)}")
    
    return data


def plot_metrics_by_cores(problem_name, moeas, core_counts, metric_names):
    """Create plots comparing metrics across MOEAs and core counts using HDF5."""
    sns.set_style("white")
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)

    for metric_name in metric_names:
        group_name = 'convergence' if metric_name == 'epsilon_progress' else 'metrics'
        data = load_hdf5_data(problem_name, moeas, core_counts, group_name)
        #print(f"Loaded {len(data)} entries for {metric_name}")
        
        if not data:
            print(f"No data found for {metric_name}! Skipping plot.")
            continue

        # Dynamically determine unique seeds for coloring
        all_seeds = list({entry['seed'] for entry in data})
        palette = sns.color_palette("husl", len(all_seeds))
        seed_to_color = {seed: palette[i] for i, seed in enumerate(sorted(all_seeds))}

        fig, axes = plt.subplots(nrows=len(core_counts), ncols=len(moeas),
                                figsize=(len(moeas)*5, len(core_counts)*4),
                                sharex='col', sharey='row')

        if len(core_counts) == 1:
            axes = np.array([axes])

        for row_idx, cores in enumerate(core_counts):
            for col_idx, moea in enumerate(moeas):
                ax = axes[row_idx, col_idx]
                
                filtered_data = [
                    entry for entry in data 
                    if entry['cores'] == cores 
                    and entry['algorithm'] == moea
                    and metric_name in entry
                    and 'nfe' in entry
                ]

                if not filtered_data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue

                for entry in filtered_data:
                    seed = entry['seed']
                    nfe = entry['nfe']
                    metric_vals = entry[metric_name]
                    sorted_idx = np.argsort(nfe)
                    nfe_sorted = nfe[sorted_idx]
                    metric_vals_sorted = metric_vals[sorted_idx]
                    points_to_skip = 0
                    if metric_name == 'time_efficiency':
                        points_to_skip = 10 # Skip first 10 points for time efficiency for better visualisation scale

                    ax.plot(nfe_sorted[points_to_skip:], metric_vals_sorted[points_to_skip:],
                            color=seed_to_color[seed],
                            label=f'Seed {seed}' if (col_idx == 0 and row_idx == 0) else "")

                if row_idx == 0:
                    ax.set_title(f'{moea}')
                if col_idx == 0:
                    ax.set_ylabel(f'{cores} cores\n{metric_name.replace("_", " ").capitalize()}')
                if row_idx == len(core_counts) - 1:
                    ax.set_xlabel('nfe')

        handles = [plt.Line2D([0], [0], color=seed_to_color[seed], lw=2) 
                 for seed in sorted(all_seeds)]
        labels = [f'Seed {seed}' for seed in sorted(all_seeds)]
        fig.legend(handles, labels, loc='lower center', ncol=min(5, len(all_seeds)))
        sns.despine(fig)
        plt.tight_layout()
        plt.savefig(f'./figures/{problem_name}/{metric_name}_by_cores.png', dpi=300)
        plt.show()
        plt.close()

def plot_runtime_comparison(problem_name, moeas, core_counts, seeds_list=None):
    """
    Create runtime comparison plot by reading the 'runtime' attribute from HDF5 files,
    mirroring the file iteration logic of load_hdf5_data.

    Args:
        problem_name (str): Name of the problem (e.g., 'JUSTICE').
        moeas (list): List of algorithm names.
        core_counts (list): List of core counts (e.g., [16, 32, 48]).
        seeds_list (list, optional): List of specific seed numbers used.
                                     Defaults to DEFAULT_SEEDS if None.
    """
    # Determine which seeds to iterate over
    seeds_to_iterate = seeds_list if seeds_list is not None else DEFAULT_SEEDS

    fig, ax = plt.subplots(figsize=(10, 6))
    # Ensure the base figures directory exists
    base_figure_dir = "./figures"
    os.makedirs(base_figure_dir, exist_ok=True)
    # Ensure the problem-specific directory exists
    output_dir = os.path.join(base_figure_dir, problem_name)
    os.makedirs(output_dir, exist_ok=True)

    runtime_entries = []

    # Iterate through directory structure
    problem_base_dir = os.path.join(BASE_RESULTS_DIR, problem_name)
    for cores in core_counts:
        cores_dir = os.path.join(problem_base_dir, f"{cores}cores") 
        for moea in moeas:
            moea_dir = os.path.join(cores_dir, moea)
            for seed in seeds_to_iterate:
                seed_dir_name = f"seed{seed}"
                seed_dir_path = os.path.join(moea_dir, seed_dir_name)
                # Construct the expected HDF5 filename
                h5_filename = f"results_{problem_name}_{moea}_{cores}cores_seed{seed}.h5"
                h5_filepath = os.path.join(seed_dir_path, h5_filename)

                if os.path.exists(h5_filepath):
                    with h5py.File(h5_filepath, 'r') as hf:
                        # Read the runtime attribute from the root level
                        runtime_seconds = hf.attrs.get("runtime")
                        if runtime_seconds is not None:
                            runtime_entries.append({
                                'algorithm': moea,
                                'cores': cores,
                                'seed': seed,
                                'runtime': float(runtime_seconds) # Runtime in seconds
                            })
    runtime_df = pd.DataFrame(runtime_entries)

    runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].agg(['mean', 'std']).reset_index()
    runtime_summary['std'] = runtime_summary['std'].fillna(0)

    for moea in moeas:
        moea_data = runtime_summary[runtime_summary['algorithm'] == moea]
        if not moea_data.empty:
            label = f'{moea} (Mean ± Std Dev)'
            ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'],
                        marker='o', linestyle='-', linewidth=2, capsize=5,
                        label=label)
        else:
            print(f"No summary data to plot for {moea}")

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title(f'{problem_name} Runtime Comparison vs Cores')
    ax.legend()
    ax.set_xticks(core_counts)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'runtime_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_speedup_comparison(problem_name, moeas, core_counts, seeds_list=None, show_individual_seeds=False):
    """
    Create speedup comparison plot by reading the 'runtime' attribute from HDF5 files.

    Speedup is calculated relative to the runtime on the minimum core count.

    Args:
        problem_name (str): Name of the problem.
        moeas (list): List of algorithm names.
        core_counts (list): List of core counts (must be sorted ascending).
        seeds_list (list, optional): List of specific seed numbers. Defaults to DEFAULT_SEEDS.
        show_individual_seeds (bool, optional): Plot individual seed speedups. Defaults to False.
    """
    seeds_to_iterate = seeds_list if seeds_list is not None else DEFAULT_SEEDS
    core_counts = sorted(core_counts)
    baseline_cores = core_counts[0] # Speedup relative to the lowest core count

    fig, ax = plt.subplots(figsize=(10, 6))
    base_figure_dir = "./figures"
    os.makedirs(base_figure_dir, exist_ok=True)
    output_dir = os.path.join(base_figure_dir, problem_name)
    os.makedirs(output_dir, exist_ok=True)

    runtime_entries = []

    problem_base_dir = os.path.join(BASE_RESULTS_DIR, problem_name)
    for cores in core_counts:
        cores_dir = os.path.join(problem_base_dir, f"{cores}cores")
        if not os.path.isdir(cores_dir): continue
        for moea in moeas:
            moea_dir = os.path.join(cores_dir, moea)
            if not os.path.isdir(moea_dir): continue
            for seed in seeds_to_iterate:
                seed_dir_name = f"seed{seed}"
                seed_dir_path = os.path.join(moea_dir, seed_dir_name)
                h5_filename = f"results_{problem_name}_{moea}_{cores}cores_seed{seed}.h5"
                h5_filepath = os.path.join(seed_dir_path, h5_filename)

                if os.path.exists(h5_filepath):
                    with h5py.File(h5_filepath, 'r') as hf:
                        runtime_seconds = hf.attrs.get("runtime")
                        if runtime_seconds is not None and float(runtime_seconds) > 0: 
                            runtime_entries.append({
                                'algorithm': moea, 'cores': cores, 'seed': seed,
                                'runtime': float(runtime_seconds)
                            })

    runtime_df = pd.DataFrame(runtime_entries)

    speedup_data = []
    mean_speedups = []

    # Calculate mean runtimes first
    runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].mean().reset_index()

    for moea in moeas:
        algo_runtime_summary = runtime_summary[runtime_summary['algorithm'] == moea]
        if algo_runtime_summary.empty: continue

        # Get mean baseline runtime (at minimum cores)
        baseline_summary_row = algo_runtime_summary[algo_runtime_summary['cores'] == baseline_cores]
        mean_baseline_time = baseline_summary_row['runtime'].iloc[0]

        # Calculate mean speedup
        for _, row in algo_runtime_summary.iterrows():
            mean_speedups.append({
                'algorithm': moea, 'cores': row['cores'],
                'speedup': mean_baseline_time / row['runtime']
            })

        # Calculate individual seed speedup if requested
        if show_individual_seeds:
            algo_runtime_all = runtime_df[runtime_df['algorithm'] == moea]
            for seed in seeds_to_iterate:
                seed_runtime_data = algo_runtime_all[algo_runtime_all['seed'] == seed]
                baseline_seed_row = seed_runtime_data[seed_runtime_data['cores'] == baseline_cores]
                seed_baseline_time = baseline_seed_row['runtime'].iloc[0]

                for _, row in seed_runtime_data.iterrows():
                    speedup_data.append({
                        'algorithm': moea, 'cores': row['cores'], 'seed': seed,
                        'speedup': seed_baseline_time / row['runtime']
                    })

    plotted_individual_labels = set()
    if show_individual_seeds and speedup_data:
        seed_df = pd.DataFrame(speedup_data)
        palette = sns.color_palette("husl", len(seeds_to_iterate))
        seed_to_color = {seed: palette[i] for i, seed in enumerate(sorted(seeds_to_iterate))}

        for moea in moeas:
             moea_seed_data = seed_df[seed_df['algorithm'] == moea]
             for seed, group in moea_seed_data.groupby('seed'):
                 # Create a unique label for legend only once per seed
                 label = f'Seed {seed}' if seed not in plotted_individual_labels else None
                 ax.scatter(group['cores'], group['speedup'],
                           alpha=0.4, marker='x', s=30, # Use 'x' markers for seeds
                           color=seed_to_color[seed],
                           label=label)
                 if label: plotted_individual_labels.add(seed) # Track plotted labels

    # Plot mean speedup if data exists
    if mean_speedups:
        mean_df = pd.DataFrame(mean_speedups)
        moea_palette = sns.color_palette("tab10", len(moeas)) # Different palette for means
        moea_to_color = {moea: moea_palette[i] for i, moea in enumerate(moeas)}

        for moea in moeas:
            moea_data = mean_df[mean_df['algorithm'] == moea].sort_values('cores') # Ensure sorted for line plot
            if not moea_data.empty:
                ax.plot(moea_data['cores'], moea_data['speedup'],
                        marker='o', linewidth=2, color=moea_to_color[moea],
                        label=f'{moea} (Mean Speedup)')

    # Plot Ideal speedup line
    max_cores = core_counts[-1]
    ax.plot([baseline_cores, max_cores], [1, max_cores / baseline_cores],
            'k--', linewidth=1.5, label='Ideal Speedup')

    ax.set_xlabel('Number of cores')
    ax.set_ylabel(f'Speedup (Relative to {baseline_cores} cores)')
    ax.set_title(f'{problem_name} Parallel Speedup Comparison')
    ax.legend(fontsize='small') # Adjust legend size if needed
    ax.set_xticks(core_counts)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'speedup_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_final_metric_comparison(problem_name, moeas, core_counts, metric_name):
    """
    Create a plot comparing final metric values across MOEAs and core counts using HDF5 data.
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    metric_name : str
        Name of the metric to plot
    """
    if metric_name == 'epsilon_progress':
        print("Skipping epsilon_progress as it's not in the metrics group.")
        return None  # Skip epsilon_progress as it's not in metrics files
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_metric_data = []
    
    # Load data from HDF5 files
    data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')
    
    for entry in data:
        if metric_name in entry and 'nfe' in entry:
            # Get the final value (highest nfe)
            nfe = entry['nfe']
            metric_values = entry[metric_name]
            if len(nfe) > 0 and len(metric_values) > 0:
                final_value = metric_values[-1]  # Last value corresponds to highest nfe
                final_metric_data.append({
                    'algorithm': entry['algorithm'],
                    'cores': entry['cores'],
                    'seed': entry['seed'],
                    'value': final_value
                })
    
    if not final_metric_data:
        print(f"No data found for metric '{metric_name}'.")
        return None

    final_df = pd.DataFrame(final_metric_data)
    
    # Calculate mean and std for each algorithm and core count
    summary = final_df.groupby(['algorithm', 'cores'])['value'].agg(['mean', 'std']).reset_index()
    
    # Plot final metric value vs cores for each algorithm
    for moea in moeas:
        moea_data = summary[summary['algorithm'] == moea]
        ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'], 
                    marker='o', capsize=5, label=f'{moea} (Mean ± Std Dev)')
    
    ax.set_xlabel('Number of cores')
    ax.set_ylabel(metric_name.replace('_', ' ').capitalize())
    ax.set_title(f'{problem_name} Final {metric_name.replace("_", " ").capitalize()} vs cores')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save and display the figure
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    save_path = f'./figures/{problem_name}/final_{metric_name}_comparison.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_hypervolume_over_time(problem_name, moeas, core_counts, seeds_list=None):
    """
    Plot hypervolume improvement over *estimated* wall-clock time using HDF5 data.
    Estimates time by linearly scaling NFE based on the total runtime of each run.

    Args:
        problem_name (str): Name of the problem.
        moeas (list): List of MOEAs to compare.
        core_counts (list): List of core counts used in experiments.
        seeds_list (list, optional): List of specific seed numbers. Defaults to DEFAULT_SEEDS.
    """
    # Determine seeds to iterate over
    seeds_to_iterate = seeds_list if seeds_list is not None else DEFAULT_SEEDS

    fig, axes = plt.subplots(1, len(core_counts), figsize=(5*len(core_counts), 6), sharey=True)
    axes = [axes] if len(core_counts) == 1 else axes # Ensure axes is always iterable

    # Ensure the output directories exist
    base_figure_dir = "./figures"
    os.makedirs(base_figure_dir, exist_ok=True)
    output_dir = os.path.join(base_figure_dir, problem_name)
    os.makedirs(output_dir, exist_ok=True)

    runtime_entries = []
    problem_base_dir = os.path.join(BASE_RESULTS_DIR, problem_name)

    for cores_rt in core_counts:
        cores_dir = os.path.join(problem_base_dir, f"{cores_rt}cores")
        if not os.path.isdir(cores_dir): continue
        for moea_rt in moeas:
            moea_dir = os.path.join(cores_dir, moea_rt)
            if not os.path.isdir(moea_dir): continue
            for seed_rt in seeds_to_iterate:
                seed_dir_name = f"seed{seed_rt}"
                seed_dir_path = os.path.join(moea_dir, seed_dir_name)
                h5_filename = f"results_{problem_name}_{moea_rt}_{cores_rt}cores_seed{seed_rt}.h5"
                h5_filepath = os.path.join(seed_dir_path, h5_filename)

                if os.path.exists(h5_filepath):
                    with h5py.File(h5_filepath, 'r') as hf:
                        runtime_seconds = hf.attrs.get("runtime")
                        if runtime_seconds is not None and float(runtime_seconds) > 0:
                            runtime_entries.append({
                                'algorithm': moea_rt, 'cores': cores_rt, 'seed': seed_rt,
                                'runtime': float(runtime_seconds)
                            })

    all_runtime_df = pd.DataFrame(runtime_entries)
    # Create a lookup dictionary for faster access: {(algo, cores, seed): runtime}
    runtime_lookup = {(r['algorithm'], r['cores'], r['seed']): r['runtime'] for _, r in all_runtime_df.iterrows()}

    metrics_data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')

    moea_palette = sns.color_palette("tab10", len(moeas))
    moea_to_color = {moea: moea_palette[i] for i, moea in enumerate(moeas)}

    for i, cores in enumerate(core_counts):
        ax = axes[i]
        ax_has_data = False # Flag to check if any data was plotted on this axis

        for moea in moeas:
            color = moea_to_color[moea]
            algo_core_metrics = [
                entry for entry in metrics_data
                if entry['algorithm'] == moea and entry['cores'] == cores
                and 'hypervolume' in entry and 'nfe' in entry
                and entry['hypervolume'] is not None and len(entry['hypervolume']) > 0
                and entry['nfe'] is not None and len(entry['nfe']) > 0
            ]

            if not algo_core_metrics: continue # Skip if no metric data for this combo

            all_seed_times = []
            all_seed_hvs = []
            max_time_for_mean = 0 # Track max estimated time for mean plot scaling

            # Plot individual seeds first (for alpha blending)
            for entry in algo_core_metrics:
                seed = entry['seed']
                nfe = entry['nfe']
                hv = entry['hypervolume']

                # Sort by NFE just in case
                sorted_idx = np.argsort(nfe)
                nfe_sorted = nfe[sorted_idx]
                hv_sorted = hv[sorted_idx]

                # Look up the specific runtime for this seed
                specific_runtime = runtime_lookup.get((moea, cores, seed))

                if specific_runtime and len(nfe_sorted) > 1:
                    max_nfe = nfe_sorted[-1]
                    if max_nfe > 0: # Avoid division by zero
                        # Estimate time axis by scaling NFE
                        # Time for NFE=0 is 0, Time for NFE=max_nfe is specific_runtime
                        estimated_times = (nfe_sorted / max_nfe) * specific_runtime
                        ax.plot(estimated_times, hv_sorted, alpha=0.2, color=color, linewidth=1)
                        ax_has_data = True

                        # Store for mean calculation (need common time points, interpolate later)
                        all_seed_times.append(estimated_times)
                        all_seed_hvs.append(hv_sorted)
                        max_time_for_mean = max(max_time_for_mean, estimated_times[-1])

            # Calculate and plot mean trend (more complex with varying time axes)
            if all_seed_times:
                # Create a common time axis for interpolation
                common_time = np.linspace(0, max_time_for_mean, num=200) # Adjust num for resolution
                interpolated_hvs = []
                for t_vals, h_vals in zip(all_seed_times, all_seed_hvs):
                     # Interpolate each seed's HV onto the common time axis
                     # Use bounds_error=False, fill_value=(h_vals[0], h_vals[-1]) to handle extrapolation
                     interpolated = np.interp(common_time, t_vals, h_vals, left=h_vals[0], right=h_vals[-1])
                     interpolated_hvs.append(interpolated)

                if interpolated_hvs:
                     # Calculate mean across seeds at each common time point
                     mean_hv_over_common_time = np.mean(interpolated_hvs, axis=0)
                     ax.plot(common_time, mean_hv_over_common_time,
                             linewidth=2.5, color=color, label=f'{moea} (Mean)')
                     ax_has_data = True

        if not ax_has_data:
            ax.text(0.5, 0.5, 'No Data or Runtimes', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'{cores} Cores')
        ax.set_xlabel('Estimated Wall-clock Time (s)')
        ax.grid(True, linestyle='--', alpha=0.7)
        # Ensure x-axis starts at or near 0
        ax.set_xlim(left=-0.02 * ax.get_xlim()[1]) # Allow slight negative margin if needed


    # Common Y Label and Legend
    axes[0].set_ylabel('Hypervolume')
    # Add legend to the first plot if it has data, otherwise try the next
    for ax_leg in axes:
        if ax_leg.has_data():
             handles, labels = ax_leg.get_legend_handles_labels()
             if handles: # Only add legend if there are lines plotted
                 fig.legend(handles, labels, loc='lower center', ncol=len(moeas), bbox_to_anchor=(0.5, -0.05))
                 break # Add only one legend for the figure

    plt.suptitle(f'{problem_name} Hypervolume Over Estimated Time', fontsize=16, y=1.02) # Adjust y for spacing
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect to make space for legend below

    save_path = os.path.join(output_dir, 'hypervolume_over_estimated_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)