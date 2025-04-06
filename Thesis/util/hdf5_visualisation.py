import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import h5py
import pandas as pd

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
                    ax.plot(nfe[sorted_idx], metric_vals[sorted_idx],
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

def plot_runtime_comparison(problem_name, moeas, core_counts):
    """Create runtime comparison plot from HDF5 data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load runtime data from metrics group
    runtime_data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')
    
    # Build DataFrame with proper time efficiency handling
    runtime_entries = []
    for entry in runtime_data:
        if 'time_efficiency' in entry:
            # Use last value of time efficiency array as total runtime
            time_series = entry['time_efficiency']
            runtime = time_series[-1] if len(time_series) > 0 else np.nan
        else:
            runtime = np.nan
            
        runtime_entries.append({
            'algorithm': entry['algorithm'],
            'cores': entry['cores'],
            'seed': entry['seed'],
            'runtime': runtime
        })
    
    runtime_df = pd.DataFrame(runtime_entries)
    
    if runtime_df.empty:
        print("No runtime data found.")
        return None

    # Plot mean runtime with error bars
    runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].agg(['mean', 'std']).reset_index()
    
    for moea in moeas:
        moea_data = runtime_summary[runtime_summary['algorithm'] == moea]
        if not moea_data.empty:
            ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'],
                        marker='o', linewidth=2, label=f'{moea} (Mean)')
        else:
            print(f"No data for {moea}")

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title(f'{problem_name} Runtime comparison')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and display in notebook
    save_path = f'./figures/{problem_name}/runtime_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_speedup_comparison(problem_name, moeas, core_counts, show_individual_seeds=False):
    """
    Create speedup comparison plot from HDF5 data
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    show_individual_seeds : bool, optional
        Whether to show individual seed data points (default: False)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Load runtime data from HDF5 files
    runtime_entries = []
    data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')
    
    for entry in data:
        if 'time_efficiency' in entry:
            time_series = entry['time_efficiency']
            if len(time_series) > 0:
                runtime = time_series[-1]  # Last value as total runtime
                runtime_entries.append({
                    'algorithm': entry['algorithm'],
                    'cores': entry['cores'],
                    'seed': entry['seed'],
                    'runtime': runtime
                })

    if not runtime_entries:
        print("No runtime data found in HDF5 files")
        return None

    runtime_df = pd.DataFrame(runtime_entries)
    
    # Calculate baseline (minimum core count for each algorithm-seed combination)
    speedup_data = []
    baseline_cores = min(core_counts)
    
    # Individual seed speedup calculation
    if show_individual_seeds:
        for (moea, seed), group in runtime_df.groupby(['algorithm', 'seed']):
            baseline = group[group['cores'] == baseline_cores]['runtime']
            if not baseline.empty:
                baseline_time = baseline.values[0]
                for _, row in group.iterrows():
                    if row['runtime'] > 0:
                        speedup_data.append({
                            'algorithm': moea,
                            'cores': row['cores'],
                            'seed': seed,
                            'speedup': baseline_time / row['runtime']
                        })

    # Mean speedup calculation
    mean_speedups = []
    runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].mean().reset_index()
    for moea in moeas:
        moea_data = runtime_summary[runtime_summary['algorithm'] == moea]
        baseline = moea_data[moea_data['cores'] == baseline_cores]['runtime']
        if not baseline.empty:
            baseline_time = baseline.values[0]
            for _, row in moea_data.iterrows():
                if row['runtime'] > 0:
                    mean_speedups.append({
                        'algorithm': moea,
                        'cores': row['cores'],
                        'speedup': baseline_time / row['runtime']
                    })

    # Plotting
    if show_individual_seeds and speedup_data:
        seed_df = pd.DataFrame(speedup_data)
        for (moea, seed), group in seed_df.groupby(['algorithm', 'seed']):
            ax.scatter(group['cores'], group['speedup'], alpha=0.4, marker='o',
                      label=f'{moea} (Seed {seed})' if seed == 0 else None)

    if mean_speedups:
        mean_df = pd.DataFrame(mean_speedups)
        for moea in moeas:
            moea_data = mean_df[mean_df['algorithm'] == moea]
            ax.plot(moea_data['cores'], moea_data['speedup'], 
                   marker='o', linewidth=2, label=f'{moea} (Mean)')

    # Ideal speedup line
    max_cores = max(core_counts)
    ax.plot([baseline_cores, max_cores], [1, max_cores/baseline_cores], 
           'k--', label='Ideal Speedup')

    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Speedup')
    ax.set_title(f'{problem_name} Speedup Comparison')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save and display
    save_path = f'./figures/{problem_name}/speedup_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

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
                    marker='o', label=f'{moea} (Mean)')
    
    ax.set_xlabel('Number of cores')
    ax.set_ylabel(metric_name.replace('_', ' ').capitalize())
    ax.set_title(f'{problem_name} Final {metric_name.replace("_", " ").capitalize()} vs cores')
    ax.legend()
    
    # Use log scale for x-axis
    ax.set_xscale('log', base=2)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save and display the figure
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    save_path = f'./figures/{problem_name}/final_{metric_name}_comparison.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def load_runtime_data(problem_name, moeas, core_counts):
    """Load runtime data from HDF5 files."""
    runtime_entries = []
    
    # Load data for all MOEAs
    data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')
    
    for entry in data:
        try:
            if 'time_efficiency' in entry:
                time_series = entry['time_efficiency']
                if len(time_series) > 0:
                    runtime_entries.append({
                        'algorithm': entry['algorithm'],
                        'cores': entry['cores'],
                        'seed': entry['seed'],
                        'runtime': time_series[-1]  # Last value = total runtime
                    })
            else:
                print(f"Missing time_efficiency in {entry['algorithm']}/{entry['cores']}cores/seed{entry['seed']}")
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            continue
    
    df = pd.DataFrame(runtime_entries)
    
    return df


def load_metrics_data(problem_name, moeas, core_counts):
    """Load metrics data from HDF5 files into wide-format DataFrame."""
    metrics_data = []
    data = load_hdf5_data(problem_name, moeas, core_counts, 'metrics')
    
    for entry in data:
        if 'nfe' in entry and 'time_efficiency' in entry:
            # Create a DataFrame with metric columns
            df = pd.DataFrame({
                'nfe': entry['nfe'],
                'time_efficiency': entry['time_efficiency'],
                'hypervolume': entry.get('hypervolume', np.nan),
                'generational_distance': entry.get('generational_distance', np.nan),
                # Add other metrics as needed
                'algorithm': entry['algorithm'],
                'cores': entry['cores'],
                'seed': entry['seed']
            })
            metrics_data.append(df)
    
    return pd.concat(metrics_data) if metrics_data else pd.DataFrame()

def plot_hypervolume_efficiency(problem_name, moeas, core_counts):
    """
    Plot hypervolume improvement rate using HDF5 data.
    """
    sns.set_style("white")
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Create subplots for each core count
    fig, axes = plt.subplots(1, len(core_counts), figsize=(5 * len(core_counts), 6), sharey=True)
    if len(core_counts) == 1:
        axes = [axes]
    
    # Load metrics data from HDF5
    metrics_df = load_metrics_data(problem_name, moeas, core_counts)
    
    # Check if time_efficiency exists in the loaded data
    if metrics_df.empty or 'time_efficiency' not in metrics_df.columns:
        print("Missing or empty time_efficiency data.")
        return None
    
    # Iterate over core counts to create individual subplots
    for i, cores in enumerate(core_counts):
        ax = axes[i]
        
        # Filter data for this core count
        core_metrics = metrics_df[metrics_df['cores'] == cores]
        
        if core_metrics.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Iterate over MOEAs
        for moea_idx, moea in enumerate(moeas):
            moea_color = plt.cm.tab10(moea_idx)
            
            # Filter data for this MOEA
            moea_metrics = core_metrics[core_metrics['algorithm'] == moea]
            
            # Plot individual seed data
            for seed, seed_data in moea_metrics.groupby('seed'):
                if not seed_data.empty and 'nfe' in seed_data.columns and 'time_efficiency' in seed_data.columns:
                    seed_data = seed_data.sort_values('nfe')
                    ax.plot(seed_data['nfe'], seed_data['time_efficiency'], alpha=0.3, color=moea_color)
            
            # Plot mean time efficiency across seeds
            if not moea_metrics.empty and 'nfe' in moea_metrics.columns and 'time_efficiency' in moea_metrics.columns:
                mean_efficiency = moea_metrics.groupby('nfe')['time_efficiency'].mean().reset_index()
                ax.plot(mean_efficiency['nfe'], mean_efficiency['time_efficiency'], 
                        linewidth=2, label=f'{moea} (Mean)', color=moea_color)

        ax.set_title(f'{cores} Cores')
        ax.set_xlabel('nfe')
        if i == 0:
            ax.set_ylabel('Hypervolume Improvement Rate')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend()

    plt.suptitle(f'{problem_name} Time Efficiency (Hypervolume Improvement Rate)', fontsize=16)
    plt.tight_layout()
    
    save_path = f'./figures/{problem_name}/time_efficiency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_hypervolume_over_time(problem_name, moeas, core_counts):
    """
    Plot hypervolume improvement over wall-clock time using HDF5 data.
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    """
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Load data from HDF5 files
    metrics_df = load_metrics_data(problem_name, moeas, core_counts)
    runtime_df = load_runtime_data(problem_name, moeas, core_counts)
    
    if metrics_df.empty or runtime_df.empty:
        print("Missing data - check HDF5 files exist with required datasets")
        return None

    # Create figure
    fig, axes = plt.subplots(1, len(core_counts), figsize=(5*len(core_counts), 6), sharey=True)
    axes = [axes] if len(core_counts) == 1 else axes

    for i, cores in enumerate(core_counts):
        ax = axes[i]
        
        # Filter data for this core count
        core_metrics = metrics_df[(metrics_df['cores'] == cores) & 
                                (metrics_df['hypervolume'].notna())]
        core_runtime = runtime_df[runtime_df['cores'] == cores]

        if core_metrics.empty or core_runtime.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue

        # Plot each algorithm
        for moea_idx, moea in enumerate(moeas):
            color = plt.cm.tab10(moea_idx)
            
            # Get average runtime for this algorithm-core combination
            avg_runtime = core_runtime[core_runtime['algorithm'] == moea]['runtime'].mean()
            
            # Plot individual seeds
            moea_metrics = core_metrics[core_metrics['algorithm'] == moea]
            for seed, seed_data in moea_metrics.groupby('seed'):
                if not seed_data.empty:
                    seed_data = seed_data.sort_values('nfe')
                    if avg_runtime > 0 and not seed_data['nfe'].empty:
                        max_nfe = seed_data['nfe'].max()
                        times = seed_data['nfe'] * avg_runtime / max_nfe
                        ax.plot(times, seed_data['hypervolume'], 
                               alpha=0.3, color=color)

            # Plot mean trend
            if not moea_metrics.empty and avg_runtime > 0:
                mean_hv = moea_metrics.groupby('nfe')['hypervolume'].mean().reset_index()
                max_nfe = mean_hv['nfe'].max()
                times = mean_hv['nfe'] * avg_runtime / max_nfe
                ax.plot(times, mean_hv['hypervolume'], 
                       linewidth=2, color=color, label=f'{moea} (Mean)')

        ax.set_title(f'{cores} Cores')
        ax.set_xlabel('Estimated Wall-clock Time (s)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel('Hypervolume')
            ax.legend()

    plt.suptitle(f'{problem_name} Hypervolume Over Time', fontsize=16)
    plt.tight_layout()
    
    save_path = f'./figures/{problem_name}/hypervolume_over_time.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()