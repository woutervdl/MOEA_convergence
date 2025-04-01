import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_metrics_by_cores(problem_name, moeas, core_counts, seeds, metric_names):
    """
    Create plots comparing metrics across MOEAs and core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    metric_names : list
        List of metrics to plot
    """
    sns.set_style("white")
    
    # Create directory for figures if it doesn't exist
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Plot metrics over nfe for each MOEA and core count
    for metric_name in metric_names:
        # Create a figure with subplots for each MOEA and core count
        fig, axes = plt.subplots(nrows=len(core_counts), ncols=len(moeas), 
                                figsize=(len(moeas)*5, len(core_counts)*4), 
                                sharex='col', sharey='row')
        
        # If only one core count, make axes 2D
        if len(core_counts) == 1:
            axes = np.array([axes])
        
        # Iterate over each core count
        for row_idx, cores in enumerate(core_counts):
            # Iterate over each MOEA
            for col_idx, moea in enumerate(moeas):
                ax = axes[row_idx, col_idx]
                
                # Create a color palette with enough colors for each seed
                palette = sns.color_palette("husl", seeds)
                
                # Plot each seed's data
                for seed in range(seeds):
                    if metric_name == 'epsilon_progress':
                        # Load from convergence file
                        convergence_file = os.path.join("./results", problem_name, f"{cores}_cores", 
                                                      moea, f"seed{seed}_convergence.csv")
                        if os.path.exists(convergence_file):
                            convergence_df = pd.read_csv(convergence_file)
                            if 'epsilon_progress' in convergence_df.columns and 'nfe' in convergence_df.columns:
                                convergence_df = convergence_df.sort_values('nfe')
                                ax.plot(convergence_df['nfe'], convergence_df['epsilon_progress'], 
                                        color=palette[seed], label=f'Seed {seed+1}' if col_idx == 0 and row_idx == 0 else "")
                    else:
                        # Load from metrics file
                        metrics_file = os.path.join("./results", problem_name, f"{cores}_cores", 
                                                  moea, f"seed{seed}_metrics.csv")
                        if os.path.exists(metrics_file):
                            metrics_df = pd.read_csv(metrics_file)
                            if metric_name in metrics_df.columns:
                                metrics_df = metrics_df.sort_values('nfe')
                                ax.plot(metrics_df['nfe'], metrics_df[metric_name], 
                                        color=palette[seed], label=f'Seed {seed+1}' if col_idx == 0 and row_idx == 0 else "")
                
                # Set title for the first row
                if row_idx == 0:
                    ax.set_title(f'{moea}')
                
                # Set y label for the first column
                if col_idx == 0:
                    ax.set_ylabel(f'{cores} cores\n{metric_name.replace("_", " ").capitalize()}')
                
                # Set x label for the last row
                if row_idx == len(core_counts) - 1:
                    ax.set_xlabel('nfe')
        
        # Add a legend to the figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        
        # Despine the figure
        sns.despine(fig)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'./figures/{problem_name}/{metric_name}_by_cores.png', dpi=300)
        #plt.close()
    
    return fig

def plot_runtime_comparison(problem_name, moeas, core_counts, seeds, show_individual_seeds=False):
    """
    Create a plot comparing runtime across MOEAs and core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    show_individual_seeds : bool, optional
        Whether to show individual seed data points (default: False)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load and process runtime data
    runtime_data = []
    for cores in core_counts:
        runtime_file = os.path.join("./results", problem_name, f"runtimes_{cores}_cores.csv")
        if os.path.exists(runtime_file):
            df = pd.read_csv(runtime_file)
            runtime_data.append(df)
    
    if runtime_data:
        runtime_df = pd.concat(runtime_data)
        
        if show_individual_seeds:
            # Plot individual seed data points with transparency
            for moea in moeas:
                moea_data = runtime_df[runtime_df['algorithm'] == moea]
                for seed in range(seeds):
                    seed_data = moea_data[moea_data['seed'] == seed]
                    ax.scatter(seed_data['cores'], seed_data['runtime'], 
                              alpha=0.4, marker='o', 
                              label=f'{moea} (Seed {seed})' if seed == 0 else None)
        
        # Calculate mean and std of runtime for each algorithm and core count
        runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].agg(['mean', 'std']).reset_index()
        
        # Plot mean runtime with error bars
        for moea in moeas:
            moea_data = runtime_summary[runtime_summary['algorithm'] == moea]
            ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'], 
                        marker='o', linewidth=2, label=f'{moea} (Mean)')
        
        ax.set_xlabel('Number of cores')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title(f'{problem_name} Runtime comparison')
        ax.legend()
        
        # Use log scale for both axes
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        
        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'./figures/{problem_name}/runtime_comparison.png', dpi=300)
        
    return fig

def plot_speedup_comparison(problem_name, moeas, core_counts, seeds, show_individual_seeds=False):
    """
    Create a plot comparing speedup across MOEAs and core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    show_individual_seeds : bool, optional
        Whether to show individual seed data points (default: False)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load and process runtime data
    runtime_data = []
    for cores in core_counts:
        runtime_file = os.path.join("./results", problem_name, f"runtimes_{cores}_cores.csv")
        if os.path.exists(runtime_file):
            df = pd.read_csv(runtime_file)
            runtime_data.append(df)
    
    if runtime_data:
        runtime_df = pd.concat(runtime_data)
        
        if show_individual_seeds:
            # Calculate speedup for each seed
            speedup_data_by_seed = []
            for moea in moeas:
                for seed in range(seeds):
                    seed_data = runtime_df[(runtime_df['algorithm'] == moea) & 
                                          (runtime_df['seed'] == seed)]
                    
                    if len(seed_data) > 0:
                        baseline = seed_data[seed_data['cores'] == min(core_counts)]['runtime'].values[0]
                        
                        for _, row in seed_data.iterrows():
                            speedup_data_by_seed.append({
                                'algorithm': row['algorithm'],
                                'cores': row['cores'],
                                'seed': row['seed'],
                                'speedup': baseline / row['runtime']
                            })
            
            # Plot individual seed speedup with transparency
            if speedup_data_by_seed:
                seed_df = pd.DataFrame(speedup_data_by_seed)
                for moea in moeas:
                    for seed in range(seeds):
                        data = seed_df[(seed_df['algorithm'] == moea) & (seed_df['seed'] == seed)]
                        if not data.empty:
                            ax.scatter(data['cores'], data['speedup'], 
                                      alpha=0.4, marker='o', 
                                      label=f'{moea} (Seed {seed})' if seed == 0 else None)
        
        # Calculate mean runtime for each algorithm and core count
        runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].agg(['mean']).reset_index()
        
        # Calculate speedup relative to single core
        speedup_data = []
        for moea in moeas:
            baseline = runtime_summary[(runtime_summary['algorithm'] == moea) & 
                                      (runtime_summary['cores'] == min(core_counts))]['mean'].values[0]
            
            for _, row in runtime_summary[runtime_summary['algorithm'] == moea].iterrows():
                speedup_data.append({
                    'algorithm': row['algorithm'],
                    'cores': row['cores'],
                    'speedup': baseline / row['mean']
                })
        
        speedup_df = pd.DataFrame(speedup_data)
        
        # Plot mean speedup
        for moea in moeas:
            moea_data = speedup_df[speedup_df['algorithm'] == moea]
            ax.plot(moea_data['cores'], moea_data['speedup'], marker='o', linewidth=2,
                   label=f'{moea} (Mean)')
        
        # # Add ideal speedup line
        # max_cores = max(core_counts)
        # ax.plot([1, max_cores], [1, max_cores], 'k--', label='Ideal Speedup')
        
        ax.set_xlabel('Number of cores')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{problem_name} Speedup comparison')
        ax.legend()
        
        # Use log scale for x-axis
        ax.set_xscale('log', base=2)
        
        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'./figures/{problem_name}/speedup_comparison.png', dpi=300)
    
    return fig

def plot_final_metric_comparison(problem_name, moeas, core_counts, seeds, metric_name):
    """
    Create a plot comparing final metric values across MOEAs and core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    metric_name : str
        Name of the metric to plot
    """
    if metric_name == 'epsilon_progress':
        return None  # Skip epsilon_progress as it's not in metrics files
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_metric_data = []
    
    for cores in core_counts:
        for moea in moeas:
            for seed in range(seeds):
                metrics_file = os.path.join("./results", problem_name, f"{cores}_cores", 
                                          moea, f"seed{seed}_metrics.csv")
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file)
                    if metric_name in metrics_df.columns:
                        # Get the final value (highest nfe)
                        metrics_df = metrics_df.sort_values('nfe')
                        final_value = metrics_df.iloc[-1][metric_name]
                        
                        final_metric_data.append({
                            'algorithm': moea,
                            'cores': cores,
                            'seed': seed,
                            'value': final_value
                        })
    
    if final_metric_data:
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
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'./figures/{problem_name}/final_{metric_name}_comparison.png', dpi=300)
    
    return fig

def load_runtime_data(problem_name, core_counts):
    """
    Load runtime data for a problem across all core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    core_counts : list
        List of core counts used in experiments
        
    Returns:
    --------
    DataFrame
        Combined runtime data
    """
    runtime_data = []
    for cores in core_counts:
        runtime_file = os.path.join("./results", problem_name, f"runtimes_{cores}_cores.csv")
        if os.path.exists(runtime_file):
            df = pd.read_csv(runtime_file)
            runtime_data.append(df)
    
    if runtime_data:
        return pd.concat(runtime_data)
    else:
        return pd.DataFrame()

def load_metrics_data(problem_name, moeas, core_counts, seeds):
    """
    Load metrics data for a problem across all MOEAs, core counts, and seeds
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
        
    Returns:
    --------
    DataFrame
        Combined metrics data
    """
    metrics_data = []
    
    for cores in core_counts:
        for moea in moeas:
            for seed in range(seeds):
                metrics_file = os.path.join("./results", problem_name, f"{cores}_cores", 
                                          moea, f"seed{seed}_metrics.csv")
                if os.path.exists(metrics_file):
                    df = pd.read_csv(metrics_file)
                    df['algorithm'] = moea
                    df['cores'] = cores
                    df['seed'] = seed
                    metrics_data.append(df)
    
    if metrics_data:
        return pd.concat(metrics_data)
    else:
        return pd.DataFrame()

def load_convergence_data(problem_name, moeas, core_counts, seeds):
    """
    Load convergence data for a problem across all MOEAs, core counts, and seeds
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
        
    Returns:
    --------
    DataFrame
        Combined convergence data
    """
    convergence_data = []
    
    for cores in core_counts:
        for moea in moeas:
            for seed in range(seeds):
                convergence_file = os.path.join("./results", problem_name, f"{cores}_cores", 
                                              moea, f"seed{seed}_convergence.csv")
                if os.path.exists(convergence_file):
                    df = pd.read_csv(convergence_file)
                    df['algorithm'] = moea
                    df['cores'] = cores
                    df['seed'] = seed
                    convergence_data.append(df)
    
    if convergence_data:
        return pd.concat(convergence_data)
    else:
        return pd.DataFrame()
    
def plot_hypervolume_efficiency(problem_name, moeas, core_counts, seeds):
    """
    Create plots comparing hypervolume improvement rate (time efficiency) across MOEAs and core counts
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    """
    sns.set_style("white")
    
    # Create directory for figures if it doesn't exist
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Create a figure with subplots for each core count
    fig, axes = plt.subplots(1, len(core_counts), figsize=(5*len(core_counts), 6), sharey=True)
    
    # If only one core count, make axes iterable
    if len(core_counts) == 1:
        axes = [axes]
    
    # For each core count
    for i, cores in enumerate(core_counts):
        ax = axes[i]
        
        # Filter data for this core count
        metrics_data = load_metrics_data(problem_name, moeas, [cores], seeds)
        
        if not metrics_data.empty:
            # For each algorithm
            for moea_idx, moea in enumerate(moeas):
                moea_color = plt.cm.tab10(moea_idx)
                
                # For each seed
                for seed in range(seeds):
                    # Get data for this algorithm and seed
                    seed_data = metrics_data[(metrics_data['algorithm'] == moea) & 
                                            (metrics_data['seed'] == seed)]
                    
                    if not seed_data.empty and 'time_efficiency' in seed_data.columns:
                        # Sort by nfe
                        seed_data = seed_data.sort_values('nfe')
                        
                        # Plot time efficiency
                        ax.plot(seed_data['nfe'], seed_data['time_efficiency'], 
                               alpha=0.3, color=moea_color)
                
                # Calculate and plot mean time efficiency
                moea_data = metrics_data[metrics_data['algorithm'] == moea]
                if not moea_data.empty and 'time_efficiency' in moea_data.columns:
                    # Group by nfe and calculate mean
                    mean_efficiency = moea_data.groupby('nfe')['time_efficiency'].mean().reset_index()
                    
                    # Plot mean with bold line
                    ax.plot(mean_efficiency['nfe'], mean_efficiency['time_efficiency'], 
                           linewidth=2, label=f'{moea} (Mean)', color=moea_color)
        
        ax.set_title(f'{cores} Cores')
        ax.set_xlabel('nfe')
        
        if i == 0:
            ax.set_ylabel('Hypervolume Improvement Rate')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot to avoid clutter
        if i == 0:
            ax.legend()
    
    plt.suptitle(f'{problem_name} Time efficiency (Hypervolume Improvement Rate)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./figures/{problem_name}/time_efficiency.png', dpi=300)
    
    return fig

def plot_hypervolume_over_time(problem_name, moeas, core_counts, seeds):
    """
    Plot hypervolume improvement over wall-clock time
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem
    moeas : list
        List of MOEAs to compare
    core_counts : list
        List of core counts used in experiments
    seeds : int
        Number of seeds used
    """
    # Create directory for figures if it doesn't exist
    os.makedirs(f"./figures/{problem_name}", exist_ok=True)
    
    # Load metrics and runtime data
    metrics_data = load_metrics_data(problem_name, moeas, core_counts, seeds)
    runtime_data = load_runtime_data(problem_name, core_counts)
    
    if metrics_data.empty or runtime_data.empty:
        return None
    
    # Create figure
    fig, axes = plt.subplots(1, len(core_counts), figsize=(5*len(core_counts), 6), sharey=True)
    
    # If only one core count, make axes iterable
    if len(core_counts) == 1:
        axes = [axes]
    
    # For each core count
    for i, cores in enumerate(core_counts):
        ax = axes[i]
        
        # Filter data for this core count
        core_metrics = metrics_data[metrics_data['cores'] == cores]
        core_runtime = runtime_data[runtime_data['cores'] == cores]
        
        # For each algorithm
        for moea_idx, moea in enumerate(moeas):
            # Get average runtime for this algorithm
            avg_runtime = core_runtime[core_runtime['algorithm'] == moea]['runtime'].mean()
            
            # For each seed
            for seed in range(seeds):
                # Get data for this algorithm and seed
                seed_data = core_metrics[(core_metrics['algorithm'] == moea) & 
                                        (core_metrics['seed'] == seed)]
                
                if not seed_data.empty and 'hypervolume' in seed_data.columns:
                    # Sort by nfe
                    seed_data = seed_data.sort_values('nfe')
                    
                    # Calculate approximate time for each nfe point
                    # Assuming linear relationship between nfe and time
                    max_nfe = seed_data['nfe'].max()
                    times = seed_data['nfe'] * avg_runtime / max_nfe
                    
                    # Plot hypervolume over time
                    ax.plot(times, seed_data['hypervolume'], alpha=0.3, 
                           color=plt.cm.tab10(moea_idx))
            
            # Plot average trend
            avg_data = core_metrics[core_metrics['algorithm'] == moea].groupby('nfe')['hypervolume'].mean()
            if not avg_data.empty:
                nfes = avg_data.index.values
                times = nfes * avg_runtime / max(nfes)
                ax.plot(times, avg_data.values, linewidth=2, 
                       label=f'{moea} (Mean)', color=plt.cm.tab10(moea_idx))
        
        ax.set_title(f'{cores} Cores')
        ax.set_xlabel('Estimated Wall-clock Time (s)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel('Hypervolume')
            ax.legend()
    
    plt.suptitle(f'{problem_name} Hypervolume over time', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./figures/{problem_name}/hypervolume_over_time.png', dpi=300)
    
    return fig