import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import h5py

def plot_metrics_by_cores(problem_names, moeas, core_counts, num_seeds, metric_names):
    """Plot metrics by cores for multiple seeds."""
    for problem_name in problem_names:  # Iterate through problem names
        sns.set_style("white")
        os.makedirs(f"./figures/{problem_name}", exist_ok=True)

        for metric_name in metric_names:
            fig, axes = plt.subplots(
                nrows=len(core_counts), 
                ncols=len(moeas),
                figsize=(len(moeas)*5, len(core_counts)*4),
                sharex='col', sharey='row'
            )
            
            if len(core_counts) == 1:
                axes = np.array([axes])

            for row_idx, cores in enumerate(core_counts):
                for col_idx, moea in enumerate(moeas):
                    ax = axes[row_idx, col_idx]
                    palette = sns.color_palette("husl", num_seeds)

                    for seed_idx in range(num_seeds):
                        # Ensure all path components are strings
                        file_path = os.path.join(
                            "./results",
                            str(problem_name),  # Explicit string conversion
                            f"{cores}_cores",
                            str(moea),  # Explicit string conversion
                            f"results_{problem_name}_{moea}_{cores}cores_seed_{seed_idx}.h5"
                        )
                        
                        if os.path.exists(file_path):
                            with h5py.File(file_path, "r") as hf:
                                group = "convergence" if metric_name == "epsilon_progress" else "metrics"
                                
                                if group in hf and metric_name in hf[group]:
                                    nfe = hf[group]["nfe"][:]
                                    values = hf[group][metric_name][:]
                                    ax.plot(nfe, values, color=palette[seed_idx], 
                                           label=f'Seed {seed_idx+1}' if col_idx == 0 and row_idx == 0 else "")

                    if row_idx == 0:
                        ax.set_title(f'{moea}')
                    if col_idx == 0:
                        ax.set_ylabel(f'{cores} cores\n{metric_name.replace("_", " ").capitalize()}')
                    if row_idx == len(core_counts) - 1:
                        ax.set_xlabel('nfe')

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99))
            sns.despine(fig)
            plt.tight_layout()
            plt.savefig(f'./figures/{problem_name}/{metric_name}_by_cores.png', dpi=300)
    
    return fig


def plot_runtime_comparison(problem_name, moeas, core_counts, num_seeds, show_individual_seeds=False):
    """Plot runtime comparison across cores and algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    runtime_data = []

    for cores in core_counts:
        for moea in moeas:
            for seed_idx in range(num_seeds):
                file_path = os.path.join(
                    "./results", problem_name, f"{cores}_cores", moea,
                    f"results_{problem_name}_{moea}_{cores}cores_seed_{seed_idx}.h5"
                )
                
                if os.path.exists(file_path):
                    with h5py.File(file_path, "r") as hf:
                        runtime_data.append({
                            "algorithm": moea,
                            "cores": cores,
                            "seed": seed_idx,
                            "runtime": hf.attrs["runtime"]
                        })

    if runtime_data:
        runtime_df = pd.DataFrame(runtime_data)
        
        if show_individual_seeds:
            for moea in moeas:
                moea_data = runtime_df[runtime_df['algorithm'] == moea]
                ax.scatter(moea_data['cores'], moea_data['runtime'], alpha=0.4)

        runtime_summary = runtime_df.groupby(['algorithm', 'cores'])['runtime'].agg(['mean', 'std']).reset_index()
        
        for moea in moeas:
            moea_data = runtime_summary[runtime_summary['algorithm'] == moea]
            ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'],
                       marker='o', linewidth=2, label=moea)

    ax.set(xlabel='Number of cores', ylabel='Runtime (seconds)', 
           title=f'{problem_name} Runtime comparison',
           xscale='log', yscale='log', base=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/{problem_name}/runtime_comparison.png', dpi=300)
    return fig

def load_metrics_data(problem_name, moeas, core_counts, num_seeds):
    """Load metrics data from HDF5 files."""
    metrics_data = []

    for cores in core_counts:
        for moea in moeas:
            for seed_idx in range(num_seeds):
                file_path = os.path.join(
                    "./results", problem_name, f"{cores}_cores", moea,
                    f"results_{problem_name}_{moea}_{cores}cores_seed_{seed_idx}.h5"
                )
                
                if os.path.exists(file_path):
                    with h5py.File(file_path, "r") as hf:
                        if "metrics" in hf:
                            metrics = {col: hf["metrics"][col][:] for col in hf["metrics"]}
                            metrics.update({
                                "algorithm": [moea] * len(metrics["nfe"]),
                                "cores": [cores] * len(metrics["nfe"]),
                                "seed": [seed_idx] * len(metrics["nfe"])
                            })
                            metrics_data.append(pd.DataFrame(metrics))

    return pd.concat(metrics_data) if metrics_data else pd.DataFrame()

def plot_final_metric_comparison(problem_name, moeas, core_counts, num_seeds, metric_name):
    """Plot final metric comparison across cores and algorithms."""
    if metric_name == 'epsilon_progress':
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    final_metric_data = []

    for cores in core_counts:
        for moea in moeas:
            for seed_idx in range(num_seeds):
                file_path = os.path.join(
                    "./results", problem_name, f"{cores}_cores", moea,
                    f"results_{problem_name}_{moea}_{cores}cores_seed_{seed_idx}.h5"
                )
                
                if os.path.exists(file_path):
                    with h5py.File(file_path, "r") as hf:
                        if "metrics" in hf and metric_name in hf["metrics"]:
                            metrics = pd.DataFrame({col: hf["metrics"][col][:] for col in hf["metrics"]})
                            final_value = metrics.sort_values("nfe").iloc[-1][metric_name]
                            final_metric_data.append({
                                "algorithm": moea,
                                "cores": cores,
                                "seed": seed_idx,
                                "value": final_value
                            })

    if final_metric_data:
        final_df = pd.DataFrame(final_metric_data)
        summary = final_df.groupby(['algorithm', 'cores'])['value'].agg(['mean', 'std']).reset_index()
        
        for moea in moeas:
            moea_data = summary[summary['algorithm'] == moea]
            ax.errorbar(moea_data['cores'], moea_data['mean'], yerr=moea_data['std'],
                       marker='o', label=moea)

    ax.set(xlabel='Number of cores', ylabel=metric_name.replace('_', ' ').capitalize(),
           title=f'{problem_name} Final {metric_name.replace("_", " ").capitalize()} vs cores',
           xscale='log', base=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/{problem_name}/final_{metric_name}_comparison.png', dpi=300)
    return fig
