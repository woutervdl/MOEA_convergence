import os
import h5py
import pandas as pd
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem
from Thesis.util.model_definitions import get_justice_model

# Settings
justice_root = "../hdf5_results/JUSTICE"
core_counts = [16, 32, 48]
moeas = ["eps_nsgaii", "borg", "generational_borg"]
seeds = ["seed12345", "seed23403", "seed39349", "seed60930", "seed93489"]
epsilons = [0.01, 0.25, 10, 10]
model = get_justice_model()
problem = to_problem(model, searchover="levers")

# Container for DataFrames
df_list = []

# Loop over all combinations
for cores in core_counts:
    for moea in moeas:
        for seed in seeds:
            h5_path = os.path.join(justice_root, f"{cores}cores", moea, seed, f"final_state_JUSTICE_{moea}_{cores}cores_{seed}.h5")
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, "r") as f:
                    results_group = f["final_archive"]
                    data = {key: results_group[key][()] for key in results_group}
                    df = pd.DataFrame(data)
                    df_list.append(df)
            else:
                print(f"File not found: {h5_path}")

global_ref_set = epsilon_nondominated(df_list, epsilons, problem)

# Save result
csv_output = "../global_JUSTICE_ref_set.csv"
global_ref_set.to_csv(csv_output, index=False)

print(f"Combined DataFrame saved to:\n{csv_output}")