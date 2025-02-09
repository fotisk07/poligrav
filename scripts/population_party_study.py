import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from simulation import run_simulation  # Assuming __init__.py is in the scripts folder

# ------------------------
# Parameter Study Setup
# ------------------------
N_ind_list = [50, 100, 200]  # Different population sizes.
N_party_list = [1, 2, 3]         # Different numbers of parties.
steps = 1000
dt = 0.005
d = 50
epsilon = 1e-3

results = {}

# Loop over parameter combinations.
for N_ind in tqdm(N_ind_list, desc="Population sizes", ncols=80):
    for N_party in N_party_list:
        key = (N_ind, N_party)
        print(f"\nRunning simulation for N_ind = {N_ind}, N_party = {N_party}")
        
        # Create a tqdm progress bar for this simulation run.
        pbar = tqdm(total=steps, desc=f"Simulating for N_ind={N_ind}, N_party={N_party}", ncols=80, leave=False)
        
        def progress_callback(fraction):
            current = fraction * steps
            delta = current - pbar.n
            if delta > 0:
                pbar.update(delta)
        
        metric_evo = run_simulation(N_ind, N_party, steps, dt, d, epsilon, progress_callback=progress_callback)
        pbar.close()
        results[key] = metric_evo

# ------------------------
# Plotting the Results
# ------------------------
plt.figure(figsize=(12, 8))
for key, metric_evo in results.items():
    N_ind, N_party = key
    label = f"N_ind={N_ind}, N_party={N_party}"
    plt.plot(metric_evo, label=label)
plt.xlabel("Time Step")
plt.ylabel("Mean |v_i · v_j| (Individuals)")
plt.title("Evolution of Opinion Alignment for Different Population and Party Sizes")
plt.legend()
plt.grid(True)
plt.tight_layout()

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_path = os.path.join(results_dir, "population_party_study.png")
plt.savefig(output_path)
print("Saved graph to", output_path)
plt.show()
