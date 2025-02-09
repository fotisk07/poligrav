import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from simulation import run_simulation  # Assuming __init__.py is in the scripts folder

# ------------------------
# Simulation Parameters
# ------------------------
d = 10          # Dimensionality of the space
N_ind = 100     # Number of individuals
N_party = 2     # Number of major parties
steps = 1000    # Number of simulation steps
dt = 0.005      # Time step for the simulation
epsilon = 1e-3  # Softening constant

# ------------------------
# Set up a tqdm progress bar
# ------------------------
pbar = tqdm(total=steps, desc="Simulation progress", ncols=80)

def progress_callback(fraction):
    # fraction is between 0 and 1
    # Calculate the current step number and update the tqdm bar accordingly.
    current = fraction * steps
    delta = current - pbar.n
    if delta > 0:
        pbar.update(delta)

# ------------------------
# Run the Simulation
# ------------------------
mean_abs_inner = run_simulation(N_ind, N_party, steps, dt, d, epsilon, progress_callback=progress_callback)
pbar.close()

# ------------------------
# Plot the Evolution of the Metric
# ------------------------
plt.figure(figsize=(8, 5))
plt.plot(mean_abs_inner, marker='o', markersize=2, linestyle='-', color='b')
plt.xlabel("Simulation Step")
plt.ylabel("Mean |v_i · v_j| (Individuals)")
plt.title("Evolution of Opinion Alignment")
plt.grid(True)
plt.tight_layout()

# Ensure the results folder exists.
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_path = os.path.join(results_dir, "alignment_evolution.png")
plt.savefig(output_path)
print("Saved graph to", output_path)
plt.show()
