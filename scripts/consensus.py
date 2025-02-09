import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from simulation import run_simulation

# ---------------------------------------------------------------------
# Consensus Study: Effects of High Dimensionality on Opinion Consensus
# ---------------------------------------------------------------------
#
# We will test how the consensus (measured as the mean absolute inner 
# product among individuals) evolves for different dimensions of the 
# opinion space. For speed, we use a small group of individuals.
#
# Parameters:
#   - dimensions: list of dimensions (d) to test.
#   - N_ind: number of individuals (set small for faster runs).
#   - N_party: number of parties.
#   - steps: simulation steps.
#   - dt, epsilon: simulation parameters.
# ---------------------------------------------------------------------

# List of dimensions to test (e.g., from low to high)
dimensions = [2, 5, 10, 20, 50, 100]

# For fast runs, we use a small group.
N_ind = 50
N_party = 1

steps = 500    # Fewer simulation steps for speed
dt = 0.005
epsilon = 1e-3
drag_coeff = 0.1 # Drag coefficient

# Dictionary to hold simulation results for each dimension.
results = {}

# Loop over each dimension
for d in dimensions:
    print(f"Running simulation for dimension d = {d}")
    
    # Create a tqdm progress bar for this simulation run.
    pbar = tqdm(total=steps, desc=f"Dimension {d}", ncols=80)
    
    def progress_callback(fraction):
        # fraction is a number between 0 and 1.
        # Calculate the current step number and update the progress bar.
        current = fraction * steps
        delta = current - pbar.n
        if delta > 0:
            pbar.update(delta)
    
    # Run the simulation for the current dimension.
    metric_evo = run_simulation(N_ind, N_party, steps=steps, dt=dt, d=d, epsilon=epsilon, drag_coeff=drag_coeff, progress_callback=progress_callback)
    pbar.close()
    results[d] = metric_evo

# Plot the evolution of the consensus metric for each dimension.
plt.figure(figsize=(10, 6))
for d, metric_evo in results.items():
    plt.plot(metric_evo, label=f"Dimension {d}")
plt.xlabel("Simulation Step")
plt.ylabel("Mean |v_i · v_j| (Consensus Metric)")
plt.title("Consensus Evolution Across Different Dimensions")
plt.legend(title="Opinion Space Dimension")
plt.grid(True)
plt.tight_layout()

# Ensure the results folder exists and save the figure.
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
output_path = os.path.join(results_dir, "consensus_study.png")
plt.savefig(output_path)
print("Saved consensus study graph to", output_path)
plt.show()
