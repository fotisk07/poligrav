import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # progress bar

def run_simulation(N_ind, N_party, steps=1000, dt=0.005, d=50, epsilon=1e-3):
    """
    Runs the simulation for a given number of individuals (N_ind) and parties (N_party)
    in a d-dimensional opinion space for a specified number of steps.
    
    Returns the evolution (list) of the mean absolute inner product among individuals.
    """
    N_total = N_ind + N_party

    # Set up mobility: individuals are agile (mobility = 1), parties are sluggish (mobility = 0.1)
    mobility = np.ones(N_total)
    mobility[N_ind:] = 0.1

    # Set up masses: individuals are light (mass ~ 1/N_ind), parties are heavy (mass = 1)
    mass = np.ones(N_total)
    mass[:N_ind] = 1.0 / N_ind

    # ------------------------
    # Initialize Agent Vectors
    # ------------------------
    # Individuals: randomly distributed on the unit sphere in R^d.
    individuals = np.random.randn(N_ind, d)
    individuals /= np.linalg.norm(individuals, axis=1, keepdims=True)

    # Parties: generated in a random 2D subspace of R^d.
    rand_matrix = np.random.randn(d, 2)
    # QR factorization gives an orthonormal basis for a 2D subspace.
    Q, _ = np.linalg.qr(rand_matrix)
    basis = Q[:, :2]  # shape (d,2)

    # Generate party vectors in that 2D plane using random angles.
    angles = 2 * np.pi * np.random.rand(N_party)
    party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (N_party, 2)
    parties = party_2d @ basis.T  # shape (N_party, d)

    # Combine individuals and parties.
    V = np.vstack([individuals, parties])  # shape (N_total, d)

    def normalize_rows(X):
        """Normalize each row of X to unit length."""
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    # ------------------------
    # Simulation Loop
    # ------------------------
    metric_evolution = []
    for step in tqdm(range(steps), desc=f"Simulating for N_ind={N_ind}, N_party={N_party}"):
        # Compute pairwise dot products between all agent vectors.
        dot_mat = V @ V.T
        dot_mat = np.clip(dot_mat, -1.0, 1.0)

        # Compute the angles between agents.
        theta = np.arccos(dot_mat)
        np.fill_diagonal(theta, np.inf)

        # Compute the "gravitational" force factor.
        f = 1.0 / ((theta + epsilon)**2)
        f[np.isinf(f)] = 0.0

        # Compute tangent vectors.
        denom = np.sqrt(1 - dot_mat**2) + 1e-6
        T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
        T = T / denom[:, :, None]

        # Compute net force on each agent.
        force = (f * mass[None, :])[:, :, None] * T
        net_force = np.sum(force, axis=1)

        # Update positions and re-normalize.
        V = V + dt * mobility[:, None] * net_force
        V = normalize_rows(V)

        # Record the mean absolute inner product among individuals.
        V_ind = V[:N_ind]
        inner_ind = V_ind @ V_ind.T
        iu = np.triu_indices(N_ind, k=1)
        mean_abs = np.mean(np.abs(inner_ind[iu]))
        metric_evolution.append(mean_abs)

    return metric_evolution

# ------------------------
# Parameter Study Setup
# ------------------------
N_ind_list = [100, 500, 1000]  # Different population sizes.
N_party_list = [1, 2, 3]         # Different numbers of parties.
steps = 1000

results = {}
# Loop over parameter combinations (with tqdm progress bar for outer loop).
for N_ind in tqdm(N_ind_list, desc="Population sizes"):
    for N_party in N_party_list:
        key = (N_ind, N_party)
        print(f"\nRunning simulation for N_ind = {N_ind}, N_party = {N_party}")
        metric_evo = run_simulation(N_ind, N_party, steps=steps, dt=0.005, d=50, epsilon=1e-3)
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

# Ensure the results folder exists
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save the figure
output_path = os.path.join(results_dir, "population_party_study.png")
plt.savefig(output_path)
print("Saved graph to", output_path)
plt.show()
