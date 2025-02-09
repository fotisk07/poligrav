import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # progress bar

# ------------------------
# Simulation Parameters
# ------------------------
d = 10          # Dimensionality of the space (increase as desired)
N_ind = 500     # Number of individuals
N_party = 2     # Number of major parties
N_total = N_ind + N_party

steps = 1000    # Number of simulation steps
dt = 0.005      # Time step for the simulation
epsilon = 1e-3  # Softening constant for force magnitude (to avoid singularities)

# Mobility factors:
# Individuals are more “mobile” and parties move slower.
mobility = np.ones(N_total)
mobility[N_ind:] = 0.1  # Parties update 10x slower

# Masses: individuals are light (mass ~ 1/N_ind) and parties are heavy (mass 1)
mass = np.ones(N_total)
mass[:N_ind] = 1.0 / N_ind

# ------------------------
# Initialize Agent Vectors
# ------------------------

# Generate individuals: uniformly random on the unit sphere in R^d.
individuals = np.random.randn(N_ind, d)
individuals /= np.linalg.norm(individuals, axis=1, keepdims=True)

# Generate parties: first choose a random 2D subspace of R^d.
rand_matrix = np.random.randn(d, 2)
# Use QR factorization to get an orthonormal basis for a 2D subspace.
Q, _ = np.linalg.qr(rand_matrix)
basis = Q[:, :2]  # shape (d,2)

# Now generate party vectors in that 2D plane.
angles = 2 * np.pi * np.random.rand(N_party)
# Coordinates in 2D for each party:
party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (N_party, 2)
# Embed into R^d:
parties = party_2d @ basis.T  # shape (N_party, d)

# Combine individuals and parties into one array.
V = np.vstack([individuals, parties])  # shape (N_total, d)

def normalize_rows(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

# ------------------------
# Prepare for Simulation
# ------------------------
# To record the evolution of the mean absolute inner product among individuals.
mean_abs_inner = []

# Use tqdm with a variable so we can update its postfix.
pbar = tqdm(range(steps), desc="Simulating alignment evolution")
for step in pbar:
    # Compute pairwise dot products between all agent vectors.
    dot_mat = V @ V.T
    dot_mat = np.clip(dot_mat, -1.0, 1.0)
    
    # Compute the angle between each pair: theta_ij = arccos(v_i dot v_j).
    theta = np.arccos(dot_mat)
    # Ignore self–interaction by setting the diagonal to infinity.
    np.fill_diagonal(theta, np.inf)
    
    # Compute the gravitational factor f(theta) = 1/(theta+epsilon)^2.
    f = 1.0 / ((theta + epsilon) ** 2)
    f[np.isinf(f)] = 0.0  # Replace any infinite values with zero.
    
    # Compute the tangent vectors.
    denom = np.sqrt(1 - dot_mat**2) + 1e-6  # Avoid division by zero.
    # For each pair (i, j), compute T_ij = (v_j - (v_i dot v_j)v_i) / ||v_j - (v_i dot v_j)v_i||
    T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
    T = T / denom[:, :, None]
    
    # Compute the net force on each agent.
    # For each i, sum over j: mass_j * f_ij * T_ij.
    force = (f * mass[None, :])[:, :, None] * T  # shape (N_total, N_total, d)
    net_force = np.sum(force, axis=1)             # shape (N_total, d)
    
    # Update positions using overdamped dynamics and re-normalize.
    V = V + dt * mobility[:, None] * net_force
    V = normalize_rows(V)
    
    # Record the mean absolute inner product among individuals.
    V_ind = V[:N_ind]  # Only the individuals
    inner_ind = V_ind @ V_ind.T
    # Use the upper triangle (excluding diagonal) to avoid double-counting.
    iu = np.triu_indices(N_ind, k=1)
    mean_abs = np.mean(np.abs(inner_ind[iu]))
    mean_abs_inner.append(mean_abs)
    
    # Update the progress bar's postfix with the current metric.
    pbar.set_postfix(mean_metric=f"{mean_abs:.3f}")

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

# Ensure the results folder exists
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save the figure
output_path = os.path.join(results_dir, "alignment_evolution.png")
plt.savefig(output_path)
print("Saved graph to", output_path)
plt.show()
