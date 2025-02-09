import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# (Simulation parameters remain the same as before)
d = 10
N_ind = 500
N_party = 2
N_total = N_ind + N_party
steps = 1000
dt = 0.005
epsilon = 1e-3

mobility = np.ones(N_total)
mobility[N_ind:] = 0.1
mass = np.ones(N_total)
mass[:N_ind] = 1.0 / N_ind

# Initialize individuals and parties (same as before) ...
# [Initialization code omitted for brevity; assume V is created as before]

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

def normalize_rows_inplace(V):
    # Compute norms (since V is expected to be non-zero, no division by zero)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V /= norms  # In-place division

# Pre-allocate temporary arrays if possible (optional)
# For example, if you know the shape of dot_mat, you can do:
dot_mat = np.empty((N_total, N_total))
theta = np.empty_like(dot_mat)

mean_abs_inner = []
pbar = tqdm(range(steps), desc="Simulating alignment evolution")
for step in pbar:
    # Compute dot products (V @ V.T) and store in dot_mat
    np.dot(V, V.T, out=dot_mat)
    np.clip(dot_mat, -1.0, 1.0, out=dot_mat)

    # Compute theta = arccos(dot_mat) elementwise, reusing the theta array
    np.arccos(dot_mat, out=theta)
    np.fill_diagonal(theta, np.inf)

    # Compute f = 1/(theta+epsilon)^2
    f = 1.0 / ((theta + epsilon)**2)
    f[np.isinf(f)] = 0.0  # Replace any infinite values with zero.

    # Compute the tangent vectors T:
    # Instead of allocating a new T every time, you could try to reuse a pre-allocated array,
    # but note that T has shape (N_total, N_total, d).
    T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
    T = T / (np.sqrt(1 - dot_mat**2)[:, :, None] + 1e-6)

    force = (f * mass[None, :])[:, :, None] * T
    net_force = np.sum(force, axis=1)

    V = V + dt * mobility[:, None] * net_force
    normalize_rows_inplace(V)

    # Record the metric (same as before)
    V_ind = V[:N_ind]
    inner_ind = V_ind @ V_ind.T
    iu = np.triu_indices(N_ind, k=1)
    mean_abs = np.mean(np.abs(inner_ind[iu]))
    mean_abs_inner.append(mean_abs)
    pbar.set_postfix(mean_metric=f"{mean_abs:.3f}")

# (Plotting and saving the graph remain the same)
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
plt.figure(figsize=(8, 5))
plt.plot(mean_abs_inner, marker='o', markersize=2, linestyle='-', color='b')
plt.xlabel("Simulation Step")
plt.ylabel("Mean |v_i · v_j| (Individuals)")
plt.title("Evolution of Opinion Alignment")
plt.grid(True)
plt.tight_layout()
output_path = os.path.join(results_dir, "alignment_evolution.png")
plt.savefig(output_path)
print("Saved graph to", output_path)
plt.show()
