import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Parameters
# ------------------------
d = 50              # Dimension of opinion space
N_ind = 100         # Number of individuals
N_party = 0        # Number of major parties
N_total = N_ind + N_party

steps = 3000        # Number of simulation steps
dt = 0.005          # Time step (try reducing if things become unstable)
epsilon = 1e-3      # Softening constant for force magnitude (to avoid singularities)

# Mobility factors: 
# Here we let individuals be more “mobile” and parties move slower.
mobility = np.ones(N_total)
mobility[N_ind:] = 0.1  # parties update 10x slower

# Masses: individuals are light (mass ~1/N_ind) and parties are heavy (mass 1)
mass = np.ones(N_total)
mass[:N_ind] = 1.0 / N_ind

# ------------------------
# Initialize agent vectors
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
# For example, choose random angles for each party.
angles = 2 * np.pi * np.random.rand(N_party)
# Coordinates in 2D for each party:
party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (N_party, 2)
# Embed into R^d:
parties = party_2d @ basis.T  # shape (N_party, d)
# (They are already unit because party_2d rows are unit and basis is orthonormal.)

# Combine individuals and parties into one array.
V = np.vstack([individuals, parties])  # shape (N_total, d)

# ------------------------
# Prepare for simulation
# ------------------------
# To record the evolution of the mean absolute inner product among individuals.
mean_abs_inner = []

def normalize_rows(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

# ------------------------
# Simulation loop
# ------------------------
for step in range(steps):
    # Compute pairwise dot products between all agent vectors.
    # V has shape (N_total, d), so dot_mat[i,j] = v_i dot v_j.
    dot_mat = V @ V.T
    # Clip dot values to avoid numerical issues.
    dot_mat = np.clip(dot_mat, -1.0, 1.0)
    
    # Compute the angle between each pair: theta_ij = arccos(v_i dot v_j).
    theta = np.arccos(dot_mat)
    # For i==j, we want to ignore self–interaction.
    np.fill_diagonal(theta, np.inf)  # so that 1/(inf+epsilon)^2 -> 0

    # Compute the “gravitational” factor f(theta) = 1/(theta+epsilon)^2.
    # (You can experiment with different exponents.)
    f = 1.0 / ((theta + epsilon)**2)  # shape (N_total, N_total)
    
    # For the tangent vector T_ij: note that for unit vectors,
    #   ||v_j - (v_i dot v_j)v_i|| = sqrt(1 - (v_i dot v_j)^2).
    denom = np.sqrt(1 - dot_mat**2) + 1e-6  # add a small number to avoid division by 0.
    
    # We want T_ij = (v_j - (v_i dot v_j) v_i) / ||v_j - (v_i dot v_j)v_i||.
    # Use broadcasting: for each pair (i,j), subtract (dot_mat[i,j] * V[i]) from V[j].
    # We'll build a (N_total, N_total, d) array.
    # Warning: For large N_total and d this may use a lot of memory.
    T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
    T = T / denom[:, :, None]
    
    # Now compute the net force for each agent.
    # For each i, sum over j: m_j * f_ij * T_ij.
    # The masses for j need to be broadcast appropriately.
    force = (f * mass[None, :])[:, :, None] * T  # shape (N_total, N_total, d)
    net_force = np.sum(force, axis=1)  # sum over j, shape (N_total, d)
    
    # Update each vector: move in the tangent space by a small amount.
    # (Here we use an overdamped dynamics update:
    #   v_i(new) = normalize( v_i + dt * mobility_i * net_force_i ).)
    V = V + dt * mobility[:, None] * net_force
    V = normalize_rows(V)
    
    # Optionally record the mean absolute inner product among individuals.
    V_ind = V[:N_ind]  # only the individuals
    inner_ind = V_ind @ V_ind.T
    # Exclude self–interactions (diagonal) by using the upper triangle.
    iu = np.triu_indices(N_ind, k=1)
    mean_abs = np.mean(np.abs(inner_ind[iu]))
    mean_abs_inner.append(mean_abs)
    
    # (Optional) Print progress every 100 steps.
    if step % 100 == 0:
        print(f"Step {step}: mean |inner product| among individuals = {mean_abs:.3f}")

# ------------------------
# Plot the evolution
# ------------------------
plt.figure(figsize=(8, 5))
plt.plot(mean_abs_inner, label="Mean |inner product| (individuals)")
plt.xlabel("Time step")
plt.ylabel("Mean |v_i · v_j|")
plt.title("Evolution of Opinion Alignment")
plt.legend()
plt.grid(True)
plt.show()

