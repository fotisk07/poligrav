import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------
# Simulation Parameters
# ------------------------
d = 2                      # Set to 2 for 2D, 3 for 3D simulation.
N_ind = 30                 # Number of individuals (kept small for visualization speed)
N_party = 1                # Number of parties
N_total = N_ind + N_party  # Total number of agents

steps = 500                # Number of simulation steps (frames)
dt = 0.01                  # Time step for simulation updates
epsilon = 1e-3             # Softening constant to avoid division by zero
drag_coeff = 0.1           # Drag coefficient to dampen large forces

# Mobility: individuals are agile (1), parties are sluggish (0.1)
mobility = np.ones(N_total)
mobility[N_ind:] = 0.1

# Masses: individuals are light (mass = 1/N_ind), parties are heavy (mass = 1)
mass = np.ones(N_total)
mass[:N_ind] = 1.0 / N_ind

# ------------------------
# Initialize Agent Vectors
# ------------------------
# Individuals: generate random vectors in R^d and normalize them to unit length.
individuals = np.random.randn(N_ind, d)
individuals /= np.linalg.norm(individuals, axis=1, keepdims=True)

if d == 2:
    # For 2D, generate parties as random angles.
    angles = 2 * np.pi * np.random.rand(N_party)
    party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (N_party, 2)
    parties = party_2d  # already in R^2 and unit length
elif d == 3:
    # For 3D, generate parties using spherical coordinates.
    phi = np.random.rand(N_party) * np.pi       # polar angle: [0, π]
    theta = np.random.rand(N_party) * 2 * np.pi   # azimuthal angle: [0, 2π]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    parties = np.column_stack((x, y, z))
else:
    raise ValueError("This script only supports d = 2 or d = 3 for visualization.")

# Combine individuals and parties.
V = np.vstack([individuals, parties])  # Shape: (N_total, d)

def normalize_rows(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

# ------------------------
# Set Up the Plot (2D or 3D)
# ------------------------
if d == 2:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    # Create two scatter objects: individuals (blue circles) and parties (red squares)
    scat_ind = ax.scatter(V[:N_ind, 0], V[:N_ind, 1], c='blue', s=50, marker='o', label='Individuals')
    scat_party = ax.scatter(V[N_ind:, 0], V[N_ind:, 1], c='red', s=150, marker='s', label='Parties')
    ax.legend()
    metric_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)
elif d == 3:
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1, 1, 1])
    scat_ind = ax.scatter(V[:N_ind, 0], V[:N_ind, 1], V[:N_ind, 2],
                          c='blue', s=50, marker='o', label='Individuals')
    scat_party = ax.scatter(V[N_ind:, 0], V[N_ind:, 1], V[N_ind:, 2],
                            c='red', s=150, marker='s', label='Parties')
    ax.legend()
    metric_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)
else:
    raise ValueError("Visualization only supports 2D or 3D.")

# ------------------------
# Animation Update Function
# ------------------------
def update(frame):
    global V
    # Compute pairwise dot products.
    dot_mat = V @ V.T
    dot_mat = np.clip(dot_mat, -1.0, 1.0)
    
    # Compute angles between agents.
    theta_mat = np.arccos(dot_mat)
    np.fill_diagonal(theta_mat, np.inf)
    
    # Compute the gravitational force factor.
    f = 1.0 / ((theta_mat + epsilon) ** 2)
    f[np.isinf(f)] = 0.0
    
    # Compute the tangent vectors.
    denom = np.sqrt(1 - dot_mat**2) + 1e-6
    T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
    T = T / denom[:, :, None]
    
    # Compute net force on each agent.
    force = (f * mass[None, :])[:, :, None] * T
    net_force = np.sum(force, axis=1)
    
    # --- Apply drag (vectorized) ---
    norms = np.linalg.norm(net_force, axis=1, keepdims=True)
    factors = 1.0 / (1.0 + drag_coeff * norms)
    net_force *= factors
    
    # Update positions using overdamped dynamics and re-normalize.
    V = V + dt * mobility[:, None] * net_force
    V = normalize_rows(V)
    
    # Compute the consensus metric (mean absolute inner product among individuals).
    V_ind = V[:N_ind]
    inner_ind = V_ind @ V_ind.T
    iu = np.triu_indices(N_ind, k=1)
    metric = np.mean(np.abs(inner_ind[iu]))
    if d == 2:
        metric_text.set_text(f"Metric: {metric:.3f}")
    else:  # d == 3
        metric_text.set_text(f"Metric: {metric:.3f}")
    
    # Update scatter plot positions.
    if d == 2:
        scat_ind.set_offsets(V[:N_ind, :])
        scat_party.set_offsets(V[N_ind:, :])
    elif d == 3:
        # For 3D, update using _offsets3d.
        scat_ind._offsets3d = (V[:N_ind, 0], V[:N_ind, 1], V[:N_ind, 2])
        scat_party._offsets3d = (V[N_ind:, 0], V[N_ind:, 1], V[N_ind:, 2])
    
    return scat_ind, scat_party, metric_text

# ------------------------
# Create and Run the Animation
# ------------------------
anim = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
plt.show()
