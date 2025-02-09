import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------
# Simulation Parameters
# ------------------------
d = 50      # dimensionality of the space
N = 100     # number of people
epsilon = 1e-3  # to avoid division by zero
dt = 0.01
N_steps = 100   # Total number of simulation steps
drag_coeff = 0.1 # Drag coefficient

# ------------------------
# Initialisation
# ------------------------
people = 2 * np.random.rand(N, d) - 1  
people /= np.linalg.norm(people, axis=1, keepdims=True) 

def normalize_rows(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

# ------------------------
# Set up the Plot
# ------------------------
fig, ax = plt.subplots()
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
scat = ax.scatter(people[:, 0], people[:, 1])

# Create a text object to display the current step and total steps.
step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# ------------------------
# Animation Update Function
# ------------------------
def update(frame):
    global people
    # Compute pairwise dot products and corresponding angles.
    dot_mat = np.clip(people @ people.T, -1.0, 1.0)
    theta = np.arccos(dot_mat)
    np.fill_diagonal(theta, np.inf)  # Avoid self-interaction.
    
    # Compute the force magnitude based on angular separation.
    f = 1.0 / (theta**2 + epsilon**2)

    
    # Compute the tangent vectors.
    denom = np.sqrt(1 - dot_mat**2) + 1e-6  # Avoid division by 0.
    T = people[None, :, :] - (dot_mat[:, :, None] * people[:, None, :])
    T = T / denom[:, :, None]
    
    # Sum forces from all other particles.
    force = f[:, :, None] * T         # shape (N, N, d)
    net_force = np.sum(force, axis=1)   # shape (N, d)
    
    # Update positions and re-normalize to project back onto the circle.
    people = people + dt * net_force
    people = normalize_rows(people)
    
    # Update the scatter plot with new positions.
    scat.set_offsets(people)
    
    # Update the step counter display.
    step_text.set_text(f"Step: {frame + 1} / {N_steps}")
    
    return scat, step_text

# ------------------------
# Create the Animation
# ------------------------
ani = animation.FuncAnimation(fig, update, frames=range(N_steps),
                              interval=50, blit=True, repeat=False)

plt.show()
