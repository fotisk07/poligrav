import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Function ---
def run_simulation(N_ind, N_party, steps, dt, d, epsilon):
    """
    Runs the political opinion simulation.
    
    Parameters:
    - N_ind: Number of individuals
    - N_party: Number of parties
    - steps: Number of simulation steps
    - dt: Time step
    - d: Dimension of opinion space
    - epsilon: Softening constant
    
    Returns:
    - metric_evolution: A list of the mean absolute inner product (alignment metric) at each step.
    """
    N_total = N_ind + N_party

    # Define mobility: individuals are agile (1), parties are sluggish (0.1)
    mobility = np.ones(N_total)
    mobility[N_ind:] = 0.1

    # Define masses: individuals are light (mass = 1/N_ind), parties are heavy (mass = 1)
    mass = np.ones(N_total)
    mass[:N_ind] = 1.0 / N_ind

    # Initialize agent vectors
    # Individuals: random unit vectors in R^d
    individuals = np.random.randn(N_ind, d)
    individuals /= np.linalg.norm(individuals, axis=1, keepdims=True)

    # Parties: generate in a random 2D subspace of R^d
    rand_matrix = np.random.randn(d, 2)
    Q, _ = np.linalg.qr(rand_matrix)
    basis = Q[:, :2]
    angles = 2 * np.pi * np.random.rand(N_party)
    party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    parties = party_2d @ basis.T

    # Combine individuals and parties
    V = np.vstack([individuals, parties])

    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    metric_evolution = []
    for step in range(steps):
        dot_mat = V @ V.T
        dot_mat = np.clip(dot_mat, -1.0, 1.0)
        theta = np.arccos(dot_mat)
        np.fill_diagonal(theta, np.inf)
        f = 1.0 / ((theta + epsilon)**2)
        f[np.isinf(f)] = 0.0
        denom = np.sqrt(1 - dot_mat**2) + 1e-6
        T = V[None, :, :] - (dot_mat[:, :, None] * V[:, None, :])
        T = T / denom[:, :, None]
        force = (f * mass[None, :])[:, :, None] * T
        net_force = np.sum(force, axis=1)
        V = V + dt * mobility[:, None] * net_force
        V = normalize_rows(V)
        V_ind = V[:N_ind]
        inner_ind = V_ind @ V_ind.T
        iu = np.triu_indices(N_ind, k=1)
        mean_abs = np.mean(np.abs(inner_ind[iu]))
        metric_evolution.append(mean_abs)
    return metric_evolution

# --- Streamlit App Interface ---
st.title("Poligrav: Political Opinion Simulation")

st.sidebar.header("Simulation Parameters")
N_ind = st.sidebar.number_input("Number of Individuals", min_value=50, max_value=1000, value=500, step=50)
N_party = st.sidebar.number_input("Number of Parties", min_value=1, max_value=10, value=2, step=1)
steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=5000, value=1000, step=100)
dt = st.sidebar.slider("Time Step (dt)", min_value=0.001, max_value=0.01, value=0.005, step=0.001, format="%.3f")
d = st.sidebar.number_input("Dimension (d)", min_value=2, max_value=100, value=50, step=1)
epsilon = st.sidebar.number_input("Epsilon", min_value=1e-5, max_value=1e-2, value=1e-3, step=1e-3, format="%.4f")

if st.sidebar.button("Run Simulation"):
    st.write("Running simulation... This may take a moment.")
    metric_evo = run_simulation(N_ind, N_party, steps, dt, d, epsilon)
    st.write("Simulation complete!")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metric_evo, marker='o', markersize=2, linestyle='-', color='b')
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean |v_i · v_j| (Individuals)")
    ax.set_title("Evolution of Opinion Alignment")
    ax.grid(True)
    st.pyplot(fig)
