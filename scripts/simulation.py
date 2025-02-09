import numpy as np

def run_simulation(N_ind, N_party, steps=1000, dt=0.005, d=10, epsilon=1e-3, progress_callback=None):
    """
    Runs the simulation for a given number of individuals and parties.
    Optionally calls progress_callback(fraction) on each iteration.

    Returns:
        metric_evolution: A list of the alignment metric at each step.
    """
    N_total = N_ind + N_party

    # Mobility: individuals agile, parties sluggish.
    mobility = np.ones(N_total)
    mobility[N_ind:] = 0.1

    # Masses: individuals light, parties heavy.
    mass = np.ones(N_total)
    mass[:N_ind] = 1.0 / N_ind

    # Initialize Agent Vectors
    individuals = np.random.randn(N_ind, d)
    individuals /= np.linalg.norm(individuals, axis=1, keepdims=True)

    rand_matrix = np.random.randn(d, 2)
    Q, _ = np.linalg.qr(rand_matrix)
    basis = Q[:, :2]
    angles = 2 * np.pi * np.random.rand(N_party)
    party_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    parties = party_2d @ basis.T

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

        # If a progress_callback is provided, call it with the fraction completed.
        if progress_callback is not None:
            progress_callback((step + 1) / steps)
    return metric_evolution
