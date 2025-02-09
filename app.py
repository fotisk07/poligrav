import streamlit as st
import matplotlib.pyplot as plt
import os
from scripts.simulation import run_simulation

st.title("Poligrav: Political Opinion Simulation")
st.sidebar.header("Simulation Parameters")

N_ind = st.sidebar.number_input("Number of Individuals", min_value=50, max_value=1000, value=500, step=50)
N_party = st.sidebar.number_input("Number of Parties", min_value=1, max_value=10, value=2, step=1)
steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=5000, value=1000, step=100)
dt = st.sidebar.slider("Time Step (dt)", min_value=0.001, max_value=0.01, value=0.005, step=0.001, format="%.3f")
d = st.sidebar.number_input("Dimension (d)", min_value=2, max_value=100, value=50, step=1)
epsilon = st.sidebar.number_input("Epsilon", min_value=1e-5, max_value=1e-2, value=1e-3, step=1e-3, format="%.4f")
drag_coeff = st.sidebar.number_input("Epsilon", min_value=0, max_value=1, value=1e-3, step=1e-3, format="%.4f")

if st.sidebar.button("Run Simulation"):
    st.write("Running simulation... This may take a moment.")
    
    # Create a Streamlit progress bar.
    progress_bar = st.progress(0)
    
    # Define a function to update the progress bar.
    def update_progress(fraction):
        progress_bar.progress(fraction)
    
    metric_evo = run_simulation(N_ind, N_party, steps, dt, d, epsilon, drag_coeff=drag_coeff,progress_callback=update_progress)
    st.write("Simulation complete!")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metric_evo, marker='o', markersize=2, linestyle='-', color='b')
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean |v_i · v_j| (Individuals)")
    ax.set_title("Evolution of Opinion Alignment")
    ax.grid(True)
    st.pyplot(fig)
