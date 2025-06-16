# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
import pykoopman as pk
from pykoopman.common import lorenz


st.title("Lorenz Attractors & Divergence")

st.sidebar.header("Attractor 1 Initial Conditions")
x0_1 = st.sidebar.slider("x_0 (A1)", -10.0, 10.0, 1.0, 0.01)
y0_1 = st.sidebar.slider("y_0 (A1)", -10.0, 10.0, 1.0, 0.01)
z0_1 = st.sidebar.slider("z_0 (A1)", -10.0, 10.0, 1.0, 0.01)

st.sidebar.header("Attractor 2 Initial Conditions")
x0_2 = st.sidebar.slider("x_0 (A2)", -10.0, 10.0, 1.0, 0.01)
y0_2 = st.sidebar.slider("y_0 (A2)", -10.0, 10.0, 1.0, 0.01)
z0_2 = st.sidebar.slider("z_0 (A2)", -10.0, 10.0, 1.0, 0.01)

t_final = st.sidebar.number_input("t_final", min_value = 1.0, max_value = 100.0, value = 50.0, step = 1.0)
n_steps = st.sidebar.slider("n_steps", 100, 10000, 5000, 100)
t = np.linspace(0, t_final, n_steps)

traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)

dist = np.linalg.norm(traj2 - traj1, axis = 1)

fig1 = plt.figure(figsize = (8, 6))
ax = fig1.add_subplot(111, projection = '3d')
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], color = "blue", label = "Attractor 1", lw = 1.5)
ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], color = "green", label = "Attractor 2", lw = 1.5)
ax.set_title("Lorenz Attractors")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend()
st.pyplot(fig1)

fig2 = plt.figure(figsize = (8, 3))
plt.plot(t, dist, color = "red", lw = 2)
plt.title("Distance Between Trajectories Over Time")
plt.xlabel("Time")
plt.ylabel("‖X_1(t) – X_2(t)‖")
plt.tight_layout()
st.pyplot(fig2)



st.set_page_config(layout="wide")
st.title("Lorenz & HAVOK Explorer")

# --- attractor initial conditions (shared) ---
st.sidebar.header("Attractor Initial Conditions")
x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01)
y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01)
z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01)

# --- choose module ---
module = st.sidebar.radio(
    "Module",
    ("Attractors & Divergence", "HAVOK Reconstruction")
)

if module == "Attractors & Divergence":
    # Your existing code, slightly reformatted:
    st.sidebar.header("Divergence Settings")
    t_final = st.sidebar.number_input("t_final", 1.0, 100.0, 50.0, 1.0)
    n_steps = st.sidebar.slider("n_steps", 100, 10000, 5000, 100)
    t = np.linspace(0, t_final, n_steps)

    traj1 = integrate.odeint(lorenz, [x0, y0, z0], t)
    traj2 = integrate.odeint(lorenz, [x0 + 0.1, y0, z0], t)  # example small perturbation

    dist = np.linalg.norm(traj2 - traj1, axis=1)

    # 3D plot
    fig1 = plt.figure(figsize=(8, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], color="blue", label="Attractor 1", lw=1.5)
    ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], color="green", label="Attractor 2", lw=1.5)
    ax.set_title("Lorenz Attractors")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend()
    st.pyplot(fig1)

    # divergence plot
    fig2 = plt.figure(figsize=(8, 3))
    plt.plot(t, dist, lw=2)
    plt.title("Distance Between Trajectories Over Time")
    plt.xlabel("Time"); plt.ylabel("‖X₁(t) – X₂(t)‖")
    plt.tight_layout()
    st.pyplot(fig2)

else:
    st.header("HAVOK Reconstruction of Time‐Delay Attractor")

    # --- HAVOK inputs ---
    dt = st.sidebar.number_input(
        "Time granularity (dt)",
        min_value=1e-4, max_value=1.0, value=0.01, step=1e-4, format="%.4f"
    )
    tau = st.sidebar.slider(
        "Time delay (τ)", 
        min_value=dt, max_value=10.0, value=1.0, step=dt, format="%.2f"
    )
    embed_dim = st.sidebar.slider(
        "Embedding dimension (m)", 
        min_value=2, max_value=100, value=10, step=1
    )
    t_final_h = st.sidebar.number_input(
        "Total time for HAVOK", 1.0, 200.0, 50.0, 1.0
    )

    # --- prepare data ---
    t_h = np.arange(0, t_final_h, dt)
    X = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x_series = X[:, 0]  # only first coordinate

    delay_steps = max(1, int(tau / dt))
    n_delays = embed_dim - 1  # so total embedding = n_delays+1

    # --- build and fit HAVOK model ---
    TDC = pk.observables.TimeDelay(delay=delay_steps, n_delays=n_delays)
    Diff = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    # fit on the 1D series (reshape for sklearn‐style API)
    model.fit(x_series.reshape(-1,1), dt=dt)

    # --- simulate / reconstruct ---
    u = model.regressor.forcing_signal
    x0_embed = x_series[: n_delays + 1].reshape(-1,1)
    t_sim = t_h[n_delays:] - t_h[n_delays]
    x_pred = model.simulate(x0_embed, t_sim, u).flatten()

    # --- plot reconstruction ---
    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(t_h[n_delays:], x_series[n_delays:], '-b', label="Original x")
    plt.plot(t_h[n_delays:], x_pred,      '--r', label="HAVOK Reconstructed")
    plt.xlabel("Time"); plt.ylabel("x")
    plt.title(f"HAVOK (τ={tau:.2f}, m={embed_dim}, dt={dt:.3f})")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig3)
