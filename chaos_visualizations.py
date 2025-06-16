# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import streamlit as st
import pykoopman as pk
from pykoopman.common import lorenz
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide", page_title="Lorenz & HAVOK Explorer")
st.title("Lorenz & HAVOK Explorer")

# Shared IC sliders
st.sidebar.header("Initial Conditions")
x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01, key="x0")
y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01, key="y0")
z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01, key="z0")

module = st.sidebar.radio("Module",
    ["Attractors & Divergence", "HAVOK Reconstruction"],
    key="module"
)

if module == "Attractors & Divergence":
    st.sidebar.header("Divergence Settings")
    t_final = st.sidebar.number_input(
        "t_final", 1.0, 100.0, 50.0, 1.0,
        key="t_final_div"
    )
    n_steps = st.sidebar.slider(
        "n_steps", 100, 10_000, 5_000, 100,
        key="n_steps_div"
    )
    t = np.linspace(0, t_final, n_steps)

    traj1 = integrate.odeint(lorenz, [x0, y0, z0], t)
    traj2 = integrate.odeint(lorenz, [x0+0.1, y0, z0], t)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    # 3D plot
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(*traj1.T, color="blue",  label="Attractor 1", lw=1.5)
    ax.plot(*traj2.T, color="green", label="Attractor 2", lw=1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Lorenz Attractors")
    ax.legend()
    st.pyplot(fig)

    # divergence
    fig2 = plt.figure(figsize=(8,3))
    plt.plot(t, dist, color="red", lw=2)
    plt.xlabel("Time"); plt.ylabel("‖X₁(t)–X₂(t)‖")
    plt.title("Distance Over Time")
    plt.tight_layout()
    st.pyplot(fig2)

else:
    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input(
        "dt", 1e-4, 1.0, 0.01, 1e-4, format="%.4f", key="dt_havok"
    )
    tau       = st.sidebar.slider(
        "τ", dt, 10.0, 1.0, dt, format="%.2f", key="tau_havok"
    )
    embed_dim = st.sidebar.slider(
        "m", 2, 100, 10, 1, key="m_havok"
    )
    t_final_h = st.sidebar.number_input(
        "Total time for HAVOK", 1.0, 200.0, 50.0, 1.0,
        key="t_final_havok"
    )

    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0,y0,z0], t_h)
    x_series = X[:,0]
    delay_steps = int(tau / dt)
    n_delays   = embed_dim - 1

    TDC   = pk.observables.TimeDelay(delay=delay_steps, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1,1), dt=dt)

    u       = model.regressor.forcing_signal
    x0_e    = x_series[:n_delays+1].reshape(-1,1)
    t_sim   = t_h[n_delays:] - t_h[n_delays]
    x_pred  = model.simulate(x0_e, t_sim, u).flatten()

    fig3 = plt.figure(figsize=(8,4))
    plt.plot(t_h[n_delays:], x_series[n_delays:], "-b", label="Original x")
    plt.plot(t_h[n_delays:], x_pred,      "--r", label="HAVOK Reconstructed")
    plt.xlabel("Time"); plt.ylabel("x")
    plt.title(f"HAVOK (τ={tau:.2f}, m={embed_dim}, dt={dt:.3f})")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig3)
