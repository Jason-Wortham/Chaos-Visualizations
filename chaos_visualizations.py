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
from mpl_toolkits.mplot3d import Axes3D  # noqa

# 1) Must be first
st.set_page_config(layout="wide", page_title="Lorenz & HAVOK Explorer")
st.title("Lorenz & HAVOK Explorer")

# 2) Module selector
module = st.sidebar.radio(
    "Select Module",
    ["Attractors & Divergence", "HAVOK Reconstruction"],
    key="which_module"
)

# 3) Attractors & Divergence (unchanged) branch
if module == "Attractors & Divergence":
    st.sidebar.header("Attractor 1 Initial Conditions")
    x0_1 = st.sidebar.slider("x₀ (A1)", -10.0, 10.0, 1.0, 0.01, key="x0_1")
    y0_1 = st.sidebar.slider("y₀ (A1)", -10.0, 10.0, 1.0, 0.01, key="y0_1")
    z0_1 = st.sidebar.slider("z₀ (A1)", -10.0, 10.0, 1.0, 0.01, key="z0_1")

    st.sidebar.header("Attractor 2 Initial Conditions")
    x0_2 = st.sidebar.slider("x₀ (A2)", -10.0, 10.0, 1.0, 0.01, key="x0_2")
    y0_2 = st.sidebar.slider("y₀ (A2)", -10.0, 10.0, 1.0, 0.01, key="y0_2")
    z0_2 = st.sidebar.slider("z₀ (A2)", -10.0, 10.0, 1.0, 0.01, key="z0_2")

    st.sidebar.header("Divergence Settings")
    t_final = st.sidebar.number_input(
        "t_final", 1.0, 100.0, 50.0, 1.0, key="t_final_div"
    )
    n_steps = st.sidebar.slider(
        "n_steps", 100, 10_000, 5_000, 100, key="n_steps_div"
    )

    t     = np.linspace(0, t_final, n_steps)
    traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
    traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    # 3D plot
    fig1 = plt.figure(figsize=(8,6))
    ax1  = fig1.add_subplot(111, projection="3d")
    ax1.plot(*traj1.T, color="blue",  label="Attractor 1", lw=1.5)
    ax1.plot(*traj2.T, color="green", label="Attractor 2", lw=1.5)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("Lorenz Attractors")
    ax1.legend()
    st.pyplot(fig1)

    # divergence curve
    fig2 = plt.figure(figsize=(8,3))
    plt.plot(t, dist, color="red", lw=2)
    plt.xlabel("Time"); plt.ylabel("‖X₁(t)–X₂(t)‖")
    plt.title("Distance Between Trajectories Over Time")
    plt.tight_layout()
    st.pyplot(fig2)

# 4) HAVOK Reconstruction branch – with “tau between rows”
else:
    st.sidebar.header("HAVOK: Initial Conditions")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01, key="x0_h")
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01, key="y0_h")
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01, key="z0_h")

    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input(
        "dt", 1e-4, 1.0, 0.01, 1e-4, format="%.4f", key="dt_havok"
    )
    tau       = st.sidebar.number_input(
        "Time delay τ", min_value=dt, max_value=10.0,
        value=1.0, step=dt, format="%.4f", key="tau_havok"
    )
    embed_dim = st.sidebar.number_input(
        "Embedding dimension m (≥3)",
        min_value=3, max_value=100, value=10, step=1, key="m_havok"
    )
    t_final_h = st.sidebar.number_input(
        "Total time for HAVOK", 1.0, 200.0, 50.0, 1.0, key="t_final_havok"
    )

    # 4.a) simulate Lorenz to get x_series
    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x_series = X[:, 0]

    # 4.b) build & fit HAVOK model on x₁
    #     we still need a TimeDelay with delay=1 for regression
    n_delays = embed_dim - 1
    TDC   = pk.observables.TimeDelay(delay=1, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1,1), dt=dt)

    # 4.c) simulate forward
    u_full = model.regressor.forcing_signal.reshape(-1,1)
    f      = u_full.shape[0]
    d      = n_delays
    x0_e   = x_series[: d+1].reshape(-1,1)
    t_sim  = t_h[d:d+f] - t_h[d]
    u_sim  = u_full[:f]
    x_pred = model.simulate(x0_e, t_sim, u_sim).flatten()

    # 4.d) **manual** delay‐embedding so that
    #      • columns step by 1 dt, and
    #      • rows step by tau (in steps = int(tau/dt))
    lag_steps = max(1, int(tau / dt))
    max_idx   = len(x_pred) - (embed_dim - 1) * lag_steps

    # build an array of shape (m, max_idx)
    M = np.zeros((embed_dim, max_idx))
    for i in range(embed_dim):
        M[i, :] = x_pred[i*lag_steps : i*lag_steps + max_idx]

    # pick the first 3 rows for a 3D plot
    xs, ys, zs = M[0], M[1], M[2]

    fig3 = plt.figure(figsize=(6,6))
    ax3  = fig3.add_subplot(111, projection="3d")
    ax3.plot(xs, ys, zs, lw=1.5)
    ax3.set_xlabel("x(t)")
    ax3.set_ylabel(f"x(t+τ)  τ={tau:.3f}")
    ax3.set_zlabel(f"x(t+2τ) τ={2*tau:.3f}")
    ax3.set_title("HAVOK Reconstructed Time‑Delay Attractor")
    st.pyplot(fig3)
