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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) Page config
st.set_page_config(layout="wide", page_title="Lorenz & HAVOK Explorer")
st.title("Lorenz & HAVOK Explorer")

# 2) Module chooser
module = st.sidebar.radio(
    "Select Module",
    ["Attractors & Divergence", "HAVOK Reconstruction"],
    key="which_module"
)

if module == "Attractors & Divergence":
    # — Divergence branch UI —
    st.sidebar.header("Attractor 1 Initial Conditions")
    x0_1 = st.sidebar.number_input("x₀ (A1)", -20.0, 20.0, -8.0, step=0.1, key="x0_1")
    y0_1 = st.sidebar.number_input("y₀ (A1)", -20.0, 20.0,  8.0, step=0.1, key="y0_1")
    z0_1 = st.sidebar.number_input("z₀ (A1)",   0.0, 54.0, 27.0, step=0.1, key="z0_1")

    st.sidebar.header("Attractor 2 Initial Conditions")
    x0_2 = st.sidebar.number_input("x₀ (A2)", -20.0, 20.0, -8.1, step=0.1, key="x0_2")
    y0_2 = st.sidebar.number_input("y₀ (A2)", -20.0, 20.0,  8.0, step=0.1, key="y0_2")
    z0_2 = st.sidebar.number_input("z₀ (A2)",   0.0, 54.0, 27.0, step=0.1, key="z0_2")

    st.sidebar.header("Divergence Settings")
    dt_div   = st.sidebar.number_input("dt (for divergence)", 1e-4, 1.0, 0.001, step=1e-4, format="%.4f", key="dt_div")
    t_end    = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, step=1.0, key="t_end_div")

    # integrate
    t_div = np.arange(0, t_end, dt_div)
    traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t_div)
    traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t_div)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    # plot attractors
    fig1 = plt.figure(figsize=(8,6))
    ax1  = fig1.add_subplot(111, projection="3d")
    ax1.plot(*traj1.T, color="blue",  lw=0.5, label="True Lorenz A1")
    ax1.plot(*traj2.T, color="green", lw=0.5, label="Perturbed A2")
    ax1.set_title("Lorenz Trajectories")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend()
    st.pyplot(fig1)

    # plot divergence
    fig2 = plt.figure(figsize=(8,3))
    plt.plot(t_div, dist, '-r')
    plt.xlabel("Time"); plt.ylabel("‖ΔX‖")
    plt.title("Separation of Two Trajectories")
    plt.tight_layout()
    st.pyplot(fig2)


else:
    # — HAVOK branch UI —
    st.sidebar.header("Initial Condition")
    x0 = st.sidebar.number_input("x₀", -20.0, 20.0, -8.0, step=0.1, key="x0_h")
    y0 = st.sidebar.number_input("y₀", -20.0, 20.0,  8.0, step=0.1, key="y0_h")
    z0 = st.sidebar.number_input("z₀",   0.0, 54.0, 27.0, step=0.1, key="z0_h")

    st.sidebar.header("HAVOK Settings")
    dt       = st.sidebar.number_input("Time step dt",    1e-4, 1.0, 0.001, step=1e-4, format="%.4f", key="dt_havok")
    tau      = st.sidebar.number_input("Time delay τ (s)", dt,    10.0, dt*30, step=dt,  format="%.4f", key="tau_havok")
    d        = st.sidebar.number_input("Embedding dimension d", 3, 200, 100,    step=1,    key="d_havok")
    t_final  = st.sidebar.number_input("Total time T",    1.0, 200.0, 20.0,    step=1.0,  key="t_final_havok")
    SVD_RANK = st.sidebar.number_input("SVD rank r",      1,    100,   15,      step=1,    key="svd_havok")

    # 1) simulate “truth”
    t_h      = np.arange(0, t_final, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x1       = X[:, 0]

    # 2) set up delays & derivatives
    lag_steps = max(1, int(tau / dt))
    n_delays  = d - 1
    TDC       = pk.observables.TimeDelay(delay=1, n_delays=n_delays)
    Diff      = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK     = pk.regression.HAVOK(svd_rank=SVD_RANK, plot_sv=True)

    # 3) build & fit Koopman model
    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x1.reshape(-1,1), dt=dt)

    # 4) extract forcing & simulate reduced system
    u_full = model.regressor.forcing_signal.reshape(-1,1)
    seed   = x1[: n_delays + 1].reshape(-1,1)
    t_sim  = t_h[n_delays:] - t_h[n_delays]
    u_sim  = u_full[n_delays:]
    x_pred = model.simulate(seed, t_sim, u_sim).flatten()

    # 5) manual 3D delay‐embedding of x_pred
    max_idx = x_pred.shape[0] - 2*lag_steps
    X1p     = x_pred[                  :max_idx]
    X2p     = x_pred[lag_steps         :lag_steps+max_idx]
    X3p     = x_pred[2*lag_steps       :2*lag_steps+max_idx]

    # also align true Lorenz
    true3   = X[n_delays : n_delays+max_idx, :]

    # 6) plot side‐by‐side
    fig = plt.figure(figsize=(12,5))

    # true 3D Lorenz
    axT = fig.add_subplot(1, 2, 1, projection="3d")
    axT.plot(true3[:,0], true3[:,1], true3[:,2], '-b', lw=0.5)
    axT.set_title("True Lorenz Attractor")
    axT.set_xlabel("x₁"); axT.set_ylabel("x₂"); axT.set_zlabel("x₃")

    # HAVOK‐reconstructed 3D
    axH = fig.add_subplot(1, 2, 2, projection="3d")
    axH.plot(X1p, X2p, X3p, '-r', lw=0.5)
    axH.set_title("HAVOK‐Reconstructed Attractor")
    axH.set_xlabel(r'$x(t)$')
    axH.set_ylabel(r'$x(t+\tau)$')
    axH.set_zlabel(r'$x(t+2\tau)$')

    plt.tight_layout()
    st.pyplot(fig)
