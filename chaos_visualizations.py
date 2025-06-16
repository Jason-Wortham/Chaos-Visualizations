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
import plotly.graph_objs as go

# 1) Must be first Streamlit command
st.set_page_config(layout="wide", page_title="Lorenz & HAVOK Explorer")
st.title("Lorenz & HAVOK Explorer")

# 2) Module selector
module = st.sidebar.radio(
    "Select Module",
    ["Attractors & Divergence", "HAVOK Reconstruction"],
    key="which_module"
)

# 3) Attractors & Divergence branch
if module == "Attractors & Divergence":
    # Sidebar controls
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

    # Compute trajectories
    t     = np.linspace(0, t_final, n_steps)
    traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
    traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    # Interactive 3D plot with Plotly
    trace1 = go.Scatter3d(
        x=traj1[:,0], y=traj1[:,1], z=traj1[:,2],
        mode='lines', line=dict(color='blue', width=2),
        name='Attractor 1'
    )
    trace2 = go.Scatter3d(
        x=traj2[:,0], y=traj2[:,1], z=traj2[:,2],
        mode='lines', line=dict(color='green', width=2),
        name='Attractor 2'
    )
    fig1 = go.Figure(data=[trace1, trace2])
    fig1.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Lorenz Attractors"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Static divergence vs time
    fig2 = plt.figure(figsize=(8, 3))
    plt.plot(t, dist, color="red", lw=2)
    plt.xlabel("Time"); plt.ylabel("‖X₁(t)–X₂(t)‖")
    plt.title("Distance Between Trajectories Over Time")
    plt.tight_layout()
    st.pyplot(fig2)

# 4) HAVOK Reconstruction branch
else:
    # Sidebar controls
    st.sidebar.header("HAVOK: Initial Conditions")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01, key="x0_h")
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01, key="y0_h")
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01, key="z0_h")

    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input(
        "dt", 1e-4, 1.0, 0.01, 1e-4, format="%.4f", key="dt_havok"
    )
    tau       = st.sidebar.number_input(
        "Time delay τ", min_value=dt, max_value=10.0, value=1.0,
        step=dt, format="%.4f", key="tau_havok"
    )
    embed_dim = st.sidebar.number_input(
        "Embedding dimension m (≥3)", min_value=3, max_value=100,
        value=10, step=1, key="m_havok"
    )
    t_final_h = st.sidebar.number_input(
        "Total time for HAVOK", 1.0, 200.0, 50.0, 1.0, key="t_final_havok"
    )

    # Simulate for training
    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x_series = X[:, 0]

    # Build & fit the HAVOK model
    n_delays = embed_dim - 1
    TDC   = pk.observables.TimeDelay(delay=1, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1, 1), dt=dt)

    # Simulate forward
    u_full = model.regressor.forcing_signal.reshape(-1, 1)
    f      = u_full.shape[0]
    d      = n_delays
    x0_e   = x_series[: d+1].reshape(-1, 1)
    t_sim  = t_h[d : d+f] - t_h[d]
    u_sim  = u_full[:f]
    x_pred = model.simulate(x0_e, t_sim, u_sim).flatten()

    # Build the full m‑dim delay embedding
    emb_pred = TDC.transform(x_pred.reshape(-1, 1))  # shape = (f, m)
    xs, ys, zs = emb_pred[:, 0], emb_pred[:, 1], emb_pred[:, 2]

    # Interactive 3D reconstruction
    trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines', line=dict(color='firebrick', width=2),
        name='HAVOK Reconstructed'
    )
    fig3 = go.Figure(data=[trace])
    fig3.update_layout(
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title=f'x(t – {1*dt:.3f})',
            zaxis_title=f'x(t – {2*dt:.3f})'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="HAVOK Reconstructed Time‑Delay Attractor"
    )
    st.plotly_chart(fig3, use_container_width=True)
