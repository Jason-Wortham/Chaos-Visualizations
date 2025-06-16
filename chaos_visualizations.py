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
import plotly.graph_objs as go

# 1) Must be first Streamlit command
st.set_page_config(layout="wide", page_title="Lorenz & Koopman Explorer")
st.title("Lorenz & Koopman Explorer")

# 2) Module selector
module = st.sidebar.radio(
    "Select Module",
    ["Attractors & Divergence", "DMD Reconstruction"],
    key="which_module",
)

# 3) Attractors & Divergence
if module == "Attractors & Divergence":
    st.sidebar.header("Attractor 1 Initial Conditions")
    x0_1 = st.sidebar.slider("x₀ (A1)", -10.0, 10.0, 1.0, 0.01)
    y0_1 = st.sidebar.slider("y₀ (A1)", -10.0, 10.0, 1.0, 0.01)
    z0_1 = st.sidebar.slider("z₀ (A1)", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("Attractor 2 Initial Conditions")
    x0_2 = st.sidebar.slider("x₀ (A2)", -10.0, 10.0, 1.0, 0.01)
    y0_2 = st.sidebar.slider("y₀ (A2)", -10.0, 10.0, 1.0, 0.01)
    z0_2 = st.sidebar.slider("z₀ (A2)", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("Divergence Settings")
    t_final = st.sidebar.number_input("t_final", 1.0, 100.0, 50.0, 1.0)
    n_steps = st.sidebar.slider("n_steps", 100, 10_000, 5_000, 100)

    # integrate two Lorenz trajectories
    t     = np.linspace(0, t_final, n_steps)
    traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
    traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    # interactive 3D plot
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
    fig1 = go.Figure([trace1, trace2])
    fig1.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Lorenz Attractors"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # static divergence plot
    fig2 = plt.figure(figsize=(8,3))
    plt.plot(t, dist, color='red', lw=2)
    plt.xlabel("Time"); plt.ylabel("‖X₁(t)–X₂(t)‖")
    plt.title("Distance Between Trajectories")
    plt.tight_layout()
    st.pyplot(fig2)

# 4) DMD Reconstruction
else:
    st.sidebar.header("DMD: Initial Conditions & Settings")

    # — state ICs —
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01)

    # — time settings —
    dt_dmd      = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_dmd = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, 1.0)
    t_eval      = np.arange(0, t_final_dmd, dt_dmd)

    # integrate the true Lorenz trajectory
    X = integrate.odeint(
        lorenz, [x0, y0, z0], t_eval,
        atol=1e-12, rtol=1e-12
    )
    N = X.shape[0]
    if N < 2:
        st.error("Need at least 2 time points."); st.stop()

    # choose DMD rank
    max_rank = N - 1
    dmd_rank = st.sidebar.slider(
        "DMD SVD rank", 1, max_rank, min(20, max_rank), 1
    )

    # build & fit plain‐vanilla DMD (using the PyDMDRegressor)
    dmd_reg = pk.regression.PyDMDRegressor(svd_rank=dmd_rank)
    dmd_model = pk.Koopman(regressor=dmd_reg)
    dmd_model.fit(X[:-1], X[1:])

    # one‐step iterate to get a forecast
    X_pred = np.zeros_like(X)
    X_pred[0] = X[0]
    for k in range(1, N):
        X_pred[k] = dmd_model.predict(X_pred[k-1].reshape(1, -1))[0]

    # plot the DMD‐predicted trajectory in (x,y,z)
    st.subheader("DMD‑Predicted Lorenz State")
    trace = go.Scatter3d(
        x=X_pred[:,0],
        y=X_pred[:,1],
        z=X_pred[:,2],
        mode='lines',
        line=dict(color='crimson', width=2),
        name='DMD'
    )
    fig = go.Figure([trace])
    fig.update_layout(
        scene=dict(
            xaxis_title='x_pred',
            yaxis_title='y_pred',
            zaxis_title='z_pred'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="DMD Reconstructed Lorenz Attractor"
    )
    st.plotly_chart(fig, use_container_width=True)


