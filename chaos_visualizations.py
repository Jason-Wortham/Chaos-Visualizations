# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

# resource: https://pypi.org/project/pykoopman/

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import streamlit as st
import pykoopman as pk
from pykoopman.common import lorenz
import plotly.graph_objs as go

st.set_page_config(layout="wide", page_title="Chaos and Reconstruction Explorer")
st.title("Chaos and Reconstruction Explorer")

module = st.sidebar.radio(
    "Select Module",
    ["Attractors and Divergence", "HAVOK Reconstruction", "DMD Reconstruction"],
    key="which_module",
)

if module == "Attractors and Divergence":
    st.sidebar.header("Attractor 1 Initial Conditions")
    x0_1 = st.sidebar.slider("x_0 (Attractor 1)", -10.0, 10.0, 1.0, 0.01)
    y0_1 = st.sidebar.slider("y_0 (Attractor 1)", -10.0, 10.0, 1.0, 0.01)
    z0_1 = st.sidebar.slider("z_0 (Attractor 1)", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("Attractor 2 Initial Conditions")
    x0_2 = st.sidebar.slider("x_0 (Attractor 2)", -10.0, 10.0, 1.0, 0.01)
    y0_2 = st.sidebar.slider("y_0 (Attractor 2)", -10.0, 10.0, 1.0, 0.01)
    z0_2 = st.sidebar.slider("z_0 (Attractor 2)", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("Divergence Settings")
    t_final = st.sidebar.number_input("t_final", 1.0, 100.0, 50.0, 1.0)
    n_steps = st.sidebar.slider("n_steps", 100, 10_000, 5_000, 100)

    t = np.linspace(0, t_final, n_steps)
    traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
    traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)
    dist  = np.linalg.norm(traj2 - traj1, axis=1)

    trace1 = go.Scatter3d(
        x=traj1[:,0], y=traj1[:,1], z=traj1[:,2],
        mode='lines', line=dict(color='blue', width=2), name='Attractor 1'
    )
    trace2 = go.Scatter3d(
        x=traj2[:,0], y=traj2[:,1], z=traj2[:,2],
        mode='lines', line=dict(color='green', width=2), name='Attractor 2'
    )
    fig1 = go.Figure([trace1, trace2])
    fig1.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Lorenz Attractors"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(t, dist, color='red', lw=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("‖X₁(t)–X₂(t)‖")
    ax2.set_title("Distance Between Trajectories")
    st.pyplot(fig2)

elif module == "HAVOK Reconstruction":
    st.sidebar.header("HAVOK: Initial Conditions")
    x0 = st.sidebar.slider("x_0", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y_0", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z_0", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_h = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, 1.0)
    max_steps = max(1, int(t_final_h / dt) - 1)
    tau_steps = st.sidebar.number_input("Time Delay", 1, max_steps, 17, 1, format="%d")
    embed_dim = st.sidebar.number_input("Embedding Dimension", 3, 100, 6, 1, format="%d")

    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h, atol=1e-12, rtol=1e-12)
    x_series = X[:,0]; N = len(x_series)

    delay    = tau_steps
    n_delays = embed_dim - 1
    warmup   = delay * n_delays
    if N < warmup+1:
        st.error(f"Need ≥{warmup+1} points, have {N}."); st.stop()
    effective = N - warmup
    max_svd   = min(n_delays, effective)

    svd_rank = st.sidebar.number_input("SVD rank", 1, max_svd, max_svd, 1, format="%d")

    TDC   = pk.observables.TimeDelay(delay=delay, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind='finite_difference', k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=svd_rank, plot_sv=False)
    model = pk.KoopmanContinuous(observables=TDC, differentiator=Diff, regressor=HAVOK)
    model.fit(x_series.reshape(-1,1), dt=dt)

    seed   = x_series[:warmup+1].reshape(-1,1)
    t_sim  = t_h[warmup:] - t_h[warmup]
    u_sim  = model.regressor.forcing_signal.reshape(-1,1)[:len(t_sim)]
    x_pred = model.simulate(seed, t_sim, u_sim).flatten()

    d, lag = 3, delay
    max_idx = len(x_pred) - (d-1)*lag
    X1 = x_pred[:max_idx]
    X2 = x_pred[ lag:lag+max_idx ]
    X3 = x_pred[2*lag:2*lag+max_idx]

    trace_h = go.Scatter3d(
        x=X1, y=X2, z=X3,
        mode='lines', line=dict(color='steelblue', width=2),
        name='HAVOK'
    )
    fig3 = go.Figure([trace_h])
    fig3.update_layout(
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title=f'x(t+{lag}·dt)',
            zaxis_title=f'x(t+{2*lag}·dt)',
            camera=dict(eye=dict(x=-1.5, y=0.45, z=0))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="HAVOK Reconstructed Time Delay Attractor"
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.sidebar.header("DMD: Initial Conditions & Settings")
    x0 = st.sidebar.slider("x_0", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y_0", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z_0", -10.0, 10.0, 1.0, 0.01)

    dt_dmd      = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_dmd = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, 1.0)

    t_eval = np.arange(0, t_final_dmd, dt_dmd)
    X      = integrate.odeint(lorenz, [x0, y0, z0], t_eval, atol=1e-12, rtol=1e-12)
    N = X.shape[0]
    if N < 2:
        st.error("Need at least 2 time points."); st.stop()

    dmd_model = pk.Koopman(regressor=pk.regression.EDMD(svd_rank=3))
    dmd_model.fit(X[:-1], X[1:])

    X_pred = np.zeros_like(X)
    X_pred[0] = X[0]
    for k in range(1, N):
        X_pred[k] = dmd_model.predict(X_pred[k-1].reshape(1,-1))[0]

    trace_s = go.Scatter3d(
        x=X_pred[:,0], y=X_pred[:,1], z=X_pred[:,2],
        mode='lines', line=dict(color='red', width=2),
    )
    fig_s = go.Figure([trace_s])
    fig_s.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            camera=dict(eye=dict(x=2, y=-1, z=0.2))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="DMD Predicted Lorenz State"
    )
    st.plotly_chart(fig_s, use_container_width=True)

