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
    trace1 = go.Scatter3d(x=traj1[:,0], y=traj1[:,1], z=traj1[:,2],
                          mode='lines', line=dict(color='blue', width=2),
                          name='Attractor 1')
    trace2 = go.Scatter3d(x=traj2[:,0], y=traj2[:,1], z=traj2[:,2],
                          mode='lines', line=dict(color='green', width=2),
                          name='Attractor 2')
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
    plt.title("Distance Between Trajectories Over Time")
    plt.tight_layout()
    st.pyplot(fig2)

# 4) HAVOK Reconstruction branch
else:
    st.sidebar.header("HAVOK: Initial Conditions")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01)

    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_h = st.sidebar.number_input("Total time for HAVOK", 1.0, 200.0, 20.0, 1.0)
    max_steps = max(1, int(t_final_h / dt) - 1)
    tau_steps = st.sidebar.number_input(
        "Time delay (steps of dt)", min_value=1, max_value=max_steps,
        value=30, step=1, format="%d"
    )
    embed_dim = st.sidebar.number_input(
        "Embedding dimension m (≥3)", min_value=3, max_value=200,
        value=100, step=1, format="%d"
    )

    # simulate “true” Lorenz
    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h,
                                atol=1e-12, rtol=1e-12)
    x_series = X[:, 0]
    N        = len(x_series)

    # build HAVOK with row-delay = tau_steps
    n_delays = embed_dim - 1
    delay    = tau_steps
    TDC   = pk.observables.TimeDelay(delay=delay, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind='finite_difference', k=2)
    max_delays = min(n_delays, N - (delay * n_delays))
    svd_rank = int(st.sidebar.number_input("SVD rank", 2, n_delays, max_delays, 1))
    HAVOK = pk.regression.HAVOK(svd_rank=svd_rank, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1,1), dt=dt)

    # seed and simulate
    warmup = delay * n_delays
    if warmup + 1 > N:
        st.error(f"Need at least {warmup+1} points, have {N}.")
        st.stop()

    seed   = x_series[:warmup+1].reshape(-1,1)
    t_sim  = t_h[warmup:] - t_h[warmup]
    u_full = model.regressor.forcing_signal.reshape(-1,1)
    u_sim  = u_full[:len(t_sim)]
    x_pred = model.simulate(seed, t_sim, u_sim).flatten()

    # manual 3D embedding with lag = tau_steps
    d = 3
    lag = delay
    T_eff    = len(x_pred)
    max_idx  = T_eff - (d-1)*lag
    X1 = x_pred[             :max_idx]
    X2 = x_pred[lag          :lag+max_idx]
    X3 = x_pred[2*lag        :2*lag+max_idx]

    # interactive 3D reconstruction
    trace = go.Scatter3d(
        x=X1, y=X2, z=X3,
        mode='lines', line=dict(color='firebrick', width=2),
        name='HAVOK Reconstructed'
    )
    fig3 = go.Figure([trace])
    fig3.update_layout(
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title=f'x(t+{lag}·dt)',
            zaxis_title=f'x(t+{2*lag}·dt)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="HAVOK Reconstructed Time‑Delay Attractor"
    )
    st.plotly_chart(fig3, use_container_width=True)

