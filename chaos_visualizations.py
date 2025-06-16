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

# Must be first
st.set_page_config(layout="wide", page_title="Lorenz & HAVOK Explorer")
st.title("Lorenz & HAVOK Explorer")

# Module selector
module = st.sidebar.radio(
    "Select Module",
    ["Attractors & Divergence", "HAVOK Reconstruction"],
    key="which_module"
)

if module == "Attractors & Divergence":
    # … (unchanged attractor & divergence code) …
    # [omitted for brevity]
    pass

else:
    # -- Sidebar controls --
    st.sidebar.header("HAVOK: Initial Conditions")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01, key="x0_h")
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01, key="y0_h")
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01, key="z0_h")

    st.sidebar.header("HAVOK Settings")
    dt        = st.sidebar.number_input("dt", 1e-4, 1.0, 0.01, 1e-4, format="%.4f", key="dt_havok")
    tau       = st.sidebar.number_input("Time delay τ", min_value=dt, max_value=10.0,
                                        value=1.0, step=dt, format="%.4f", key="tau_havok")
    embed_dim = st.sidebar.number_input("Embedding dimension m (≥3)", min_value=3, max_value=100,
                                        value=10, step=1, key="m_havok")
    t_final_h = st.sidebar.number_input("Total time for HAVOK", 1.0, 200.0, 50.0, 1.0, key="t_final_havok")

    # -- Generate training data --
    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x_series = X[:, 0]
    N        = len(x_series)

    # -- Build & fit HAVOK with row‐delay = tau/dt steps --
    n_delays    = embed_dim - 1
    delay_steps = max(1, int(tau / dt))
    TDC   = pk.observables.TimeDelay(delay=delay_steps, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1, 1), dt=dt)

    # -- Prepare warm‐up and simulation vectors --
    warmup = delay_steps * n_delays
    if warmup + 1 > N:
        st.error(f"Need at least {warmup+1} data points, but only have {N}.")
        st.stop()

    x0_e  = x_series[: warmup + 1].reshape(-1, 1)
    t_sim = t_h[warmup:] - t_h[warmup]
    u_sim = model.regressor.forcing_signal.reshape(-1, 1)

    # -- Run simulation --
    x_pred = model.simulate(x0_e, t_sim, u_sim).flatten()

    # -- Build full delay‐embedding of x_pred --
    emb_pred = TDC.transform(x_pred.reshape(-1, 1))  # shape = (len(x_pred)-warmup, m)
    xs, ys, zs = emb_pred[:, 0], emb_pred[:, 1], emb_pred[:, 2]

    # -- Interactive 3D reconstruction --
    trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='firebrick', width=2),
        name='HAVOK Reconstructed'
    )
    fig3 = go.Figure(data=[trace])
    fig3.update_layout(
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title=f'x(t–{delay_steps}·dt)',
            zaxis_title=f'x(t–2·{delay_steps}·dt)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="HAVOK Reconstructed Time‑Delay Attractor"
    )
    st.plotly_chart(fig3, use_container_width=True)
