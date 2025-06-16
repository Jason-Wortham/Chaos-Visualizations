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
    ["Attractors & Divergence", "HAVOK Reconstruction", "EDMD Reconstruction"],
    key="which_module"
)

# 3) Attractors & Divergence
if module == "Attractors & Divergence":
    # … your existing two‑attractor code …

# 4) HAVOK Reconstruction
elif module == "HAVOK Reconstruction":
    # … your existing HAVOK code …

# 5) EDMD Reconstruction
else:
    st.sidebar.header("EDMD: Initial Conditions & Settings")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01)

    dt_edmd      = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_edmd = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, 1.0)
    max_steps    = max(1, int(t_final_edmd/dt_edmd) - 1)
    lag_steps    = st.sidebar.number_input(
        "Time delay (steps of dt)",
        min_value=1,
        max_value=max_steps,
        value=min(30, max_steps),
        step=1,
        format="%d"
    )
    svd_rank_edmd = st.sidebar.number_input(
        "EDMD SVD rank",
        min_value=1,
        max_value=3,
        value=3,
        step=1,
        format="%d"
    )

    # simulate the “true” Lorenz
    t_eval = np.arange(0, t_final_edmd, dt_edmd)
    X      = integrate.odeint(
        lorenz,
        [x0, y0, z0],
        t_eval,
        atol=1e-12,
        rtol=1e-12
    )  # shape = (N, 3)
    N = X.shape[0]
    if N < 2:
        st.error("Need at least 2 time points for EDMD."); st.stop()

    # fit EDMD on one-step snapshots
    edmd_model = pk.Koopman(
        regressor=pk.regression.EDMD(svd_rank=svd_rank_edmd)
    )
    edmd_model.fit(X[:-1], X[1:])

    # one‑step predictions for the whole trajectory
    X1_pred = edmd_model.predict(X[:-1])  # shape = (N-1, 3)
    X_pred  = np.vstack([X[0:1], X1_pred])  # prepend the true initial state

    # 5d) Interactive 3D plot of the predicted state (x,y,z)
    trace_state = go.Scatter3d(
        x=X_pred[:, 0],
        y=X_pred[:, 1],
        z=X_pred[:, 2],
        mode='lines',
        line=dict(color='red', width=2),
        name='EDMD (x,y,z)'
    )
    fig_state = go.Figure([trace_state])
    fig_state.update_layout(
        scene=dict(
            xaxis_title='x_pred',
            yaxis_title='y_pred',
            zaxis_title='z_pred'
        ),
        title='EDMD‑Predicted Lorenz State',
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_state, use_container_width=True)

    # 5e) Build the 3‑coordinate delay‑embedding of x_pred
    max_idx = N - 2*lag_steps
    ED1 = X_pred[          :max_idx, 0]
    ED2 = X_pred[lag_steps :lag_steps+max_idx, 0]
    ED3 = X_pred[2*lag_steps:2*lag_steps+max_idx, 0]

    # 5f) Interactive 3D delay‑embedding plot
    trace_emb = go.Scatter3d(
        x=ED1,
        y=ED2,
        z=ED3,
        mode='lines',
        line=dict(color='green', width=2),
        name='EDMD delay‑embed'
    )
    fig_emb = go.Figure([trace_emb])
    fig_emb.update_layout(
        scene=dict(
            xaxis_title='x_pred(t)',
            yaxis_title=f'x_pred(t+{lag_steps}·dt)',
            zaxis_title=f'x_pred(t+{2*lag_steps}·dt)'
        ),
        title='EDMD Delay‑Embedding (m=3)',
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_emb, use_container_width=True)

