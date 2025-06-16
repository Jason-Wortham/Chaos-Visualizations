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
    # … your divergence code unchanged …
    pass

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

    # integrate the Lorenz system
    t_h      = np.arange(0, t_final_h, dt)
    X        = integrate.odeint(lorenz, [x0, y0, z0], t_h)
    x_series = X[:, 0]

    # compute number of delay steps = τ / dt
    n_delays = max(1, int(tau / dt))

    # build a TimeDelay observable with delay=1 step
    TDC = pk.observables.TimeDelay(delay=1, n_delays=n_delays)
    Diff  = pk.differentiation.Derivative(kind="finite_difference", k=2)
    HAVOK = pk.regression.HAVOK(svd_rank=n_delays, plot_sv=False)

    model = pk.KoopmanContinuous(
        observables=TDC,
        differentiator=Diff,
        regressor=HAVOK
    )
    model.fit(x_series.reshape(-1, 1), dt=dt)

    # simulate using the first (n_delays+1) points as warm‑up
    x0_embed = x_series[: n_delays + 1].reshape(-1, 1)
    t_sim    = t_h[n_delays:] - t_h[n_delays]
    u_sim    = model.regressor.forcing_signal[n_delays:]
    x_pred   = model.simulate(x0_embed, t_sim, u_sim).flatten()

    # plot
    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(t_h[n_delays:], x_series[n_delays:], "-b", label="Original x")
    plt.plot(t_sim,           x_pred,            "--r", label="HAVOK Reconstructed")
    plt.xlabel("Time"); plt.ylabel("x")
    plt.title(f"HAVOK (τ={tau:.2f}, m={embed_dim}, dt={dt:.3f})")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig3)
