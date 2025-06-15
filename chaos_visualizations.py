# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from pykoopman.common import lorenz

st.title("Lorenz Attractors")

# — Sidebar controls —
st.sidebar.header("Attractor 1 Initial Conditions")
x0_1 = st.sidebar.slider("x₀ (A1)", -20.0, 20.0, 1.0, 0.1)
y0_1 = st.sidebar.slider("y₀ (A1)", -20.0, 20.0, 1.0, 0.1)
z0_1 = st.sidebar.slider("z₀ (A1)", -20.0, 20.0, 1.0, 0.1)

st.sidebar.header("Attractor 2 Initial Conditions")
x0_2 = st.sidebar.slider("x₀ (A2)", -20.0, 20.0, 1.1, 0.1)
y0_2 = st.sidebar.slider("y₀ (A2)", -20.0, 20.0, 1.0, 0.1)
z0_2 = st.sidebar.slider("z₀ (A2)", -20.0, 20.0, 1.0, 0.1)

# — Time grid —
t_final = st.sidebar.number_input("t_final", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
n_steps = st.sidebar.slider("n_steps", 100, 10000, 5000, 100)
t = np.linspace(0, t_final, n_steps)

# — Integrate both trajectories —
traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)

# — Plot —
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], label="Attractor 1", lw=1.5)
ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], label="Attractor 2", alpha=0.7, lw=1.5)
ax.set_title("Lorenz Attractors: Sensitive Dependence")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend()
st.pyplot(fig)
