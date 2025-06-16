# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pykoopman.common import lorenz

st.title("Lorenz Attractors & Divergence")

st.sidebar.header("Attractor 1 Initial Conditions")
x0_1 = st.sidebar.slider("x_0 (A1)", -10.0, 10.0, 1.0, 0.01)
y0_1 = st.sidebar.slider("y_0 (A1)", -10.0, 10.0, 1.0, 0.01)
z0_1 = st.sidebar.slider("z_0 (A1)", -10.0, 10.0, 1.0, 0.01)

st.sidebar.header("Attractor 2 Initial Conditions")
x0_2 = st.sidebar.slider("x_0 (A2)", -10.0, 10.0, 1.0, 0.01)
y0_2 = st.sidebar.slider("y_0 (A2)", -10.0, 10.0, 1.0, 0.01)
z0_2 = st.sidebar.slider("z_0 (A2)", -10.0, 10.0, 1.0, 0.01)

t_final = st.sidebar.number_input("t_final", min_value = 1.0, max_value = 100.0, value = 50.0, step = 1.0)
n_steps = st.sidebar.slider("n_steps", 100, 10000, 5000, 100)
t = np.linspace(0, t_final, n_steps)

traj1 = integrate.odeint(lorenz, [x0_1, y0_1, z0_1], t)
traj2 = integrate.odeint(lorenz, [x0_2, y0_2, z0_2], t)

dist = np.linalg.norm(traj2 - traj1, axis = 1)

fig1 = plt.figure(figsize = (8, 6))
ax = fig1.add_subplot(111, projection = '3d')
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], color = "blue", label = "Attractor 1", lw = 1.5)
ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], color = "green", label = "Attractor 2", lw = 1.5)
ax.set_title("Lorenz Attractors")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend()
st.pyplot(fig1)

fig2 = plt.figure(figsize = (8, 3))
plt.plot(t, dist, color = "red", lw = 2)
plt.title("Distance Between Trajectories Over Time")
plt.xlabel("Time")
plt.ylabel("‖X_1(t) – X_2(t)‖")
plt.tight_layout()
st.pyplot(fig2)
