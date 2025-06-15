# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

# chaos_visualizations.py
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from pykoopman.common import lorenz

st.title("Lorenz Attractors")

x0 = st.sidebar.slider("x₀", -20.0, 20.0, 1.0, 0.1)
y0 = st.sidebar.slider("y₀", -20.0, 20.0, 1.0, 0.1)
z0 = st.sidebar.slider("z₀", -20.0, 20.0, 1.0, 0.1)
delta = st.sidebar.slider("Δ perturbation", 1e-6, 1.0, 1e-3, 1e-6)

t = np.linspace(0, 50, 5000)
traj1 = integrate.odeint(lorenz, [x0, y0, z0], t)
traj2 = integrate.odeint(lorenz, [x0+delta, y0+delta, z0+delta], t)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], label="Base")
ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], label="Perturbed", alpha=0.7)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend()
st.pyplot(fig)

