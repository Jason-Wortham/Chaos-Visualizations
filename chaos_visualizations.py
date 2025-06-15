# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, clear_output

from pykoopman.common import lorenz

t_final = 50.0
n_steps = 5000
t = np.linspace(0, t_final, n_steps)

slider_x0 = widgets.FloatSlider(min=-20, max=20, step=0.1, value=1.0, description='x₀:')
slider_y0 = widgets.FloatSlider(min=-20, max=20, step=0.1, value=1.0, description='y₀:')
slider_z0 = widgets.FloatSlider(min=-20, max=20, step=0.1, value=1.0, description='z₀:')
slider_delta = widgets.FloatSlider(min=1e-6, max=1.0, step=1e-6, value=1e-3, description='Δ:')

ui = widgets.VBox([slider_x0, slider_y0, slider_z0, slider_delta])

def plot_lorenz(x0, y0, z0, delta):
    clear_output(wait=True)
    # Base and perturbed initial conditions
    x0_1 = [x0, y0, z0]
    x0_2 = [x0 + delta, y0 + delta, z0 + delta]
    # Integrate both trajectories
    traj1 = integrate.odeint(lorenz, x0_1, t)
    traj2 = integrate.odeint(lorenz, x0_2, t)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], label='Traj 1')
    ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], label='Traj 2', alpha=0.7)
    ax.set_title('Lorenz Attractor: Sensitive Dependence on Initial Conditions')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.legend()
    plt.show()

out = widgets.interactive_output(
    plot_lorenz,
    {'x0': slider_x0, 'y0': slider_y0, 'z0': slider_z0, 'delta': slider_delta}
)

display(ui, out)
