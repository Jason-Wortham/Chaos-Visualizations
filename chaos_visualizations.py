# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:40:36 2025

@author: Jason
"""

# …[all the imports and first two modules unchanged]…

# 5) dmd Reconstruction
else:  # module == "dmd Reconstruction"
    st.sidebar.header("dmd: Initial Conditions & Settings")
    x0 = st.sidebar.slider("x₀", -10.0, 10.0, 1.0, 0.01)
    y0 = st.sidebar.slider("y₀", -10.0, 10.0, 1.0, 0.01)
    z0 = st.sidebar.slider("z₀", -10.0, 10.0, 1.0, 0.01)

    dt_dmd      = st.sidebar.number_input("dt", 1e-4, 1.0, 0.001, 1e-4, format="%.4f")
    t_final_dmd = st.sidebar.number_input("Total time", 1.0, 200.0, 20.0, 1.0)

    # simulate true Lorenz
    t_eval = np.arange(0, t_final_dmd, dt_dmd)
    X      = integrate.odeint(lorenz, [x0, y0, z0], t_eval,
                              atol=1e-12, rtol=1e-12)
    N = X.shape[0]
    if N < 2:
        st.error("Need at least 2 time points."); st.stop()

    # fit DMD
    dmd_model = pk.Koopman(regressor=pk.regression.dmd(svd_rank=3))
    dmd_model.fit(X[:-1], X[1:])

    # iterate one-step predictions forward
    X_pred = np.zeros_like(X)
    X_pred[0] = X[0]
    for k in range(1, N):
        X_pred[k] = dmd_model.predict(X_pred[k-1].reshape(1,-1))[0]

    # only plot the full (x,y,z) DMD result
    st.subheader("DMD‑Predicted Lorenz State")
    trace_s = go.Scatter3d(
        x=X_pred[:,0],
        y=X_pred[:,1],
        z=X_pred[:,2],
        mode='lines',
        line=dict(color='red', width=2),
    )
    fig_s = go.Figure([trace_s])
    fig_s.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_s, use_container_width=True)


