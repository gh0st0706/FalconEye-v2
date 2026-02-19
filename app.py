import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="FalconEye Flight Diagnostic",
    layout="wide"
)

# --------------------------------------------------
# F-16 HUD STYLE
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #000000;
    color: #00ff66;
}

html, body, [class*="css"] {
    font-family: 'Courier New', monospace;
    color: #00ff66;
}

h1, h2, h3 {
    color: #00ff66;
    letter-spacing: 2px;
}

section[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #00ff66;
}

div[data-testid="metric-container"] {
    background-color: #000000;
    border: 1px solid #00ff66;
    padding: 10px;
}

.stButton>button {
    background-color: #000000;
    color: #00ff66;
    border: 1px solid #00ff66;
    border-radius: 0px;
}

div, button {
    border-radius: 0px !important;
}

hr {
    border: 1px solid #00ff66;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center;'>
FALCONEYE // FLIGHT DIAGNOSTIC SYSTEM
</h1>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("MISSION PARAMETERS")

num_points = st.sidebar.slider("Telemetry Samples", 200, 2000, 500)
anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 2.0, 5.0, 3.0)

# --------------------------------------------------
# SYNTHETIC TELEMETRY
# --------------------------------------------------

np.random.seed(42)

time = np.arange(num_points)

engine_temp = 500 + np.sin(time / 30) * 20 + np.random.normal(0, 5, num_points)
anomaly_score = np.abs(np.random.normal(1, 0.5, num_points))

# Inject random spikes
spike_indices = np.random.choice(num_points, size=int(num_points * 0.05), replace=False)
engine_temp[spike_indices] += np.random.uniform(50, 100, len(spike_indices))
anomaly_score[spike_indices] += np.random.uniform(3, 5, len(spike_indices))

data = pd.DataFrame({
    "time": time,
    "engine_temp": engine_temp,
    "anomaly_score": anomaly_score
})

anomalies = data[data["anomaly_score"] > anomaly_threshold]

# --------------------------------------------------
# METRICS PANEL
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("TOTAL SAMPLES", num_points)
col2.metric("TOTAL ANOMALIES", len(anomalies))
col3.metric("SYSTEM STATUS", "STABLE" if len(anomalies) < num_points * 0.1 else "WARNING")

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# ENGINE TEMP PLOT
# --------------------------------------------------

st.markdown("### ENGINE TEMPERATURE MONITOR")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["time"],
    y=data["engine_temp"],
    mode="lines",
    name="ENGINE TEMP"
))

fig.add_trace(go.Scatter(
    x=anomalies["time"],
    y=anomalies["engine_temp"],
    mode="markers",
    name="ANOMALY",
    marker=dict(size=8, symbol="x")
))

fig.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="#00ff66", family="Courier New"),
    xaxis=dict(
        title="TIME",
        showgrid=True,
        gridcolor="#003300",
        zeroline=False
    ),
    yaxis=dict(
        title="ENGINE TEMP",
        showgrid=True,
        gridcolor="#003300",
        zeroline=False
    ),
    legend=dict(font=dict(color="#00ff66"))
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# ANOMALY SCORE PLOT
# --------------------------------------------------

st.markdown("### ANOMALY SCORE ANALYSIS")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data["time"],
    y=data["anomaly_score"],
    mode="lines",
    name="ANOMALY SCORE"
))

fig2.add_hline(
    y=anomaly_threshold,
    line_dash="dash"
)

fig2.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="#00ff66", family="Courier New"),
    xaxis=dict(
        title="TIME",
        showgrid=True,
        gridcolor="#003300",
        zeroline=False
    ),
    yaxis=dict(
        title="SCORE",
        showgrid=True,
        gridcolor="#003300",
        zeroline=False
    ),
    legend=dict(font=dict(color="#00ff66"))
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("FALCONEYE FLIGHT ANALYTICS MODULE // READY")
