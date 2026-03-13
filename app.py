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
# AVIONICS UI STYLE
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Saira+Condensed:wght@500;600&display=swap');

:root {
    --bg: #0c0f12;
    --panel: #11161c;
    --panel-2: #0f141a;
    --text: #e6e9ee;
    --muted: #a6b0bb;
    --line: #27313b;
    --accent: #8fb3c9;
    --warn: #ffb000;
    --crit: #ff5c5c;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Segoe UI', sans-serif;
    color: var(--text);
}

h1, h2, h3, h4 {
    color: var(--accent);
    letter-spacing: 1px;
    font-family: 'Saira Condensed', 'IBM Plex Mono', sans-serif;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: var(--panel-2);
    border-right: 1px solid var(--line);
}

div[data-testid="metric-container"] {
    background-color: var(--panel);
    border: 1px solid var(--line);
    padding: 12px 14px;
}

div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    letter-spacing: 0.5px;
}

.stButton>button {
    background-color: var(--panel);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 2px;
}

input, textarea, select, .stSlider {
    color: var(--text) !important;
}

div, button {
    border-radius: 2px !important;
}

hr {
    border: 0;
    border-top: 1px solid var(--line);
}

.title-block {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.title-main {
    font-size: 30px;
    letter-spacing: 2px;
}

.title-sub {
    color: var(--muted);
    font-size: 14px;
    letter-spacing: 1px;
}

.status-chip {
    display: inline-block;
    padding: 6px 10px;
    border: 1px solid var(--line);
    background: var(--panel);
    font-size: 12px;
    letter-spacing: 1px;
}

.status-ok {
    color: var(--text);
}

.status-warn {
    color: var(--warn);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="title-block">
  <div class="title-main">FALCONEYE FLIGHT DIAGNOSTIC</div>
  <div class="title-sub">ENGINE THERMAL MONITORING & ANOMALY DETECTION</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("PARAMETERS")

num_points = st.sidebar.slider("Telemetry Samples", 200, 2000, 500)
anomaly_threshold = st.sidebar.slider("Anomaly Z-Score Threshold", 2.0, 5.0, 3.0)

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

anomaly_rate = (len(anomalies) / num_points) * 100
peak_temp = float(np.max(engine_temp))
mean_temp = float(np.mean(engine_temp))
status_label = "STABLE" if len(anomalies) < num_points * 0.1 else "CAUTION"
status_class = "status-ok" if status_label == "STABLE" else "status-warn"

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("TOTAL SAMPLES", num_points)
col2.metric("TOTAL ANOMALIES", len(anomalies))
col3.metric("ANOMALY RATE (%)", f"{anomaly_rate:.2f}")
col4.metric("PEAK TEMP (C)", f"{peak_temp:.1f}")
col5.metric("MEAN TEMP (C)", f"{mean_temp:.1f}")

st.markdown(
    f"<div class='status-chip {status_class}'>SYSTEM STATUS: {status_label}</div>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# ENGINE TEMP PLOT
# --------------------------------------------------

st.markdown("### ENGINE TEMPERATURE (C)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["time"],
    y=data["engine_temp"],
    mode="lines",
    name="ENGINE TEMP",
    line=dict(color="#cfd6dd", width=1.5)
))

fig.add_trace(go.Scatter(
    x=anomalies["time"],
    y=anomalies["engine_temp"],
    mode="markers",
    name="ANOMALY",
    marker=dict(size=7, symbol="x", color="#ffb000")
))

fig.update_layout(
    paper_bgcolor="#0c0f12",
    plot_bgcolor="#0c0f12",
    font=dict(color="#e6e9ee", family="IBM Plex Mono"),
    xaxis=dict(
        title="TIME",
        showgrid=True,
        gridcolor="#1e252b",
        zeroline=False,
        showline=True,
        linecolor="#27313b",
        tickfont=dict(color="#a6b0bb")
    ),
    yaxis=dict(
        title="TEMP (C)",
        showgrid=True,
        gridcolor="#1e252b",
        zeroline=False,
        showline=True,
        linecolor="#27313b",
        tickfont=dict(color="#a6b0bb")
    ),
    legend=dict(font=dict(color="#a6b0bb"))
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --------------------------------------------------
# ANOMALY SCORE PLOT
# --------------------------------------------------

st.markdown("### ANOMALY Z-SCORE")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data["time"],
    y=data["anomaly_score"],
    mode="lines",
    name="ANOMALY Z-SCORE",
    line=dict(color="#cfd6dd", width=1.5)
))

fig2.add_hline(
    y=anomaly_threshold,
    line_dash="dash",
    line_color="#ffb000"
)

fig2.update_layout(
    paper_bgcolor="#0c0f12",
    plot_bgcolor="#0c0f12",
    font=dict(color="#e6e9ee", family="IBM Plex Mono"),
    xaxis=dict(
        title="TIME",
        showgrid=True,
        gridcolor="#1e252b",
        zeroline=False,
        showline=True,
        linecolor="#27313b",
        tickfont=dict(color="#a6b0bb")
    ),
    yaxis=dict(
        title="Z-SCORE",
        showgrid=True,
        gridcolor="#1e252b",
        zeroline=False,
        showline=True,
        linecolor="#27313b",
        tickfont=dict(color="#a6b0bb")
    ),
    legend=dict(font=dict(color="#a6b0bb"))
)

st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("FALCONEYE FLIGHT ANALYTICS MODULE // READY")
