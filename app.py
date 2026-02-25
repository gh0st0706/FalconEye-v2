import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="FalconEye Flight Diagnostic",
    page_icon="Untitled design (2).png",
    layout="wide"
)
st.image("Untitled design (2).png", width=200)

# --------------------------------------------------
# F-16 HUD STYLE
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #000000;
    color: #00ff66;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
    color: #00ff66;
}

h1, h2, h3 {
    color: #00ff66;
    letter-spacing: 2px;
    font-family: 'Orbitron', 'Rajdhani', sans-serif;
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
<h1 style='text-align:center; font-family: Orbitron, Rajdhani, sans-serif;'>
FALCONEYE // FLIGHT DIAGNOSTIC SYSTEM
</h1>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("MISSION PARAMETERS")

uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])

num_points = st.sidebar.slider(
    "Telemetry Samples",
    200,
    2000,
    500,
    disabled=uploaded_file is not None
)
anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 2.0, 5.0, 3.0)

# --------------------------------------------------
# DATA SOURCE (UPLOADED CSV OR SYNTHETIC)
# --------------------------------------------------

data = None
source_label = "SYNTHETIC"

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        normalized = {col.strip().lower(): col for col in raw_data.columns}

        required_cols = ["engine_temp", "anomaly_score"]
        missing_cols = [col for col in required_cols if col not in normalized]

        if missing_cols:
            st.sidebar.error(
                "CSV missing required columns: " + ", ".join(missing_cols)
            )
        else:
            selected = pd.DataFrame({
                "engine_temp": pd.to_numeric(raw_data[normalized["engine_temp"]], errors="coerce"),
                "anomaly_score": pd.to_numeric(raw_data[normalized["anomaly_score"]], errors="coerce")
            })

            if "time" in normalized:
                selected["time"] = pd.to_numeric(raw_data[normalized["time"]], errors="coerce")
            else:
                selected["time"] = np.arange(len(selected))

            data = selected[["time", "engine_temp", "anomaly_score"]].dropna()

            if data.empty:
                st.sidebar.error("CSV loaded but no valid numeric rows were found.")
            else:
                source_label = "UPLOADED CSV"
                st.sidebar.success(f"Loaded {len(data)} rows from CSV")

    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")

if data is None:
    np.random.seed(42)
    time = np.arange(num_points)
    engine_temp = 500 + np.sin(time / 30) * 20 + np.random.normal(0, 5, num_points)
    anomaly_score = np.abs(np.random.normal(1, 0.5, num_points))

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

col1, col2, col3, col4 = st.columns(4)

col1.metric("TOTAL SAMPLES", len(data))
col2.metric("TOTAL ANOMALIES", len(anomalies))
col3.metric("SYSTEM STATUS", "STABLE" if len(anomalies) < len(data) * 0.1 else "WARNING")
col4.metric("DATA SOURCE", source_label)

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
    font=dict(color="#00ff66", family="Rajdhani"),
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
    font=dict(color="#00ff66", family="Rajdhani"),
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
