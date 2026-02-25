import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import EngineStateModel


def _build_live_features(df: pd.DataFrame, anomaly_threshold: float) -> pd.DataFrame:
    telemetry = df.copy().sort_values("time").reset_index(drop=True)

    if "rpm" not in telemetry.columns:
        egt_min = telemetry["engine_temp"].min()
        egt_max = telemetry["engine_temp"].max()
        if egt_max == egt_min:
            telemetry["rpm"] = 1100.0
        else:
            telemetry["rpm"] = 1100.0 + (telemetry["engine_temp"] - egt_min) * (15500.0 - 1100.0) / (egt_max - egt_min)

    if "vibration" not in telemetry.columns:
        if "anomaly_score" in telemetry.columns:
            telemetry["vibration"] = 0.08 + np.clip(telemetry["anomaly_score"], 0.0, 10.0) * 0.03
        else:
            telemetry["vibration"] = 0.10

    if "fuel_flow" not in telemetry.columns:
        telemetry["fuel_flow"] = 220.0 + 0.045 * telemetry["rpm"]

    dt = telemetry["time"].diff().replace(0, np.nan)
    telemetry["rpm_rate_of_change"] = telemetry["rpm"].diff().div(dt).fillna(0.0)
    telemetry["egt_rolling_mean"] = telemetry["engine_temp"].rolling(window=15, min_periods=1).mean()
    telemetry["vibration_rolling_std"] = telemetry["vibration"].rolling(window=15, min_periods=2).std().fillna(0.0)

    rpm_norm = ((telemetry["rpm"] - 1100.0) / (15500.0 - 1100.0)).clip(0.0, 1.0)
    egt_norm = ((telemetry["engine_temp"] - 150.0) / 750.0).clip(0.0, 1.0)
    vib_norm = ((telemetry["vibration"] - 0.05) / 1.2).clip(0.0, 1.0)

    if "efficiency" not in telemetry.columns:
        telemetry["efficiency"] = (100.0 * (0.45 * rpm_norm + 0.35 * (1.0 - egt_norm) + 0.20 * (1.0 - vib_norm))).clip(0.0, 100.0)

    telemetry["degradation_trend"] = telemetry["efficiency"].rolling(window=20, min_periods=2).mean().diff().rolling(window=8, min_periods=1).mean().fillna(0.0)

    if "anomaly_score" not in telemetry.columns:
        temp_z = (telemetry["engine_temp"] - telemetry["egt_rolling_mean"]).abs() / (telemetry["engine_temp"].rolling(15, min_periods=3).std().fillna(1.0) + 1e-6)
        vib_z = (telemetry["vibration"] - telemetry["vibration"].rolling(15, min_periods=3).mean().fillna(telemetry["vibration"])).abs() / (
            telemetry["vibration"].rolling(15, min_periods=3).std().fillna(0.01) + 1e-6
        )
        roc_z = telemetry["rpm_rate_of_change"].abs() / (telemetry["rpm_rate_of_change"].rolling(20, min_periods=3).std().fillna(1.0) + 1e-6)
        telemetry["anomaly_score"] = (1.0 + 0.7 * temp_z + 0.7 * vib_z + 0.4 * roc_z).clip(lower=0.0)

    critical_mask = (
        (telemetry["anomaly_score"] >= anomaly_threshold * 1.7)
        | (telemetry["degradation_trend"] < -0.20)
        | ((telemetry["engine_temp"] - telemetry["egt_rolling_mean"]) > 45.0)
    )
    warning_mask = (
        (telemetry["anomaly_score"] >= anomaly_threshold)
        | (telemetry["degradation_trend"] < -0.05)
        | (telemetry["vibration_rolling_std"] > 0.04)
    )
    telemetry["risk_level"] = np.where(critical_mask, "Critical", np.where(warning_mask, "Warning", "Normal"))

    return telemetry


def _system_interpretation(latest: pd.Series) -> str:
    findings = []
    if latest["engine_temp"] - latest["egt_rolling_mean"] > 25:
        findings.append("Thermal drift detected")
    if latest["vibration_rolling_std"] > 0.03:
        findings.append("Abnormal vibration growth")
    if latest["degradation_trend"] < -0.05:
        findings.append("Efficiency degradation trend detected")
    if latest["risk_level"] == "Critical" and not findings:
        findings.append("Critical anomaly score elevation")
    if not findings:
        findings.append("Engine telemetry stable and within expected envelope")
    return ". ".join(findings) + "."

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

presets = {
    "Idle / Ground": {"base": 0.15, "variation": 0.02, "desc": "Very stable, low RPM, low EGT, minimal vibration."},
    "Taxi": {"base": 0.30, "variation": 0.05, "desc": "Light oscillation with small RPM fluctuations."},
    "Takeoff": {"base": 0.95, "variation": 0.03, "desc": "Very high sustained thrust, minimal variation, high thermal stress."},
    "Climb": {"base": 0.85, "variation": 0.05, "desc": "Slight corrections under high load."},
    "Cruise": {"base": 0.55, "variation": 0.03, "desc": "Stable mid-power; good for anomaly monitoring tests."},
    "Aggressive Maneuver": {"base": 0.75, "variation": 0.25, "desc": "Strong oscillations; stress-test mode."},
}
preset_names = list(presets.keys())
preset_mode = st.sidebar.selectbox(
    "Mission Preset",
    preset_names,
    index=preset_names.index("Cruise"),
    disabled=uploaded_file is not None
)

if "base_throttle_value" not in st.session_state:
    st.session_state.base_throttle_value = presets[preset_mode]["base"]
if "throttle_variation_value" not in st.session_state:
    st.session_state.throttle_variation_value = presets[preset_mode]["variation"]
if "last_preset_mode" not in st.session_state:
    st.session_state.last_preset_mode = preset_mode

if preset_mode != st.session_state.last_preset_mode:
    st.session_state.base_throttle_value = presets[preset_mode]["base"]
    st.session_state.throttle_variation_value = presets[preset_mode]["variation"]
    st.session_state.last_preset_mode = preset_mode

st.sidebar.caption(presets[preset_mode]["desc"])

num_points = st.sidebar.slider(
    "Telemetry Samples",
    200,
    2000,
    500,
    disabled=uploaded_file is not None
)
anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 2.0, 5.0, 3.0)
base_throttle = st.sidebar.slider(
    "Base Throttle",
    0.10,
    0.95,
    0.01,
    key="base_throttle_value",
    disabled=uploaded_file is not None
)
throttle_variation = st.sidebar.slider(
    "Throttle Variation",
    0.01,
    0.40,
    0.01,
    key="throttle_variation_value",
    disabled=uploaded_file is not None
)

# --------------------------------------------------
# DATA SOURCE (UPLOADED CSV OR SYNTHETIC)
# --------------------------------------------------

data = None
source_label = "ENGINE MODEL"

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        normalized = {col.strip().lower(): col for col in raw_data.columns}

        required_cols = ["engine_temp"]
        missing_cols = [col for col in required_cols if col not in normalized]

        if missing_cols:
            st.sidebar.error(
                "CSV missing required columns: " + ", ".join(missing_cols)
            )
        else:
            selected = pd.DataFrame({
                "engine_temp": pd.to_numeric(raw_data[normalized["engine_temp"]], errors="coerce"),
            })

            if "time" in normalized:
                selected["time"] = pd.to_numeric(raw_data[normalized["time"]], errors="coerce")
            else:
                selected["time"] = np.arange(len(selected))

            optional_numeric_cols = ["rpm", "vibration", "fuel_flow", "efficiency", "anomaly_score"]
            for col in optional_numeric_cols:
                if col in normalized:
                    selected[col] = pd.to_numeric(raw_data[normalized[col]], errors="coerce")

            data = selected.dropna(subset=["time", "engine_temp"])

            if data.empty:
                st.sidebar.error("CSV loaded but no valid numeric rows were found.")
            else:
                source_label = "UPLOADED CSV"
                st.sidebar.success(f"Loaded {len(data)} rows from CSV")

    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")

if data is None:
    rng = np.random.default_rng(42)
    phase = np.linspace(0.0, 6.0 * np.pi, num_points)
    throttle_profile = (
        base_throttle
        + throttle_variation * np.sin(phase)
        + 0.07 * np.sin(phase / 3.0 + 1.2)
        + rng.normal(0.0, throttle_variation / 4.0, num_points)
    )
    throttle_profile = np.clip(throttle_profile, 0.0, 1.0)

    model = EngineStateModel(rng_seed=42)
    data = model.run_profile(throttle_profile, dt=0.5)

data = _build_live_features(data, anomaly_threshold=anomaly_threshold)
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

latest = data.iloc[-1]
interpretation = _system_interpretation(latest)

# --------------------------------------------------
# RAW TELEMETRY + DERIVED + ML OUTPUT
# --------------------------------------------------

st.markdown("### LIVE ENGINE STATE")

raw1, raw2, raw3, raw4 = st.columns(4)
raw1.metric("RPM", f"{latest['rpm']:.0f}")
raw2.metric("EGT", f"{latest['engine_temp']:.1f} C")
raw3.metric("VIBRATION", f"{latest['vibration']:.3f}")
raw4.metric("EFFICIENCY", f"{latest['efficiency']:.1f}%")

drv1, drv2, drv3, drv4 = st.columns(4)
drv1.metric("RPM RATE", f"{latest['rpm_rate_of_change']:.1f} /s")
drv2.metric("EGT ROLLING MEAN", f"{latest['egt_rolling_mean']:.1f} C")
drv3.metric("VIB ROLLING STD", f"{latest['vibration_rolling_std']:.4f}")
drv4.metric("DEGRADATION TREND", f"{latest['degradation_trend']:.4f}")

ml1, ml2 = st.columns(2)
ml1.metric("ANOMALY SCORE", f"{latest['anomaly_score']:.2f}")
ml2.metric("RISK LEVEL", latest["risk_level"])

st.info(interpretation)

st.markdown("#### LIVE FEATURE DATAFRAME")
st.dataframe(
    data[
        [
            "time",
            "rpm",
            "engine_temp",
            "vibration",
            "efficiency",
            "rpm_rate_of_change",
            "egt_rolling_mean",
            "vibration_rolling_std",
            "degradation_trend",
            "anomaly_score",
            "risk_level",
        ]
    ].tail(60),
    width="stretch",
    hide_index=True,
)

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

st.plotly_chart(fig, width="stretch")

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

st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("FALCONEYE FLIGHT ANALYTICS MODULE // READY")
