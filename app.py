import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64
import json
from pathlib import Path
from engine import EngineStateModel
from stream_processor import StreamingTelemetryProcessor, normalize_telemetry_frame

APP_DIR = Path(__file__).resolve().parent


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
    if "rpm_rate_of_change" not in telemetry.columns:
        telemetry["rpm_rate_of_change"] = telemetry["rpm"].diff().div(dt).fillna(0.0)
    if "temp_rate_of_change" not in telemetry.columns:
        telemetry["temp_rate_of_change"] = telemetry["engine_temp"].diff().div(dt).fillna(0.0)
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

    if "anomaly_confidence" not in telemetry.columns:
        confidence = (
            1.0
            - 0.06 * np.clip(telemetry["vibration_rolling_std"] / 0.05, 0.0, 1.0)
            - 0.06 * np.clip(telemetry["rpm_rate_of_change"].abs() / 2500.0, 0.0, 1.0)
        )
        telemetry["anomaly_confidence"] = confidence.clip(0.20, 0.99)

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
    telemetry["model_confidence_pct"] = (telemetry["anomaly_confidence"] * 100.0).clip(0.0, 100.0)

    if "primary_reason" not in telemetry.columns:
        temp_component = ((telemetry["engine_temp"] - telemetry["egt_rolling_mean"]).abs() / 30.0).clip(lower=0.0)
        vib_component = (telemetry["vibration_rolling_std"] / 0.03).clip(lower=0.0)
        trend_component = (-telemetry["degradation_trend"] / 0.04).clip(lower=0.0)
        components = pd.DataFrame(
            {
                "temp_deviation": temp_component,
                "vibration_excess": vib_component,
                "efficiency_drop": trend_component,
            }
        )
        telemetry["primary_reason"] = components.idxmax(axis=1)

    if "reason_codes" not in telemetry.columns:
        telemetry["reason_codes"] = np.where(telemetry["risk_level"] == "Normal", "NOMINAL", telemetry["primary_reason"].str.upper())

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
    if "primary_reason" in latest:
        findings.append(f"Top contributor: {str(latest['primary_reason']).replace('_', ' ')}")
    if "anomaly_confidence" in latest:
        findings.append(f"Model confidence: {latest['anomaly_confidence'] * 100.0:.1f}%")
    if not findings:
        findings.append("Engine telemetry stable and within expected envelope")
    return ". ".join(findings) + "."


def _fault_color_map(risk_level: str) -> dict:
    if risk_level == "Critical":
        return {
            "fuselage": "#6b0000",
            "wings": "#420000",
            "engines": "#ff1a1a",
            "canopy": "#550000",
            "status": "CRITICAL",
        }
    if risk_level == "Warning":
        return {
            "fuselage": "#0f2b10",
            "wings": "#103513",
            "engines": "#ff4d00",
            "canopy": "#1a331a",
            "status": "WARNING",
        }
    return {
        "fuselage": "#1f6f1f",
        "wings": "#2f8f2f",
        "engines": "#00ff66",
        "canopy": "#4fa36d",
        "status": "NOMINAL",
    }


def _resolve_rafale_model_data_uri(uploaded_model) -> tuple[str | None, str | None]:
    if uploaded_model is not None:
        model_name = uploaded_model.name
        model_bytes = uploaded_model.getvalue()
        ext = Path(model_name).suffix.lower()
        mime = "model/gltf-binary" if ext == ".glb" else "model/gltf+json"
        encoded = base64.b64encode(model_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}", model_name

    local_candidates = [
        APP_DIR / "assets" / "rafale.glb",
        APP_DIR / "assets" / "rafale.gltf",
        APP_DIR / "models" / "rafale.glb",
        APP_DIR / "models" / "rafale.gltf",
        APP_DIR / "rafale.glb",
        APP_DIR / "rafale.gltf",
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.is_file():
            model_bytes = candidate.read_bytes()
            ext = candidate.suffix.lower()
            mime = "model/gltf-binary" if ext == ".glb" else "model/gltf+json"
            encoded = base64.b64encode(model_bytes).decode("ascii")
            return f"data:{mime};base64,{encoded}", str(candidate.relative_to(APP_DIR))

    return None, None


def _resolve_turbofan_model_data_uri(uploaded_model) -> tuple[str | None, str | None]:
    if uploaded_model is not None:
        model_name = uploaded_model.name
        model_bytes = uploaded_model.getvalue()
        ext = Path(model_name).suffix.lower()
        mime = "model/gltf-binary" if ext == ".glb" else "model/gltf+json"
        encoded = base64.b64encode(model_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}", model_name

    local_candidates = [
        APP_DIR / "assets" / "turbofan.glb",
        APP_DIR / "assets" / "turbofan.gltf",
        APP_DIR / "models" / "turbofan.glb",
        APP_DIR / "models" / "turbofan.gltf",
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.is_file():
            model_bytes = candidate.read_bytes()
            ext = candidate.suffix.lower()
            mime = "model/gltf-binary" if ext == ".glb" else "model/gltf+json"
            encoded = base64.b64encode(model_bytes).decode("ascii")
            return f"data:{mime};base64,{encoded}", str(candidate.relative_to(APP_DIR))

    return None, None


def _threejs_rafale_html(
    model_data_uri: str,
    engine_model_data_uri: str | None,
    risk_level: str,
    anomaly_payload: dict,
    enable_hand_tracking: bool,
) -> str:
    status = str(risk_level).upper()
    return f"""
<div id="rafale-wrap" style="width:100%;height:620px;position:relative;background:radial-gradient(circle at 20% 15%, #10161c, #05080b 55%);border:1px solid #00ff66;">
  <div style="position:absolute;top:10px;left:12px;color:#00ff66;font-family:Rajdhani,Segoe UI,sans-serif;letter-spacing:1px;z-index:10;">
    DASSAULT RAFALE // 3D MODEL // STATUS: {status}
  </div>
  <div id="rafale-compat-status" style="position:absolute;bottom:10px;left:12px;color:#8bd7a5;font-family:Rajdhani,Segoe UI,sans-serif;z-index:10;">
    Loading compatibility viewer...
  </div>
  <div id="rafale-viewer-root" style="position:absolute;inset:0;"></div>
</div>

<script>
(async function() {{
  const statusEl = document.getElementById("rafale-compat-status");
  const rootEl = document.getElementById("rafale-viewer-root");
  const modelUrl = {json.dumps(model_data_uri)};
  const engineModelUrl = {json.dumps(engine_model_data_uri)};
  const anomalyPayload = {json.dumps(anomaly_payload)};
  const handTrackingEnabled = {"true" if enable_hand_tracking else "false"};
  const riskLevel = String((anomalyPayload && anomalyPayload.risk) || "normal").toLowerCase();

  function loadScript(url) {{
    return new Promise((resolve, reject) => {{
      const s = document.createElement("script");
      s.src = url;
      s.async = true;
      s.onload = () => resolve(url);
      s.onerror = () => reject(new Error("Failed: " + url));
      document.head.appendChild(s);
    }});
  }}

  async function loadAny(candidates, label) {{
    for (const url of candidates) {{
      try {{
        await loadScript(url);
        return url;
      }} catch (e) {{
        console.warn(label + " load failed", url, e);
      }}
    }}
    throw new Error(label + " failed on all CDNs");
  }}

  const THREE_URLS = [
    "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
    "https://unpkg.com/three@0.128.0/build/three.min.js"
  ];
  const ORBIT_URLS = [
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
    "https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"
  ];
  const GLTF_URLS = [
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js",
    "https://unpkg.com/three@0.128.0/examples/js/loaders/GLTFLoader.js"
  ];
  const MP_HANDS_URLS = [
    "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js",
    "https://unpkg.com/@mediapipe/hands/hands.js"
  ];

  try {{
    statusEl.textContent = "Loading Three.js libraries...";
    await loadAny(THREE_URLS, "three");
    await loadAny(ORBIT_URLS, "orbit");
    await loadAny(GLTF_URLS, "gltf");
    if (handTrackingEnabled) {{
      await loadAny(MP_HANDS_URLS, "mediapipe-hands");
    }}
  }} catch (libErr) {{
    console.error("Library bootstrap failed:", libErr);
    statusEl.textContent = "3D libraries blocked/unavailable on this network";
    return;
  }}

  if (!window.THREE || !THREE.OrbitControls || !THREE.GLTFLoader) {{
    statusEl.textContent = "Three.js libraries loaded incompletely";
    return;
  }}

  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(rootEl.clientWidth || 1024, rootEl.clientHeight || 620);
  renderer.setClearColor(0x06090d, 1);
  rootEl.innerHTML = "";
  rootEl.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, (rootEl.clientWidth || 1024) / (rootEl.clientHeight || 620), 0.01, 1000);
  camera.position.set(10, 4, 10);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.target.set(0, 0, 0);

  let trackedModel = null;
  const baseRotation = {{ x: 0, y: 0, z: 0 }};
  const desiredRotation = {{ x: 0, y: 0, z: 0 }};
  let handPaused = false;
  let lastHandOpen = null;
  let lastPalmX = null;
  let lastPalmY = null;
  let lastPalmZ = null;
  let lastPalmTwist = null;
  let lastTwoCenterX = null;
  let lastTwoCenterY = null;
  let lastTwoDepth = null;
  let lastTwoAngle = null;
  const zoomDeadband = 0.004;
  const zoomGain = 95.0;

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(12, 14, 10);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88b8ff, 0.55);
  fill.position.set(-10, 8, -6);
  scene.add(fill);

  function fitCameraToObject(obj) {{
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    const fov = camera.fov * Math.PI / 180.0;
    const dist = (maxDim * 0.5) / Math.tan(fov * 0.5) * 1.6;
    camera.position.copy(center.clone().add(new THREE.Vector3(1.0, 0.35, 1.0).normalize().multiplyScalar(dist)));
    camera.near = Math.max(0.01, maxDim / 200);
    camera.far = Math.max(200, maxDim * 250);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.minDistance = maxDim * 0.2;
    controls.maxDistance = maxDim * 20;
    controls.update();
  }}

  function meshMatches(meshName, region) {{
    const tokens = Array.isArray(region.tokens) ? region.tokens : [];
    return tokens.some((t) => meshName.includes(String(t).toLowerCase()));
  }}

  function findWingAnchors(model) {{
    const fallback = {{
      left: new THREE.Vector3(-2.0, -0.5, 0.5),
      right: new THREE.Vector3(2.0, -0.5, 0.5),
    }};
    const candidates = [];
    model.traverse(function(node) {{
      if (!node.isMesh) return;
      const n = (node.name || "").toLowerCase();
      if (!n.includes("wing")) return;
      const bb = new THREE.Box3().setFromObject(node);
      candidates.push({{ name: n, center: bb.getCenter(new THREE.Vector3()) }});
    }});
    if (candidates.length === 0) return fallback;
    const left = candidates.find((c) => c.name.includes("left") || c.center.x < 0) || null;
    const right = candidates.find((c) => c.name.includes("right") || c.center.x > 0) || null;
    return {{
      left: (left ? left.center.clone() : fallback.left.clone()).add(new THREE.Vector3(0.0, -0.35, 0.45)),
      right: (right ? right.center.clone() : fallback.right.clone()).add(new THREE.Vector3(0.0, -0.35, 0.45)),
    }};
  }}

  function mountTwinEngines(loader, model, done) {{
    if (!engineModelUrl) {{
      done(false);
      return;
    }}
    loader.load(
      engineModelUrl,
      function(engineGltf) {{
        const template = engineGltf.scene || (engineGltf.scenes && engineGltf.scenes[0]);
        if (!template) {{
          done(false);
          return;
        }}
        const anchors = findWingAnchors(model);
        const aircraftBox = new THREE.Box3().setFromObject(model);
        const aircraftSize = aircraftBox.getSize(new THREE.Vector3());
        const aircraftMax = Math.max(aircraftSize.x, aircraftSize.y, aircraftSize.z, 1e-6);

        const engineBox = new THREE.Box3().setFromObject(template);
        const engineSize = engineBox.getSize(new THREE.Vector3());
        const engineMax = Math.max(engineSize.x, engineSize.y, engineSize.z, 1e-6);
        const engineScale = (aircraftMax * 0.14) / engineMax;

        const leftEngine = template.clone(true);
        const rightEngine = template.clone(true);
        leftEngine.name = "engine_left";
        rightEngine.name = "engine_right";
        leftEngine.position.copy(anchors.left);
        rightEngine.position.copy(anchors.right);
        leftEngine.scale.setScalar(engineScale);
        rightEngine.scale.setScalar(engineScale);
        leftEngine.rotation.z = 0.05;
        rightEngine.rotation.z = -0.05;
        model.add(leftEngine);
        model.add(rightEngine);
        done(true);
      }},
      undefined,
      function(err) {{
        console.warn("Engine model load failed:", err);
        done(false);
      }}
    );
  }}

  function vComp(v, idx) {{
    if (idx === 0) return v.x;
    if (idx === 1) return v.y;
    return v.z;
  }}

  // Infer engine meshes when model names are generic.
  function inferEngineMeshes(model, engineTokens) {{
    const byName = new Set();
    const meshInfo = [];
    const overall = new THREE.Box3().setFromObject(model);
    const oSize = overall.getSize(new THREE.Vector3());
    const oCenter = overall.getCenter(new THREE.Vector3());
    const longAxis = (oSize.x >= oSize.y && oSize.x >= oSize.z) ? 0 : (oSize.y >= oSize.z ? 1 : 2);
    const orth = [0, 1, 2].filter((a) => a !== longAxis);

    model.traverse(function(node) {{
      if (!node.isMesh || !node.geometry) return;
      const meshName = (node.name || "").toLowerCase();
      if (engineTokens.some((token) => meshName.includes(token))) {{
        byName.add(node);
      }}
      const bb = new THREE.Box3().setFromObject(node);
      const c = bb.getCenter(new THREE.Vector3());
      const s = bb.getSize(new THREE.Vector3());
      const spanLong = Math.max(vComp(oSize, longAxis), 1e-6);
      const t = (vComp(c, longAxis) - vComp(overall.min, longAxis)) / spanLong;
      const spanA = Math.max(vComp(oSize, orth[0]) * 0.5, 1e-6);
      const spanB = Math.max(vComp(oSize, orth[1]) * 0.5, 1e-6);
      const radial = Math.sqrt(
        Math.pow((vComp(c, orth[0]) - vComp(oCenter, orth[0])) / spanA, 2) +
        Math.pow((vComp(c, orth[1]) - vComp(oCenter, orth[1])) / spanB, 2)
      );
      const sizeW = Math.max(1e-5, s.x * s.y * s.z);
      meshInfo.push({{ node, t, radial, sizeW }});
    }});

    if (byName.size > 0) return byName;
    if (meshInfo.length === 0) return byName;

    // Choose tail-end by scoring compact meshes near either longitudinal end.
    let minEndScore = 0.0;
    let maxEndScore = 0.0;
    for (const m of meshInfo) {{
      const dMin = m.t;
      const dMax = 1.0 - m.t;
      if (dMin < 0.40) minEndScore += (0.40 - dMin) * (1.4 - Math.min(m.radial, 1.4)) * m.sizeW;
      if (dMax < 0.40) maxEndScore += (0.40 - dMax) * (1.4 - Math.min(m.radial, 1.4)) * m.sizeW;
    }}
    const chooseMinEnd = minEndScore >= maxEndScore;

    const inferred = new Set();
    for (const m of meshInfo) {{
      const endDist = chooseMinEnd ? m.t : (1.0 - m.t);
      if (endDist < 0.33 && m.radial < 1.15) {{
        inferred.add(m.node);
      }}
    }}
    return inferred;
  }}

  function pointDist(a, b) {{
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }}

  function isFingerCurled(lm, tip, pip, mcp) {{
    return lm[tip].y > lm[pip].y && lm[tip].y > (lm[mcp].y - 0.01);
  }}

  function countExtendedFingers(lm) {{
    let count = 0;
    if (lm[8].y < lm[6].y) count += 1;
    if (lm[12].y < lm[10].y) count += 1;
    if (lm[16].y < lm[14].y) count += 1;
    if (lm[20].y < lm[18].y) count += 1;
    return count;
  }}

  function detectFist(lm) {{
    const indexCurled = isFingerCurled(lm, 8, 6, 5);
    const middleCurled = isFingerCurled(lm, 12, 10, 9);
    const ringCurled = isFingerCurled(lm, 16, 14, 13);
    const pinkyCurled = isFingerCurled(lm, 20, 18, 17);
    const curledCount = [indexCurled, middleCurled, ringCurled, pinkyCurled].filter(Boolean).length;
    const thumbFolded = pointDist(lm[4], lm[2]) < 0.08 && pointDist(lm[4], lm[5]) < 0.10;
    const palmCompact = (
      pointDist(lm[8], lm[0]) +
      pointDist(lm[12], lm[0]) +
      pointDist(lm[16], lm[0]) +
      pointDist(lm[20], lm[0])
    ) / 4.0 < 0.24;
    return curledCount >= 3 && thumbFolded && palmCompact;
  }}

  function detectOpenPalm(lm) {{
    return (
      lm[8].y < lm[6].y &&
      lm[12].y < lm[10].y &&
      lm[16].y < lm[14].y &&
      lm[20].y < lm[18].y
    );
  }}

  // Approximate hand openness: larger values = open hand, smaller = closed fist.
  function handOpenMetric(lm) {{
    const vals = [
      pointDist(lm[8], lm[5]),
      pointDist(lm[12], lm[9]),
      pointDist(lm[16], lm[13]),
      pointDist(lm[20], lm[17]),
    ];
    return (vals[0] + vals[1] + vals[2] + vals[3]) / 4.0;
  }}

  function wrapAngleDelta(delta) {{
    while (delta > Math.PI) delta -= Math.PI * 2.0;
    while (delta < -Math.PI) delta += Math.PI * 2.0;
    return delta;
  }}

  // "Holding a ball" gesture: not fist, not pinch, partially open curved hand.
  function isBallHoldGesture(lm, isFistNow) {{
    if (isFistNow) return false;
    const extended = countExtendedFingers(lm);
    return extended >= 1 && extended <= 3;
  }}

  async function initHandTracking() {{
    if (!handTrackingEnabled) return;
    if (!window.Hands || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
      statusEl.textContent = "Viewer loaded (hand tracking unavailable)";
      return;
    }}
    const hands = new window.Hands({{
      locateFile: (file) => "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" + file
    }});
    hands.setOptions({{
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.65,
      minTrackingConfidence: 0.6
    }});

    hands.onResults((results) => {{
      if (!trackedModel || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;
      const handLandmarks = results.multiHandLandmarks.filter((lm) => lm && lm.length >= 21);
      if (handLandmarks.length === 0) return;

      const anyFist = handLandmarks.some((lm) => detectFist(lm));
      if (anyFist && !handPaused) {{
        handPaused = true;
        lastHandOpen = null;
        lastTwoCenterX = null;
        lastTwoCenterY = null;
        lastTwoDepth = null;
        lastTwoAngle = null;
        statusEl.textContent = "Tracking paused (fist)";
        return;
      }}
      if (handPaused) {{
        if (handLandmarks.some((lm) => detectOpenPalm(lm))) {{
          handPaused = false;
          lastHandOpen = null;
          lastPalmX = null;
          lastPalmY = null;
          lastPalmZ = null;
          lastPalmTwist = null;
          lastTwoCenterX = null;
          lastTwoCenterY = null;
          lastTwoDepth = null;
          lastTwoAngle = null;
          statusEl.textContent = "Tracking resumed";
        }}
        return;
      }}

      // Mode A: one hand = zoom only (no rotation updates).
      if (handLandmarks.length === 1) {{
        const lm = handLandmarks[0];
        const openNow = handOpenMetric(lm);
        if (lastHandOpen !== null) {{
          const openDelta = openNow - lastHandOpen;
          if (Math.abs(openDelta) > zoomDeadband) {{
            const toCam = camera.position.clone().sub(controls.target).normalize();
            const cur = camera.position.distanceTo(controls.target);
            const next = THREE.MathUtils.clamp(cur - openDelta * zoomGain, 4.5, 26.0);
            camera.position.copy(controls.target.clone().add(toCam.multiplyScalar(next)));
            statusEl.textContent = openDelta < 0 ? "One-hand zoom in" : "One-hand zoom out";
          }} else {{
            statusEl.textContent = "One-hand zoom mode";
          }}
        }}
        lastHandOpen = openNow;

        // Reset two-hand rotation deltas to avoid jumps when second hand appears.
        lastTwoCenterX = null;
        lastTwoCenterY = null;
        lastTwoDepth = null;
        lastTwoAngle = null;
        lastPalmX = null;
        lastPalmY = null;
        lastPalmZ = null;
        lastPalmTwist = null;
        return;
      }}

      // Mode B: two hands = grab-and-rotate only.
      const sorted = handLandmarks.slice().sort((a, b) => a[9].x - b[9].x);
      const left = sorted[0];
      const right = sorted[1];
      const cx = (left[9].x + right[9].x) * 0.5;
      const cy = (left[9].y + right[9].y) * 0.5;
      const cz = ((left[9].z || 0) + (right[9].z || 0)) * 0.5;
      const lineAngle = Math.atan2(right[9].y - left[9].y, right[9].x - left[9].x);

      if (lastTwoCenterX !== null && lastTwoCenterY !== null) {{
        const dx = cx - lastTwoCenterX;
        const dy = cy - lastTwoCenterY;
        const dz = cz - lastTwoDepth;
        desiredRotation.y = THREE.MathUtils.clamp(desiredRotation.y - dx * 7.8, -1.20, 1.20);
        desiredRotation.x = THREE.MathUtils.clamp(desiredRotation.x - dy * 7.4 - dz * 2.0, -0.85, 0.85);
      }}
      if (lastTwoAngle !== null) {{
        const dA = wrapAngleDelta(lineAngle - lastTwoAngle);
        desiredRotation.z = THREE.MathUtils.clamp(desiredRotation.z + dA * 1.9, -1.05, 1.05);
      }}

      lastTwoCenterX = cx;
      lastTwoCenterY = cy;
      lastTwoDepth = cz;
      lastTwoAngle = lineAngle;
      lastHandOpen = null;
      statusEl.textContent = "Two-hand grab rotate mode";
    }});

    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.style.display = "none";
    rootEl.appendChild(video);
    const stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: "user", width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
      audio: false
    }});
    video.srcObject = stream;
    await video.play();
    statusEl.textContent = "Hand tracking active";

    async function pump() {{
      if (video.readyState >= 2) {{
        await hands.send({{ image: video }});
      }}
      requestAnimationFrame(pump);
    }}
    pump();
  }}

  const loader = new THREE.GLTFLoader();
  loader.load(
    modelUrl,
    function(gltf) {{
      const model = gltf.scene || (gltf.scenes && gltf.scenes[0]);
      if (!model) {{
        statusEl.textContent = "GLB loaded but no scene found";
        return;
      }}

      const anomalyRegions = (anomalyPayload && anomalyPayload.regions) ? anomalyPayload.regions : [];
      const red = new THREE.Color("#ff1a1a");
      const hull = new THREE.Color("#95a29b");
      const engineTokens = ["engine", "nozzle", "turbine", "exhaust", "afterburner"];
      const engineMeshSet = inferEngineMeshes(model, engineTokens);

      model.traverse(function(node) {{
        if (!node.isMesh) return;
        node.material = (node.material && node.material.clone) ? node.material.clone() : new THREE.MeshStandardMaterial();
        node.material.side = THREE.DoubleSide;
        node.material.metalness = Math.max(node.material.metalness || 0, 0.35);
        node.material.roughness = Math.min(node.material.roughness || 0.8, 0.62);
        if (node.material.color) node.material.color.lerp(hull, 0.25);
        node.material.transparent = true;
        node.material.opacity = 0.08;
        node.material.depthWrite = false;

        const meshName = (node.name || "").toLowerCase();
        const hit = anomalyRegions.find((r) => meshMatches(meshName, r));
        const isEngineMesh = engineMeshSet.has(node);
        const engineWarning = isEngineMesh && (riskLevel === "warning" || riskLevel === "critical");
        if (hit) {{
          if (!node.material.emissive) node.material.emissive = red.clone();
          node.material.emissive.copy(red);
          node.material.emissiveIntensity = 1.05;
          if (node.material.color) node.material.color.lerp(red, 0.55);
          node.material.opacity = 0.62;
          node.material.depthWrite = true;
        }} else if (engineWarning) {{
          if (!node.material.emissive) node.material.emissive = red.clone();
          node.material.emissive.copy(red);
          node.material.emissiveIntensity = 1.0;
          if (node.material.color) node.material.color.lerp(red, 0.45);
          node.material.opacity = 0.55;
          node.material.depthWrite = true;
        }}

        const edges = new THREE.EdgesGeometry(node.geometry, 18);
        const edgeMat = new THREE.LineBasicMaterial({{
          color: (hit || engineWarning) ? red : new THREE.Color("#7fffb2"),
          transparent: true,
          opacity: (hit || engineWarning) ? 1.0 : 0.95
        }});
        const lines = new THREE.LineSegments(edges, edgeMat);
        lines.renderOrder = 2;
        node.add(lines);
      }});

      scene.add(model);
      const box = new THREE.Box3().setFromObject(model);
      const maxDim = Math.max(box.getSize(new THREE.Vector3()).x, box.getSize(new THREE.Vector3()).y, box.getSize(new THREE.Vector3()).z, 1e-6);
      model.scale.setScalar(8.0 / maxDim);
      const sbox = new THREE.Box3().setFromObject(model);
      model.position.sub(sbox.getCenter(new THREE.Vector3()));
      mountTwinEngines(loader, model, function(attached) {{
        fitCameraToObject(model);
        trackedModel = model;
        baseRotation.x = model.rotation.x;
        baseRotation.y = model.rotation.y;
        baseRotation.z = model.rotation.z;
        statusEl.textContent = handTrackingEnabled
          ? "Viewer loaded (starting hand tracking...)"
          : "Viewer loaded" + (attached ? " + twin engines mounted" : "");
        initHandTracking();
      }});
    }},
    function(evt) {{
      if (evt.total) {{
        statusEl.textContent = "Loading model... " + Math.round((evt.loaded / evt.total) * 100) + "%";
      }}
    }},
    function(err) {{
      console.error("GLTF load error:", err);
      statusEl.textContent = "Model load failed (see browser console)";
      const fallback = new THREE.Mesh(
        new THREE.BoxGeometry(2.8, 0.7, 0.8),
        new THREE.MeshStandardMaterial({{ color: 0x7b8e9d, metalness: 0.5, roughness: 0.4 }})
      );
      scene.add(fallback);
      fitCameraToObject(fallback);
    }}
  );

  function onResize() {{
    const w = rootEl.clientWidth || 1024;
    const h = rootEl.clientHeight || 620;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }}
  window.addEventListener("resize", onResize);

  function animate() {{
    if (trackedModel && handTrackingEnabled && !handPaused) {{
      trackedModel.rotation.x = THREE.MathUtils.lerp(trackedModel.rotation.x, baseRotation.x + desiredRotation.x, 0.24);
      trackedModel.rotation.y = THREE.MathUtils.lerp(trackedModel.rotation.y, baseRotation.y + desiredRotation.y, 0.24);
      trackedModel.rotation.z = THREE.MathUtils.lerp(trackedModel.rotation.z, baseRotation.z + desiredRotation.z, 0.20);
    }}
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }}
  animate();
}})();
</script>
"""


def _anomaly_regions(latest: pd.Series) -> dict:
    risk = str(latest.get("risk_level", "Normal")).lower()
    reason = str(latest.get("primary_reason", "")).lower()
    reason_code = str(latest.get("reason_codes", "")).lower()

    regions = []
    if risk in {"warning", "critical"}:
        if "temp_deviation" in reason or "temp_deviation" in reason_code:
            regions.append(
                {
                    "label": "Thermal anomaly",
                    "tokens": ["engine", "nozzle", "exhaust", "afterburner", "turbine"],
                    "severity": risk,
                }
            )
        elif "vibration_excess" in reason or "vibration_excess" in reason_code:
            regions.append(
                {
                    "label": "Vibration anomaly",
                    "tokens": ["turbine", "fan", "compressor", "shaft", "engine"],
                    "severity": risk,
                }
            )
        elif "efficiency_drop" in reason or "efficiency_drop" in reason_code:
            regions.append(
                {
                    "label": "Efficiency anomaly",
                    "tokens": ["intake", "inlet", "compressor", "exhaust", "engine"],
                    "severity": risk,
                }
            )
        else:
            regions.append(
                {
                    "label": "General anomaly",
                    "tokens": ["engine", "nozzle", "turbine", "exhaust"],
                    "severity": risk,
                }
            )

    return {"risk": risk, "regions": regions}


def _aircraft_3d_figure(latest: pd.Series) -> go.Figure:
    colors = _fault_color_map(str(latest["risk_level"]))
    fig = go.Figure()

    def _mono_surface(x_vals, y_vals, z_vals, color, name, opacity=1.0):
        fig.add_trace(
            go.Surface(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                surfacecolor=np.zeros_like(x_vals),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                name=name,
                hovertemplate=f"{name}<extra></extra>",
                lighting=dict(ambient=0.35, diffuse=0.95, specular=0.55, roughness=0.45, fresnel=0.25),
                lightposition=dict(x=120, y=70, z=160),
            )
        )

    # Fuselage body (axisymmetric profile)
    theta = np.linspace(0.0, 2.0 * np.pi, 56)
    x_fus = np.linspace(-4.0, 4.0, 150)
    radius_profile = np.interp(
        x_fus,
        [-4.0, -3.3, -2.0, 0.5, 2.3, 3.3, 4.0],
        [0.03, 0.42, 0.56, 0.58, 0.42, 0.22, 0.01],
    )
    fus_x = np.tile(x_fus[:, None], (1, theta.size))
    fus_y = radius_profile[:, None] * np.cos(theta)[None, :]
    fus_z = radius_profile[:, None] * np.sin(theta)[None, :]
    _mono_surface(fus_x, fus_y, fus_z, colors["fuselage"], "Fuselage")

    # Bubble canopy
    canopy_x = np.linspace(0.8, 1.9, 46)
    canopy_y = np.linspace(-0.32, 0.32, 34)
    cx, cy = np.meshgrid(canopy_x, canopy_y)
    canopy_shape = 1.0 - ((cx - 1.35) / 0.55) ** 2 - (cy / 0.28) ** 2
    cz = 0.10 + 0.42 * np.sqrt(np.clip(canopy_shape, 0.0, None))
    cz[canopy_shape < 0.0] = np.nan
    _mono_surface(cx, cy, cz, colors["canopy"], "Canopy", opacity=0.92)

    # Engine nacelles (fault-colored)
    engine_theta = np.linspace(0.0, 2.0 * np.pi, 36)
    engine_x = np.linspace(-2.6, -0.6, 60)
    ex = np.tile(engine_x[:, None], (1, engine_theta.size))
    for side, label in [(-0.58, "Left Engine"), (0.58, "Right Engine")]:
        ey = side + 0.18 * np.cos(engine_theta)[None, :]
        ez = -0.28 + 0.16 * np.sin(engine_theta)[None, :]
        _mono_surface(ex, np.tile(ey, (engine_x.size, 1)), np.tile(ez, (engine_x.size, 1)), colors["engines"], label)

    # Wing meshes (top + bottom + edges)
    def _wing_mesh(side: float, name: str):
        top = np.array(
            [
                [0.35, 0.42 * side, 0.03],
                [-1.05, 0.50 * side, 0.03],
                [-2.35, 3.05 * side, 0.03],
                [-0.25, 2.50 * side, 0.03],
            ]
        )
        bottom = top.copy()
        bottom[:, 2] = -0.03
        verts = np.vstack([top, bottom])
        i = [0, 0, 4, 4, 0, 1, 1, 2, 2, 3, 3, 0]
        j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
        k = [2, 3, 6, 7, 5, 2, 6, 3, 7, 0, 4, 7]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                color=colors["wings"],
                opacity=0.98,
                name=name,
                hovertemplate=f"{name}<extra></extra>",
                lighting=dict(ambient=0.35, diffuse=0.95, specular=0.5, roughness=0.5),
                lightposition=dict(x=120, y=70, z=160),
            )
        )

    _wing_mesh(1.0, "Right Wing")
    _wing_mesh(-1.0, "Left Wing")

    # Horizontal stabilizers
    def _stabilizer_mesh(side: float, name: str):
        top = np.array(
            [
                [-2.85, 0.28 * side, 0.18],
                [-3.55, 0.30 * side, 0.18],
                [-3.95, 1.25 * side, 0.18],
                [-2.95, 1.10 * side, 0.18],
            ]
        )
        bottom = top.copy()
        bottom[:, 2] = 0.12
        verts = np.vstack([top, bottom])
        i = [0, 0, 4, 4, 0, 1, 1, 2, 2, 3, 3, 0]
        j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
        k = [2, 3, 6, 7, 5, 2, 6, 3, 7, 0, 4, 7]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                color=colors["wings"],
                opacity=0.98,
                name=name,
                hovertemplate=f"{name}<extra></extra>",
                lighting=dict(ambient=0.35, diffuse=0.95, specular=0.5, roughness=0.5),
                lightposition=dict(x=120, y=70, z=160),
            )
        )

    _stabilizer_mesh(1.0, "Right Stabilizer")
    _stabilizer_mesh(-1.0, "Left Stabilizer")

    # Vertical tail
    vert = np.array(
        [
            [-3.05, 0.0, 0.18],
            [-3.95, 0.0, 0.25],
            [-3.45, 0.0, 1.25],
            [-2.95, 0.0, 1.05],
            [-3.05, 0.06, 0.18],
            [-3.95, 0.06, 0.25],
            [-3.45, 0.06, 1.25],
            [-2.95, 0.06, 1.05],
        ]
    )
    fig.add_trace(
        go.Mesh3d(
            x=vert[:, 0],
            y=vert[:, 1],
            z=vert[:, 2],
            i=[0, 0, 4, 4, 0, 1, 1, 2, 2, 3, 3, 0],
            j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
            k=[2, 3, 6, 7, 5, 2, 6, 3, 7, 0, 4, 7],
            color=colors["wings"],
            opacity=0.98,
            name="Vertical Tail",
            hovertemplate="Vertical Tail<extra></extra>",
            lighting=dict(ambient=0.35, diffuse=0.95, specular=0.5, roughness=0.5),
            lightposition=dict(x=120, y=70, z=160),
        )
    )

    # Engine status markers (explicit fault signal)
    fig.add_trace(
        go.Scatter3d(
            x=[-1.1, -1.1],
            y=[-0.58, 0.58],
            z=[-0.11, -0.11],
            mode="markers+text",
            marker=dict(size=8, color=colors["engines"], symbol="circle"),
            text=["L ENG", "R ENG"],
            textposition="top center",
            name="Engine Status",
            hovertemplate="Engine status: " + colors["status"] + "<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="#00ff66", family="Rajdhani"),
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="black"),
            yaxis=dict(visible=False, backgroundcolor="black"),
            zaxis=dict(visible=False, backgroundcolor="black"),
            bgcolor="black",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.6, z=0.9),
        ),
        legend=dict(font=dict(color="#00ff66"), orientation="h", y=1.02, x=0.01),
        title="3D AIRCRAFT STATUS (HIGH-FIDELITY MODEL)",
    )
    return fig

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="FalconEye Flight Diagnostic",
    page_icon="Untitled design (2).png",
    layout="wide"
)

# --------------------------------------------------
# F-16 HUD STYLE
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@500;600;700&family=Orbitron:wght@600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

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
    font-family: 'Exo 2', 'Orbitron', 'Rajdhani', sans-serif;
}

.falcon-header {
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}

.falcon-title {
    color: #00ff66;
    font-family: 'Exo 2', 'Orbitron', 'Rajdhani', sans-serif;
    font-size: 2.0rem;
    letter-spacing: 1.2px;
    font-weight: 700;
    line-height: 1.1;
}

.falcon-subtitle {
    color: #65ff9a;
    font-family: 'Rajdhani', 'Exo 2', sans-serif;
    font-size: 1.0rem;
    letter-spacing: 0.8px;
    margin-top: 4px;
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

brand_logo_col, brand_title_col = st.columns([1, 9], vertical_alignment="center")

with brand_logo_col:
    st.image("Untitled design (2).png", width=90)

with brand_title_col:
    st.markdown(
        """
        <div class='falcon-header'>
          <div class='falcon-title'>FALCONEYE // FLIGHT DIAGNOSTIC SYSTEM</div>
          <div class='falcon-subtitle'>Tactical Engine Analytics + 3D Digital Twin</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("MISSION PARAMETERS")

uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])
rafale_model_file = st.sidebar.file_uploader("Upload Rafale GLB/GLTF", type=["glb", "gltf"])
turbofan_model_file = st.sidebar.file_uploader("Upload Turbofan GLB/GLTF (Optional)", type=["glb", "gltf"])
enable_hand_tracking = st.sidebar.checkbox("Enable Hand Tracking (MediaPipe)", value=True)
stream_chunk = st.sidebar.slider("Real-time Chunk Size", 10, 250, 40)

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
        if "time" not in normalized:
            raw_data["time"] = np.arange(len(raw_data))
        data = normalize_telemetry_frame(raw_data)
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
else:
    processor = StreamingTelemetryProcessor(max_buffer_size=max(len(data), 1000))
    records = data.to_dict("records")
    for i in range(0, len(records), stream_chunk):
        processor.ingest_records(records[i:i + stream_chunk])
    data = processor.snapshot()

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

exp1, exp2, exp3 = st.columns(3)
exp1.metric("MODEL CONFIDENCE", f"{latest['model_confidence_pct']:.1f}%")
exp2.metric("PRIMARY REASON", str(latest["primary_reason"]).replace("_", " ").upper())
exp3.metric("REASON CODES", str(latest["reason_codes"]))

if "health_score" in data.columns or "sensor_health" in data.columns:
    h1, h2 = st.columns(2)
    if "health_score" in data.columns:
        h1.metric("ENGINE HEALTH", f"{latest['health_score'] * 100.0:.1f}%")
    if "sensor_health" in data.columns:
        h2.metric("SENSOR HEALTH", f"{latest['sensor_health'] * 100.0:.1f}%")

st.info(interpretation)

st.markdown("### AIRCRAFT DIGITAL TWIN")
fault_colors = _fault_color_map(str(latest["risk_level"]))
air_col1, air_col2 = st.columns([2, 1])
model_data_uri, model_source = _resolve_rafale_model_data_uri(rafale_model_file)
engine_model_data_uri, engine_model_source = _resolve_turbofan_model_data_uri(turbofan_model_file)
anomaly_payload = _anomaly_regions(latest)

with air_col1:
    if model_data_uri:
        components.html(
            _threejs_rafale_html(
                model_data_uri,
                engine_model_data_uri,
                str(latest["risk_level"]),
                anomaly_payload,
                enable_hand_tracking,
            ),
            height=620,
            scrolling=False,
        )
    else:
        st.warning(
            "Rafale model not found. Upload a `.glb`/`.gltf` in the sidebar or add "
            "`assets/rafale.glb` to the project root."
        )
        st.plotly_chart(_aircraft_3d_figure(latest), width="stretch")

with air_col2:
    st.metric("AIRCRAFT STATE", fault_colors["status"])
    st.metric("FAULT LEVEL", str(latest["risk_level"]).upper())
    st.metric("ENGINE VISUAL", "RED" if str(latest["risk_level"]) in {"Warning", "Critical"} else "GREEN")
    st.metric("3D MODEL SOURCE", model_source if model_source else "SCHEMATIC FALLBACK")
    st.metric("ENGINE MODEL SOURCE", engine_model_source if engine_model_source else "NONE")
    active_regions = ", ".join([region["label"] for region in anomaly_payload["regions"]]) if anomaly_payload["regions"] else "NONE"
    st.metric("ANOMALY REGIONS", active_regions)
    st.caption(
        "Fault regions are highlighted in red using mesh-name matching "
        "(`engine`, `nozzle`, `turbine`, `exhaust`, etc.)."
    )
    st.caption(
        "Compatibility viewer mode enabled for reliable model rendering."
    )

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
            "anomaly_confidence",
            "primary_reason",
            "reason_codes",
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
