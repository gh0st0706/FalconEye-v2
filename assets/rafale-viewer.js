/*
  Rafale GLB/GLTF Viewer (Three.js, module)
  ------------------------------------------
  Designed for Streamlit HTML embedding and robust visibility:
  - Loads GLB/GLTF via GLTFLoader
  - Centers + uniformly scales model
  - Auto-fits camera so full aircraft is visible
  - Adds ambient + directional lighting
  - OrbitControls for rotate/zoom/pan
  - Loading progress overlay
  - Hierarchy debug logging
  - API hooks for anomaly coloring + telemetry orientation

  Usage (inside <script type="module">):
    import { createRafaleViewer } from "./assets/rafale-viewer.js";
    const viewer = createRafaleViewer({
      container: document.getElementById("viewer"),
      modelUrl: "assets/rafale.glb" // or data URI
    });
    await viewer.load();
*/

import * as THREE from "https://esm.sh/three@0.162.0";
import { OrbitControls } from "https://esm.sh/three@0.162.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://esm.sh/three@0.162.0/examples/jsm/loaders/GLTFLoader.js";
import { HandLandmarker, FilesetResolver } from "https://esm.sh/@mediapipe/tasks-vision@0.10.14";

const DEFAULT_ENGINE_TOKENS = ["engine", "nozzle", "turbine", "exhaust", "afterburner"];
const DEFAULT_ANOMALY_REGION_TOKENS = {
  thermal: ["engine", "nozzle", "exhaust", "afterburner", "turbine"],
  vibration: ["turbine", "fan", "compressor", "shaft", "engine"],
  efficiency: ["intake", "inlet", "compressor", "exhaust", "engine"],
};

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerpAngle(a, b, t) {
  let d = (b - a + Math.PI) % (Math.PI * 2) - Math.PI;
  return a + d * t;
}

export function createRafaleViewer(config) {
  if (!config || !config.container) {
    throw new Error("createRafaleViewer: `container` is required.");
  }
  if (!config.modelUrl) {
    throw new Error("createRafaleViewer: `modelUrl` is required.");
  }

  const container = config.container;
  const modelUrl = config.modelUrl;
  const targetModelSpan = config.targetModelSpan ?? 8.0; // world units
  const debug = config.debug ?? true;
  const background = config.background ?? 0x06090d;

  container.style.position = "relative";
  container.style.background = "#05080b";
  container.style.overflow = "hidden";

  const progressEl = document.createElement("div");
  progressEl.style.position = "absolute";
  progressEl.style.left = "12px";
  progressEl.style.bottom = "10px";
  progressEl.style.font = "600 13px Rajdhani, Segoe UI, sans-serif";
  progressEl.style.color = "#7fffb2";
  progressEl.style.letterSpacing = "0.6px";
  progressEl.style.zIndex = "20";
  progressEl.textContent = "Loading model...";
  container.appendChild(progressEl);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.setClearColor(background, 1);
  renderer.domElement.style.display = "block";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(background, 35, 120);

  const camera = new THREE.PerspectiveCamera(
    42,
    container.clientWidth / Math.max(1, container.clientHeight),
    0.01,
    1000
  );
  camera.position.set(10, 4, 10);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.enablePan = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 300;
  controls.target.set(0, 0, 0);

  // Lighting tuned so metallic aircraft materials stay visible.
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
  keyLight.position.set(16, 18, 12);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x88b8ff, 0.65);
  fillLight.position.set(-12, 8, -10);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xaadfff, 0.45);
  rimLight.position.set(0, 6, -20);
  scene.add(rimLight);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.MeshStandardMaterial({
      color: 0x0a0f14,
      roughness: 0.96,
      metalness: 0.02,
    })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -2.7;
  scene.add(ground);

  const rootGroup = new THREE.Group();
  scene.add(rootGroup);

  let aircraft = null;
  let animationFrame = null;
  const materialBackup = new WeakMap();
  const engineMeshes = [];
  const edgeOverlays = [];
  let skeletonEnabled = config.skeletonEnabled ?? true;
  const baseSkeletonColor = config.skeletonColor || "#7fffb2";
  const baseSkeletonLineOpacity = config.skeletonLineOpacity ?? 0.95;
  const baseSkeletonSurfaceOpacity = config.skeletonSurfaceOpacity ?? 0.08;
  const handTrackingEnabled = config.enableHandTracking ?? false;
  let mpVideo = null;
  let mpStream = null;
  let handLandmarker = null;
  let lastHandTs = 0;
  let desiredHandRotation = { pitch: 0.0, yaw: 0.0, roll: 0.0 };
  let baseModelRotation = new THREE.Euler(0.0, 0.0, 0.0, "XYZ");
  let lastGesture = "none";
  let lastGestureTs = 0;

  function debugHierarchy(node, depth = 0) {
    if (!debug) return;
    const indent = "  ".repeat(depth);
    const marker = node.isMesh ? "[Mesh]" : "[Node]";
    console.log(`${indent}${marker} ${node.name || "(unnamed)"} (${node.type})`);
    for (const child of node.children || []) {
      debugHierarchy(child, depth + 1);
    }
  }

  function classifyGesture(lm) {
    const isExtended = (tip, pip) => lm[tip].y < lm[pip].y;
    const indexUp = isExtended(8, 6);
    const middleUp = isExtended(12, 10);
    const ringUp = isExtended(16, 14);
    const pinkyUp = isExtended(20, 18);
    const thumbOpen = lm[4].x > lm[3].x;
    const extendedCount = [indexUp, middleUp, ringUp, pinkyUp, thumbOpen].filter(Boolean).length;
    const pinchDist = Math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y);
    if (pinchDist < 0.055) return { name: "pinch", pinchDist };
    if (extendedCount <= 1) return { name: "fist", pinchDist };
    if (indexUp && middleUp && !ringUp && !pinkyUp) return { name: "victory", pinchDist };
    if (extendedCount >= 4) return { name: "open_palm", pinchDist };
    return { name: "track", pinchDist };
  }

  async function initMediaPipeHandTracking() {
    if (!handTrackingEnabled) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      progressEl.textContent = "Hand tracking unavailable (no camera API)";
      return;
    }
    try {
      progressEl.textContent = "Loading MediaPipe hand tracking...";
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      mpVideo = document.createElement("video");
      mpVideo.autoplay = true;
      mpVideo.muted = true;
      mpVideo.playsInline = true;
      mpVideo.style.display = "none";
      container.appendChild(mpVideo);
      mpStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      mpVideo.srcObject = mpStream;
      await mpVideo.play();
      progressEl.textContent = "Hand tracking active";
    } catch (err) {
      console.error("MediaPipe hand tracking init failed:", err);
      progressEl.textContent = "Hand tracking blocked/unavailable";
    }
  }

  function updateMediaPipeHandTracking() {
    if (!handTrackingEnabled || !handLandmarker || !mpVideo || mpVideo.readyState < 2 || !aircraft) return;
    const now = performance.now();
    if (now - lastHandTs < 33) return;
    lastHandTs = now;
    const result = handLandmarker.detectForVideo(mpVideo, now);
    if (!result?.landmarks?.length) return;
    const lm = result.landmarks[0];
    const gesture = classifyGesture(lm);

    const centerX = (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5.0;
    const centerY = (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5.0;
    const palmDx = lm[5].x - lm[17].x;
    const palmDy = lm[5].y - lm[17].y;
    desiredHandRotation.yaw = THREE.MathUtils.clamp((centerX - 0.5) * -2.4, -1.15, 1.15);
    desiredHandRotation.pitch = THREE.MathUtils.clamp((0.5 - centerY) * 2.1, -0.75, 0.75);
    desiredHandRotation.roll = THREE.MathUtils.clamp(Math.atan2(palmDy, palmDx) * 0.9, -0.85, 0.85);

    if (gesture.name === "pinch") {
      const zoomFactor = THREE.MathUtils.clamp((gesture.pinchDist - 0.03) / 0.20, 0.0, 1.0);
      const maxDistance = 26.0;
      const minDistance = 4.5;
      const targetDistance = maxDistance - zoomFactor * (maxDistance - minDistance);
      const toCam = camera.position.clone().sub(controls.target).normalize();
      camera.position.copy(controls.target.clone().add(toCam.multiplyScalar(targetDistance)));
    }

    if (gesture.name !== lastGesture && now - lastGestureTs > 600) {
      if (gesture.name === "victory") {
        controls.autoRotate = !controls.autoRotate;
      } else if (gesture.name === "open_palm") {
        desiredHandRotation = { pitch: 0.0, yaw: 0.0, roll: 0.0 };
      }
      lastGesture = gesture.name;
      lastGestureTs = now;
    }
  }

  function normalizeModel(model) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const safeMaxDim = maxDim > 0 ? maxDim : 1.0;

    // Uniform scale to keep wildly-sized source models visible.
    const scale = targetModelSpan / safeMaxDim;
    model.scale.setScalar(scale);

    // Recompute after scale, then center at origin.
    const scaledBox = new THREE.Box3().setFromObject(model);
    const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
    model.position.sub(scaledCenter);

    // Raise slightly above deck.
    const centeredBox = new THREE.Box3().setFromObject(model);
    const centeredSize = centeredBox.getSize(new THREE.Vector3());
    const centeredMin = centeredBox.min.clone();
    model.position.y += -centeredMin.y + centeredSize.y * 0.03;

    return {
      box: new THREE.Box3().setFromObject(model),
      size: centeredSize,
      maxDim: Math.max(centeredSize.x, centeredSize.y, centeredSize.z),
      scale,
    };
  }

  function fitCameraToModel(bounds, animate = false) {
    const box = bounds.box;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);
    const fov = (camera.fov * Math.PI) / 180;
    const distance = (maxDim * 0.5) / Math.tan(fov * 0.5);
    const padding = 1.55;
    const finalDistance = distance * padding;

    const dir = new THREE.Vector3(1.0, 0.38, 1.0).normalize();
    const targetPos = center.clone().add(dir.multiplyScalar(finalDistance));

    camera.near = Math.max(0.01, maxDim / 200);
    camera.far = Math.max(200, maxDim * 250);
    camera.updateProjectionMatrix();
    controls.minDistance = maxDim * 0.2;
    controls.maxDistance = maxDim * 20;
    controls.target.copy(center);

    if (!animate) {
      camera.position.copy(targetPos);
      controls.update();
      return;
    }

    const start = camera.position.clone();
    const durationMs = 450;
    const startTime = performance.now();
    const tick = (now) => {
      const t = clamp((now - startTime) / durationMs, 0, 1);
      camera.position.lerpVectors(start, targetPos, t);
      controls.update();
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function collectEngineMeshes(model, tokens = DEFAULT_ENGINE_TOKENS) {
    engineMeshes.length = 0;
    const lowerTokens = tokens.map((t) => t.toLowerCase());
    model.traverse((obj) => {
      if (!obj.isMesh) return;
      const n = (obj.name || "").toLowerCase();
      if (lowerTokens.some((token) => n.includes(token))) {
        engineMeshes.push(obj);
      }
    });
    if (debug) console.log(`Engine mesh matches: ${engineMeshes.length}`);
  }

  function backupMaterial(mesh) {
    if (!mesh.material || materialBackup.has(mesh.material)) return;
    materialBackup.set(mesh.material, {
      color: mesh.material.color ? mesh.material.color.clone() : null,
      emissive: mesh.material.emissive ? mesh.material.emissive.clone() : null,
      emissiveIntensity:
        typeof mesh.material.emissiveIntensity === "number" ? mesh.material.emissiveIntensity : null,
      opacity: typeof mesh.material.opacity === "number" ? mesh.material.opacity : 1.0,
      transparent: !!mesh.material.transparent,
      depthWrite: typeof mesh.material.depthWrite === "boolean" ? mesh.material.depthWrite : true,
    });
  }

  function restoreMaterial(mesh) {
    const backup = materialBackup.get(mesh.material);
    if (!backup) return;
    if (backup.color && mesh.material.color) mesh.material.color.copy(backup.color);
    if (backup.emissive && mesh.material.emissive) mesh.material.emissive.copy(backup.emissive);
    if (backup.emissiveIntensity != null) mesh.material.emissiveIntensity = backup.emissiveIntensity;
    mesh.material.opacity = backup.opacity;
    mesh.material.transparent = backup.transparent;
    mesh.material.depthWrite = backup.depthWrite;
    mesh.material.needsUpdate = true;
  }

  function clearEdgeOverlays() {
    for (const edge of edgeOverlays) {
      if (edge.parent) edge.parent.remove(edge);
      if (edge.geometry) edge.geometry.dispose();
      if (edge.material) edge.material.dispose();
    }
    edgeOverlays.length = 0;
  }

  function setMeshEdgeColor(mesh, color, opacity = 1.0) {
    for (const child of mesh.children || []) {
      if (!child.isLineSegments || !child.material) continue;
      if (child.material.color) child.material.color.set(color);
      child.material.opacity = opacity;
      child.material.transparent = true;
      child.material.needsUpdate = true;
    }
  }

  function resetAllEdgeColors() {
    if (!aircraft) return;
    aircraft.traverse((obj) => {
      if (!obj.isMesh) return;
      setMeshEdgeColor(obj, baseSkeletonColor, baseSkeletonLineOpacity);
    });
  }

  function applySkeletonView(
    model,
    { enabled = true, lineColor = "#7fffb2", lineOpacity = 0.95, surfaceOpacity = 0.08, thresholdAngle = 18 } = {}
  ) {
    clearEdgeOverlays();
    model.traverse((obj) => {
      if (!obj.isMesh || !obj.geometry || !obj.material) return;
      backupMaterial(obj);
      obj.material.transparent = enabled;
      obj.material.opacity = enabled ? surfaceOpacity : 1.0;
      obj.material.depthWrite = !enabled;
      obj.material.side = THREE.DoubleSide;
      obj.material.needsUpdate = true;

      if (!enabled) return;
      const edges = new THREE.EdgesGeometry(obj.geometry, thresholdAngle);
      const edgeMat = new THREE.LineBasicMaterial({
        color: new THREE.Color(lineColor),
        transparent: true,
        opacity: lineOpacity,
      });
      const edgeLines = new THREE.LineSegments(edges, edgeMat);
      edgeLines.name = `edge-${obj.name || "mesh"}`;
      edgeLines.renderOrder = 2;
      obj.add(edgeLines);
      edgeOverlays.push(edgeLines);
    });
  }

  function setAnomalyState(state = "normal") {
    const s = String(state).toLowerCase();
    const colorMap = {
      normal: new THREE.Color("#00ff66"),
      warning: new THREE.Color("#ff7a00"),
      critical: new THREE.Color("#ff1a1a"),
    };
    const glowMap = {
      normal: 0.55,
      warning: 0.85,
      critical: 1.1,
    };
    const targetColor = colorMap[s] || colorMap.normal;
    const glow = glowMap[s] ?? glowMap.normal;

    for (const mesh of engineMeshes) {
      if (!mesh.material) continue;
      backupMaterial(mesh);
      if (mesh.material.color) mesh.material.color.lerp(targetColor, 0.4);
      if (!mesh.material.emissive) mesh.material.emissive = targetColor.clone();
      mesh.material.emissive.copy(targetColor);
      mesh.material.emissiveIntensity = glow;
      mesh.material.needsUpdate = true;
    }
  }

  function clearAnomalyHighlight() {
    if (aircraft) {
      aircraft.traverse((obj) => {
        if (!obj.isMesh || !obj.material) return;
        restoreMaterial(obj);
      });
      if (skeletonEnabled) resetAllEdgeColors();
    } else {
      for (const mesh of engineMeshes) restoreMaterial(mesh);
    }
  }

  function highlightParts(tokens, color = "#ffd400", intensity = 0.9) {
    if (!aircraft) return [];
    const hits = [];
    const lowerTokens = (tokens || []).map((t) => String(t).toLowerCase());
    aircraft.traverse((obj) => {
      if (!obj.isMesh) return;
      const n = (obj.name || "").toLowerCase();
      if (!lowerTokens.some((token) => n.includes(token))) return;
      if (!obj.material) return;
      backupMaterial(obj);
      if (!obj.material.emissive) obj.material.emissive = new THREE.Color(color);
      obj.material.emissive.set(color);
      obj.material.emissiveIntensity = intensity;
      obj.material.needsUpdate = true;
      hits.push(obj.name || "(unnamed)");
    });
    return hits;
  }

  function highlightAnomalyRegions(
    regions = [],
    { color = "#ff1a1a", intensity = 1.15, clearExisting = true, fallbackToEngine = true, surfaceOpacity = 0.62 } = {}
  ) {
    if (!aircraft) return [];
    if (clearExisting) clearAnomalyHighlight();

    const regionDefs = Array.isArray(regions) ? regions : [];
    const matched = [];
    const colorObj = new THREE.Color(color);

    const normalized = regionDefs.map((region) => {
      const key = String(region?.key || region?.label || "").toLowerCase();
      const defaultTokens = DEFAULT_ANOMALY_REGION_TOKENS[key] || [];
      const tokens = (Array.isArray(region?.tokens) && region.tokens.length > 0)
        ? region.tokens.map((t) => String(t).toLowerCase())
        : defaultTokens;
      return {
        label: region?.label || key || "anomaly",
        tokens,
      };
    }).filter((r) => r.tokens.length > 0);

    aircraft.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const n = (obj.name || "").toLowerCase();
      const hitRegion = normalized.find((r) => r.tokens.some((token) => n.includes(token)));
      if (!hitRegion) return;
      backupMaterial(obj);
      if (obj.material.color) obj.material.color.lerp(colorObj, 0.55);
      if (!obj.material.emissive) obj.material.emissive = colorObj.clone();
      obj.material.emissive.copy(colorObj);
      obj.material.emissiveIntensity = intensity;
      obj.material.transparent = true;
      obj.material.opacity = surfaceOpacity;
      obj.material.depthWrite = true;
      obj.material.needsUpdate = true;
      if (skeletonEnabled) setMeshEdgeColor(obj, color, 1.0);
      matched.push({ mesh: obj.name || "(unnamed)", region: hitRegion.label });
    });

    if (fallbackToEngine && matched.length === 0) {
      for (const mesh of engineMeshes) {
        if (!mesh.material) continue;
        backupMaterial(mesh);
        if (mesh.material.color) mesh.material.color.lerp(colorObj, 0.55);
        if (!mesh.material.emissive) mesh.material.emissive = colorObj.clone();
        mesh.material.emissive.copy(colorObj);
        mesh.material.emissiveIntensity = intensity;
        mesh.material.transparent = true;
        mesh.material.opacity = surfaceOpacity;
        mesh.material.depthWrite = true;
        mesh.material.needsUpdate = true;
        if (skeletonEnabled) setMeshEdgeColor(mesh, color, 1.0);
        matched.push({ mesh: mesh.name || "(unnamed)", region: "engine-fallback" });
      }
    }

    return matched;
  }

  // External telemetry hook: angles in radians.
  function setTelemetryOrientation({ roll = 0, pitch = 0, yaw = 0, lerp = 0.18 } = {}) {
    if (!aircraft) return;
    const t = clamp(lerp, 0, 1);
    aircraft.rotation.x = lerpAngle(aircraft.rotation.x, pitch, t);
    aircraft.rotation.y = lerpAngle(aircraft.rotation.y, yaw, t);
    aircraft.rotation.z = lerpAngle(aircraft.rotation.z, roll, t);
  }

  function onResize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  window.addEventListener("resize", onResize);

  function startRenderLoop() {
    const clock = new THREE.Clock();
    const loop = () => {
      // Subtle deck glow pulse to keep dark UI alive.
      const t = clock.getElapsedTime();
      if (ground.material) {
        ground.material.emissive = new THREE.Color(0x00ff66);
        ground.material.emissiveIntensity = 0.012 + 0.008 * Math.sin(t * 1.4);
      }
      updateMediaPipeHandTracking();
      if (aircraft && handTrackingEnabled) {
        aircraft.rotation.x = THREE.MathUtils.lerp(aircraft.rotation.x, baseModelRotation.x + desiredHandRotation.pitch, 0.16);
        aircraft.rotation.y = THREE.MathUtils.lerp(aircraft.rotation.y, baseModelRotation.y + desiredHandRotation.yaw, 0.16);
        aircraft.rotation.z = THREE.MathUtils.lerp(aircraft.rotation.z, baseModelRotation.z + desiredHandRotation.roll, 0.14);
      }
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(loop);
    };
    loop();
  }

  async function load() {
    progressEl.textContent = "Loading model... 0%";

    const loader = new GLTFLoader();
    const gltf = await new Promise((resolve, reject) => {
      loader.load(
        modelUrl,
        (result) => resolve(result),
        (xhr) => {
          if (xhr.total) {
            const pct = Math.round((xhr.loaded / xhr.total) * 100);
            progressEl.textContent = `Loading model... ${pct}%`;
          } else {
            progressEl.textContent = `Loading model... ${Math.round(xhr.loaded / 1024)} KB`;
          }
        },
        (err) => reject(err)
      );
    });

    aircraft = gltf.scene || gltf.scenes?.[0];
    if (!aircraft) {
      throw new Error("GLTF loaded but no scene found.");
    }

    if (debug) {
      console.group("Rafale model hierarchy");
      debugHierarchy(aircraft, 0);
      console.groupEnd();
    }

    rootGroup.clear();
    rootGroup.add(aircraft);

    const bounds = normalizeModel(aircraft);
    baseModelRotation = new THREE.Euler(aircraft.rotation.x, aircraft.rotation.y, aircraft.rotation.z, "XYZ");
    fitCameraToModel(bounds, false);
    collectEngineMeshes(aircraft, config.engineTokens || DEFAULT_ENGINE_TOKENS);
    applySkeletonView(aircraft, {
      enabled: skeletonEnabled,
      lineColor: baseSkeletonColor,
      lineOpacity: baseSkeletonLineOpacity,
      surfaceOpacity: baseSkeletonSurfaceOpacity,
      thresholdAngle: config.skeletonThresholdAngle ?? 18,
    });

    progressEl.textContent = "Model loaded";
    setTimeout(() => {
      progressEl.style.opacity = "0.0";
      progressEl.style.transition = "opacity 400ms ease";
    }, 600);

    if (!animationFrame) startRenderLoop();
    if (handTrackingEnabled) initMediaPipeHandTracking();
    return { bounds, engineMeshes: [...engineMeshes] };
  }

  function dispose() {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = null;
    clearEdgeOverlays();
    window.removeEventListener("resize", onResize);
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentElement === container) {
      container.removeChild(renderer.domElement);
    }
    if (progressEl.parentElement === container) {
      container.removeChild(progressEl);
    }
    if (mpStream) {
      for (const track of mpStream.getTracks()) track.stop();
      mpStream = null;
    }
    if (mpVideo && mpVideo.parentElement === container) {
      container.removeChild(mpVideo);
    }
  }

  return {
    load,
    dispose,
    setAnomalyState,
    clearAnomalyHighlight,
    highlightParts,
    highlightAnomalyRegions,
    setSkeletonMode: (enabled = true, options = {}) => {
      skeletonEnabled = !!enabled;
      if (!aircraft) return;
      if (!skeletonEnabled) {
        applySkeletonView(aircraft, { enabled: false });
        return;
      }
      applySkeletonView(aircraft, {
        enabled: true,
        lineColor: options.lineColor || config.skeletonColor || "#7fffb2",
        lineOpacity: options.lineOpacity ?? config.skeletonLineOpacity ?? 0.95,
        surfaceOpacity: options.surfaceOpacity ?? config.skeletonSurfaceOpacity ?? 0.08,
        thresholdAngle: options.thresholdAngle ?? config.skeletonThresholdAngle ?? 18,
      });
    },
    setTelemetryOrientation,
    fitCamera: () => {
      if (!aircraft) return;
      const bounds = normalizeModel(aircraft);
      fitCameraToModel(bounds, true);
    },
    getEngineMeshNames: () => engineMeshes.map((m) => m.name || "(unnamed)"),
    getScene: () => scene,
    getCamera: () => camera,
    getControls: () => controls,
  };
}

// Optional global bridge for inline Streamlit scripts.
if (typeof window !== "undefined") {
  window.createRafaleViewer = createRafaleViewer;
}
