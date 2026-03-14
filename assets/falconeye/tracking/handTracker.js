(function(global) {
  'use strict';

  const FalconEye = global.FalconEye = global.FalconEye || {};
  FalconEye.tracking = FalconEye.tracking || {};

  class HandTracker {
    constructor(options) {
      this.camera = options.camera;
      this.video = options.video;
      this.onFrame = options.onFrame || function() {};
      this.maxHands = options.maxHands || 2;
      this.depthBase = options.depthBase || 0.45;
      this.depthScale = options.depthScale || 0.65;
      this.smoothing = options.smoothing || 0.0;
      this._hands = null;
      this._running = false;
      this._lastFrame = null;
    }

    async start() {
      if (!global.Hands) {
        throw new Error('MediaPipe Hands not available');
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access unavailable');
      }

      this._hands = new global.Hands({
        locateFile: (file) => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + file
      });
      this._hands.setOptions({
        maxNumHands: this.maxHands,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.6
      });

      this._hands.onResults((results) => {
        const frame = this._buildFrame(results);
        if (!frame) return;
        this._lastFrame = frame;
        this.onFrame(frame);
      });

      if (!this.video) {
        this.video = document.createElement('video');
        this.video.autoplay = true;
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.style.display = 'none';
        document.body.appendChild(this.video);
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      this.video.srcObject = stream;
      await this.video.play();

      this._running = true;
      const pump = async () => {
        if (!this._running) return;
        if (this.video.readyState >= 2) {
          await this._hands.send({ image: this.video });
        }
        requestAnimationFrame(pump);
      };
      pump();
    }

    stop() {
      this._running = false;
      if (this.video && this.video.srcObject) {
        const tracks = this.video.srcObject.getTracks();
        tracks.forEach((t) => t.stop());
        this.video.srcObject = null;
      }
    }

    getLastFrame() {
      return this._lastFrame;
    }

    _buildFrame(results) {
      if (!results || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return null;
      const hands = [];
      for (let i = 0; i < results.multiHandLandmarks.length; i += 1) {
        const landmarks = results.multiHandLandmarks[i];
        if (!landmarks || landmarks.length < 21) continue;
        const handedness = results.multiHandedness && results.multiHandedness[i]
          ? results.multiHandedness[i].label
          : 'Unknown';
        const id = handedness + '_' + i;
        hands.push(this._mapHand(id, handedness, landmarks));
      }
      if (hands.length === 0) return null;
      return { time: performance.now(), hands };
    }

    _mapHand(id, handedness, landmarks) {
      const palmCenter = this._palmCenter(landmarks);
      const worldPalm = this._toWorld(palmCenter);
      return {
        id,
        handedness,
        landmarks,
        world: {
          wrist: this._toWorld(landmarks[0]),
          indexTip: this._toWorld(landmarks[8]),
          thumbTip: this._toWorld(landmarks[4]),
          palmCenter: worldPalm
        },
        palmCenter
      };
    }

    _palmCenter(landmarks) {
      const idx = [0, 5, 9, 13, 17];
      let x = 0;
      let y = 0;
      let z = 0;
      for (let i = 0; i < idx.length; i += 1) {
        const lm = landmarks[idx[i]];
        x += lm.x;
        y += lm.y;
        z += lm.z || 0;
      }
      return { x: x / idx.length, y: y / idx.length, z: z / idx.length };
    }

    _toWorld(lm) {
      const THREE = global.THREE;
      if (!THREE || !this.camera) {
        return { x: 0, y: 0, z: 0 };
      }
      const ndc = new THREE.Vector3((lm.x - 0.5) * 2.0, (0.5 - lm.y) * 2.0, 0.5);
      ndc.unproject(this.camera);
      const direction = ndc.sub(this.camera.position).normalize();
      const depth = Math.max(0.1, this.depthBase + (-lm.z || 0) * this.depthScale);
      const world = this.camera.position.clone().add(direction.multiplyScalar(depth));
      return world;
    }
  }

  FalconEye.tracking.HandTracker = HandTracker;
})(window);
