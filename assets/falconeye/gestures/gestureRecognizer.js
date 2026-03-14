(function(global) {
  'use strict';

  const FalconEye = global.FalconEye = global.FalconEye || {};
  FalconEye.gestures = FalconEye.gestures || {};

  class MovingAverage {
    constructor(size) {
      this.size = size;
      this.values = [];
    }

    add(vec) {
      this.values.push(vec.clone());
      if (this.values.length > this.size) this.values.shift();
      return this.average();
    }

    average() {
      if (this.values.length === 0) return null;
      const out = this.values[0].clone();
      for (let i = 1; i < this.values.length; i += 1) {
        out.add(this.values[i]);
      }
      return out.multiplyScalar(1.0 / this.values.length);
    }
  }

  class GestureRecognizer {
    constructor(options) {
      this.bufferSize = options.bufferSize || 10;
      this.smoothingWindow = options.smoothingWindow || 4;
      this.pinchThreshold = options.pinchThreshold || 0.04;
      this.openThreshold = options.openThreshold || 0.18;
      this.closedThreshold = options.closedThreshold || 0.11;
      this.swipeThreshold = options.swipeThreshold || 0.12;
      this.swipeCooldown = options.swipeCooldown || 12;
      this._buffers = new Map();
      this._smooth = new Map();
      this._cooldowns = new Map();
    }

    update(frame, camera) {
      if (!frame || !frame.hands) return [];
      const gestures = [];
      for (const hand of frame.hands) {
        const id = hand.id;
        if (!this._buffers.has(id)) {
          this._buffers.set(id, []);
          this._smooth.set(id, {
            palm: new MovingAverage(this.smoothingWindow),
            index: new MovingAverage(this.smoothingWindow),
            thumb: new MovingAverage(this.smoothingWindow)
          });
          this._cooldowns.set(id, 0);
        }

        const smoothers = this._smooth.get(id);
        const smoothedPalm = smoothers.palm.add(new global.THREE.Vector3(hand.world.palmCenter.x, hand.world.palmCenter.y, hand.world.palmCenter.z));
        const smoothedIndex = smoothers.index.add(new global.THREE.Vector3(hand.world.indexTip.x, hand.world.indexTip.y, hand.world.indexTip.z));
        const smoothedThumb = smoothers.thumb.add(new global.THREE.Vector3(hand.world.thumbTip.x, hand.world.thumbTip.y, hand.world.thumbTip.z));

        const buffer = this._buffers.get(id);
        const palmLocal = this._toCameraSpace(smoothedPalm, camera);
        buffer.push({ time: frame.time, palm: palmLocal });
        if (buffer.length > this.bufferSize) buffer.shift();

        const base = this._detectBaseGestures(hand, smoothedIndex, smoothedThumb, smoothedPalm);
        const swipe = this._detectSwipe(buffer, id);
        if (swipe) base.push(swipe);
        gestures.push({ handId: id, handedness: hand.handedness, gestures: base, points: { palm: smoothedPalm, index: smoothedIndex, thumb: smoothedThumb } });
      }
      return gestures;
    }

    _detectBaseGestures(hand, indexWorld, thumbWorld, palmWorld) {
      const lm = hand.landmarks;
      const pinchDist = this._dist(lm[4], lm[8]);
      const pinch = pinchDist < this.pinchThreshold;
      const open = this._openPalm(lm);
      const grab = this._grab(lm, pinch);
      const point = this._point(lm, pinch);

      const out = [];
      if (open) out.push({ type: 'OPEN_PALM', confidence: 0.8 });
      if (pinch) out.push({ type: 'PINCH', confidence: 0.9, data: { pinchDistance: pinchDist } });
      if (grab) out.push({ type: 'GRAB', confidence: 0.75 });
      if (point) out.push({ type: 'POINT', confidence: 0.7 });
      return out;
    }

    _detectSwipe(buffer, id) {
      const cooldown = this._cooldowns.get(id) || 0;
      if (cooldown > 0) {
        this._cooldowns.set(id, cooldown - 1);
        return null;
      }
      if (buffer.length < this.bufferSize) return null;
      const start = buffer[0].palm;
      const end = buffer[buffer.length - 1].palm;
      const dx = end.x - start.x;
      const dy = Math.abs(end.y - start.y);
      if (Math.abs(dx) < this.swipeThreshold || dy > this.swipeThreshold * 0.6) return null;
      this._cooldowns.set(id, this.swipeCooldown);
      return { type: dx > 0 ? 'SWIPE_RIGHT' : 'SWIPE_LEFT', confidence: 0.8 };
    }

    _openPalm(lm) {
      const palm = this._palmCenter(lm);
      const tips = [8, 12, 16, 20];
      let extended = 0;
      for (let i = 0; i < tips.length; i += 1) {
        if (this._dist(palm, lm[tips[i]]) > this.openThreshold) extended += 1;
      }
      return extended >= 3;
    }

    _grab(lm, pinch) {
      if (pinch) return false;
      const palm = this._palmCenter(lm);
      const tips = [8, 12, 16, 20];
      let folded = 0;
      for (let i = 0; i < tips.length; i += 1) {
        if (this._dist(palm, lm[tips[i]]) < this.closedThreshold) folded += 1;
      }
      return folded >= 3;
    }

    _point(lm, pinch) {
      if (pinch) return false;
      const palm = this._palmCenter(lm);
      const indexOut = this._dist(palm, lm[8]) > this.openThreshold;
      const others = [12, 16, 20];
      let curled = 0;
      for (let i = 0; i < others.length; i += 1) {
        if (this._dist(palm, lm[others[i]]) < this.closedThreshold) curled += 1;
      }
      return indexOut && curled >= 2;
    }

    _palmCenter(lm) {
      const idx = [0, 5, 9, 13, 17];
      let x = 0;
      let y = 0;
      let z = 0;
      for (let i = 0; i < idx.length; i += 1) {
        const p = lm[idx[i]];
        x += p.x;
        y += p.y;
        z += p.z || 0;
      }
      return { x: x / idx.length, y: y / idx.length, z: z / idx.length };
    }

    _dist(a, b) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = (a.z || 0) - (b.z || 0);
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    _toCameraSpace(world, camera) {
      const vec = world.clone();
      camera.updateMatrixWorld();
      vec.applyMatrix4(camera.matrixWorldInverse);
      return vec;
    }
  }

  FalconEye.gestures.GestureRecognizer = GestureRecognizer;
})(window);
