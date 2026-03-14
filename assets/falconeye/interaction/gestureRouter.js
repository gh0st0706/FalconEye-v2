(function(global) {
  'use strict';

  const FalconEye = global.FalconEye = global.FalconEye || {};
  FalconEye.interaction = FalconEye.interaction || {};

  class GestureRouter {
    constructor(options) {
      this.recognizer = options.recognizer;
      this.sphere = options.sphere;
      this.ui = options.ui;
      this._pinchState = new Map();
      this._grabState = new Map();
      this._pointHold = new Map();
      this._pointHoldFrames = options.pointHoldFrames || 10;
    }

    update(frame, camera) {
      if (!frame) return;
      this.sphere.update();
      const results = this.recognizer.update(frame, camera);
      for (const result of results) {
        const handId = result.handId;
        const inside = this.sphere.contains(result.points.palm);
        if (!inside) {
          this._resetHand(handId);
          continue;
        }

        const types = new Set(result.gestures.map((g) => g.type));
        this._handlePinch(handId, types.has('PINCH'), result);
        this._handleGrab(handId, types.has('GRAB'), result);
        this._handlePoint(handId, types.has('POINT'), result);

        if (types.has('SWIPE_LEFT')) this.ui.onSwipeLeft(result);
        if (types.has('SWIPE_RIGHT')) this.ui.onSwipeRight(result);
        if (types.has('OPEN_PALM')) this.ui.onOpenPalm(result);
      }
    }

    _handlePinch(handId, isPinch, result) {
      const active = this._pinchState.get(handId) || false;
      if (isPinch && !active) {
        this._pinchState.set(handId, true);
        this.ui.onPinchStart(result);
      } else if (isPinch && active) {
        this.ui.onPinchHold(result);
      } else if (!isPinch && active) {
        this._pinchState.set(handId, false);
        this.ui.onPinchEnd(result);
      }
    }

    _handleGrab(handId, isGrab, result) {
      const active = this._grabState.get(handId) || false;
      if (isGrab && !active) {
        this._grabState.set(handId, true);
        this.ui.onGrabStart(result);
      } else if (isGrab && active) {
        this.ui.onGrabHold(result);
      } else if (!isGrab && active) {
        this._grabState.set(handId, false);
        this.ui.onGrabEnd(result);
      }
    }

    _handlePoint(handId, isPoint, result) {
      const current = this._pointHold.get(handId) || 0;
      if (isPoint) {
        const next = current + 1;
        this._pointHold.set(handId, next);
        if (next === this._pointHoldFrames) {
          this.ui.onPointHold(result);
        }
      } else {
        this._pointHold.set(handId, 0);
      }
    }

    _resetHand(handId) {
      if (this._pinchState.get(handId)) {
        this._pinchState.set(handId, false);
        this.ui.onPinchEnd({ handId });
      }
      if (this._grabState.get(handId)) {
        this._grabState.set(handId, false);
        this.ui.onGrabEnd({ handId });
      }
      this._pointHold.set(handId, 0);
    }
  }

  FalconEye.interaction.GestureRouter = GestureRouter;
})(window);
