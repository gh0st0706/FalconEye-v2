(function(global) {
  'use strict';

  const FalconEye = global.FalconEye = global.FalconEye || {};
  FalconEye.ui = FalconEye.ui || {};

  class HologramControls {
    constructor(options) {
      this.model = options.model;
      this.camera = options.camera;
      this.controls = options.controls;
      this.scene = options.scene;
      this.statusEl = options.statusEl || null;
      this.overlayRoot = options.overlayRoot || document.body;
      this._pinchAnchor = null;
      this._grabAnchor = null;
      this._radialOpen = false;
      this._buildOverlays();
      this._baseRotation = this.model ? this.model.rotation.clone() : null;
      this._basePosition = this.model ? this.model.position.clone() : null;
    }

    update() {
      this._updateHoverIndicator();
    }

    onPinchStart(result) {
      if (!this.model) return;
      this._pinchAnchor = {
        pos: result.points.index.clone(),
        rot: this.model.rotation.clone()
      };
      this._setStatus('Pinch start');
    }

    onPinchHold(result) {
      if (!this.model || !this._pinchAnchor) return;
      const delta = result.points.index.clone().sub(this._pinchAnchor.pos);
      this.model.rotation.y = this._pinchAnchor.rot.y + delta.x * -4.5;
      this.model.rotation.x = this._pinchAnchor.rot.x + delta.y * 4.0;
      this._setStatus('Pinch rotate');
    }

    onPinchEnd() {
      this._pinchAnchor = null;
      this._setStatus('Pinch release');
    }

    onGrabStart(result) {
      if (!this.model) return;
      this._grabAnchor = {
        pos: result.points.palm.clone(),
        base: this.model.position.clone()
      };
      this._setStatus('Grab start');
    }

    onGrabHold(result) {
      if (!this.model || !this._grabAnchor) return;
      const delta = result.points.palm.clone().sub(this._grabAnchor.pos);
      this.model.position.x = this._grabAnchor.base.x + delta.x * 4.0;
      this.model.position.y = this._grabAnchor.base.y + delta.y * 3.5;
      this.model.position.z = this._grabAnchor.base.z + delta.z * 4.0;
      this._setStatus('Grab move');
    }

    onGrabEnd() {
      this._grabAnchor = null;
      this._setStatus('Grab release');
    }

    onSwipeLeft() {
      this._setStatus('Swipe left: telemetry panel');
    }

    onSwipeRight() {
      this._setStatus('Swipe right: engine diagnostics');
    }

    onPointHold(result) {
      this._showHoverProgress(result.points.index, true);
      this._setStatus('Point hold: select');
    }

    onOpenPalm() {
      this._toggleRadialMenu();
      this.resetView();
      this._setStatus('Open palm: reset');
    }

    resetView() {
      if (!this.model) return;
      if (this._baseRotation) this.model.rotation.copy(this._baseRotation);
      if (this._basePosition) this.model.position.copy(this._basePosition);
      if (this.controls) this.controls.update();
    }

    _buildOverlays() {
      this._hoverEl = document.createElement('div');
      this._hoverEl.style.position = 'absolute';
      this._hoverEl.style.width = '48px';
      this._hoverEl.style.height = '48px';
      this._hoverEl.style.border = '2px solid #00ff66';
      this._hoverEl.style.borderRadius = '50%';
      this._hoverEl.style.pointerEvents = 'none';
      this._hoverEl.style.opacity = '0';
      this._hoverEl.style.transition = 'opacity 120ms ease';
      this._hoverEl.style.transform = 'translate(-50%, -50%)';
      this.overlayRoot.appendChild(this._hoverEl);

      this._radial = document.createElement('div');
      this._radial.style.position = 'absolute';
      this._radial.style.width = '240px';
      this._radial.style.height = '240px';
      this._radial.style.border = '1px solid rgba(0, 255, 102, 0.5)';
      this._radial.style.borderRadius = '50%';
      this._radial.style.display = 'none';
      this._radial.style.pointerEvents = 'none';
      this._radial.style.color = '#00ff66';
      this._radial.style.fontFamily = 'Rajdhani, sans-serif';
      this._radial.style.textAlign = 'center';
      this._radial.style.lineHeight = '240px';
      this._radial.innerText = 'RADIAL MENU';
      this.overlayRoot.appendChild(this._radial);
    }

    _showHoverProgress(worldPoint, locked) {
      if (!this.camera || !worldPoint) return;
      const screen = worldPoint.clone().project(this.camera);
      const x = (screen.x * 0.5 + 0.5) * this.overlayRoot.clientWidth;
      const y = (-screen.y * 0.5 + 0.5) * this.overlayRoot.clientHeight;
      this._hoverEl.style.left = x + 'px';
      this._hoverEl.style.top = y + 'px';
      this._hoverEl.style.opacity = locked ? '1' : '0.6';
    }

    _updateHoverIndicator() {
      if (!this._hoverEl) return;
    }

    _toggleRadialMenu() {
      this._radialOpen = !this._radialOpen;
      if (this._radialOpen) {
        this._radial.style.display = 'block';
        this._radial.style.left = '50%';
        this._radial.style.top = '50%';
        this._radial.style.transform = 'translate(-50%, -50%)';
      } else {
        this._radial.style.display = 'none';
      }
    }

    _setStatus(text) {
      if (this.statusEl) this.statusEl.textContent = text;
    }
  }

  FalconEye.ui.HologramControls = HologramControls;
})(window);
