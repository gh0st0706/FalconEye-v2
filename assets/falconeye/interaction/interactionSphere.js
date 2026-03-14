(function(global) {
  'use strict';

  const FalconEye = global.FalconEye = global.FalconEye || {};
  FalconEye.interaction = FalconEye.interaction || {};

  class InteractionSphere {
    constructor(options) {
      this.camera = options.camera;
      this.radius = options.radius || 0.6;
      this.visible = options.visible || false;
      this.scene = options.scene || null;
      this.center = new global.THREE.Vector3();
      this._mesh = null;
      if (this.scene) this._buildMesh();
    }

    _buildMesh() {
      const geom = new global.THREE.SphereGeometry(this.radius, 24, 16);
      const mat = new global.THREE.MeshBasicMaterial({ color: 0x00ff66, wireframe: true, transparent: true, opacity: 0.12 });
      this._mesh = new global.THREE.Mesh(geom, mat);
      this._mesh.visible = this.visible;
      this.scene.add(this._mesh);
    }

    update() {
      if (!this.camera) return;
      this.center.copy(this.camera.position);
      if (this._mesh) this._mesh.position.copy(this.center);
    }

    contains(point) {
      if (!point) return false;
      return point.distanceTo(this.center) <= this.radius;
    }
  }

  FalconEye.interaction.InteractionSphere = InteractionSphere;
})(window);
