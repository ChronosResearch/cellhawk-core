/**
 * Three.js 3D overlay for the CellHawk Digital Twin.
 *
 * Renders on a transparent canvas positioned over the MapLibre map:
 * - Drone meshes (cone geometry, oriented by heading, coloured by tier)
 * - Neural heatmap as a textured plane at ground level
 * - Altitude bars (vertical lines from ground to drone altitude)
 *
 * The overlay uses the same ENU→screen projection as the map component
 * so 3D objects stay aligned with the 2D map tiles.
 */

import * as THREE from "three";
import type { DroneState, HeatmapFrame } from "../types.js";
import { TIER_COLOURS } from "../types.js";
import { fleetStore } from "../store/fleet.js";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Parse a CSS hex colour string to a Three.js Color. */
function hexToThreeColor(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

// ─────────────────────────────────────────────────────────────────────────────
// ThreeOverlay
// ─────────────────────────────────────────────────────────────────────────────

export interface ThreeOverlayConfig {
  /** Canvas element to render into. */
  canvas:       HTMLCanvasElement;
  /** Scale factor: ENU metres → Three.js world units. */
  metersPerUnit: number;
}

export class ThreeOverlay {
  private readonly _cfg: ThreeOverlayConfig;
  private _renderer: THREE.WebGLRenderer;
  private _scene:    THREE.Scene;
  private _camera:   THREE.OrthographicCamera;
  private _droneMeshes: Map<number, THREE.Mesh> = new Map();
  private _heatmapMesh: THREE.Mesh | null = null;
  private _animFrameId: number | null = null;
  private readonly _unsubscribers: Array<() => void> = [];

  constructor(config: ThreeOverlayConfig) {
    this._cfg = config;

    this._renderer = new THREE.WebGLRenderer({
      canvas: config.canvas,
      alpha:  true,
      antialias: true,
    });
    this._renderer.setPixelRatio(window.devicePixelRatio);

    this._scene  = new THREE.Scene();
    this._camera = new THREE.OrthographicCamera(-500, 500, 500, -500, 0.1, 2000);
    this._camera.position.set(0, 0, 1000);
    this._camera.lookAt(0, 0, 0);

    // Ambient + directional light for drone cones
    this._scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(100, 100, 200);
    this._scene.add(dir);
  }

  /** Start rendering loop and subscribe to store updates. */
  start(): void {
    this._unsubscribers.push(
      fleetStore.onDroneUpdate.subscribe((d) => this._updateDroneMesh(d)),
      fleetStore.onHeatmapUpdate.subscribe((h) => this._updateHeatmap(h))
    );
    this._animate();
  }

  /** Stop rendering and clean up. */
  stop(): void {
    if (this._animFrameId !== null) {
      cancelAnimationFrame(this._animFrameId);
      this._animFrameId = null;
    }
    this._unsubscribers.forEach((fn) => fn());
    this._renderer.dispose();
  }

  resize(width: number, height: number): void {
    this._renderer.setSize(width, height, false);
    const aspect = width / height;
    const halfW = 500 * aspect;
    this._camera.left   = -halfW;
    this._camera.right  =  halfW;
    this._camera.updateProjectionMatrix();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _animate(): void {
    this._animFrameId = requestAnimationFrame(() => this._animate());
    this._renderer.render(this._scene, this._camera);
  }

  private _updateDroneMesh(state: DroneState): void {
    const scale = this._cfg.metersPerUnit;
    const x = state.position.east_m  * scale;
    const y = state.position.north_m * scale;
    const z = state.position.up_m    * scale;

    let mesh = this._droneMeshes.get(state.drone_id);
    if (!mesh) {
      // Cone geometry: tip points forward (along +Y in local space)
      const geo = new THREE.ConeGeometry(4 * scale, 10 * scale, 6);
      const mat = new THREE.MeshPhongMaterial({
        color: hexToThreeColor(TIER_COLOURS[state.tier]),
      });
      mesh = new THREE.Mesh(geo, mat);
      this._scene.add(mesh);
      this._droneMeshes.set(state.drone_id, mesh);
    }

    mesh.position.set(x, y, z);
    mesh.rotation.z = -state.heading_rad;

    // Update colour if tier changed
    const mat = mesh.material as THREE.MeshPhongMaterial;
    mat.color = hexToThreeColor(TIER_COLOURS[state.tier]);
  }

  private _updateHeatmap(frame: HeatmapFrame): void {
    // Build a DataTexture from the activation grid
    const w = frame.grid_width;
    const h = frame.grid_height;
    const data = new Uint8Array(w * h * 4);

    const maxAct = Math.max(...frame.activations, 1e-6);
    for (let i = 0; i < frame.activations.length; i++) {
      const norm = frame.activations[i]! / maxAct;
      // Hot colormap: low = blue, high = red
      data[i * 4 + 0] = Math.round(norm * 255);         // R
      data[i * 4 + 1] = Math.round((1 - norm) * 128);   // G
      data[i * 4 + 2] = Math.round((1 - norm) * 255);   // B
      data[i * 4 + 3] = Math.round(norm * 180);          // A (semi-transparent)
    }

    const texture = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
    texture.needsUpdate = true;

    const scale = this._cfg.metersPerUnit;
    const planeW = w * frame.cell_size_m * scale;
    const planeH = h * frame.cell_size_m * scale;

    if (this._heatmapMesh) {
      this._scene.remove(this._heatmapMesh);
      (this._heatmapMesh.material as THREE.MeshBasicMaterial).dispose();
    }

    const geo = new THREE.PlaneGeometry(planeW, planeH);
    const mat = new THREE.MeshBasicMaterial({
      map:         texture,
      transparent: true,
      depthWrite:  false,
    });
    this._heatmapMesh = new THREE.Mesh(geo, mat);
    this._heatmapMesh.position.set(
      (frame.grid_origin_east_m  + (w * frame.cell_size_m) / 2) * scale,
      (frame.grid_origin_north_m + (h * frame.cell_size_m) / 2) * scale,
      0.5 * scale
    );
    this._scene.add(this._heatmapMesh);
  }
}
