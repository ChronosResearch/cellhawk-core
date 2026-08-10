/**
 * Fleet state store — reactive single source of truth.
 *
 * Holds the latest DroneState for every connected drone and notifies
 * subscribers on every update.  Designed for direct use by map and
 * Three.js overlay components without a framework dependency.
 */

import type { DroneState, DangerEntry, HeatmapFrame } from "../types.js";

type Listener<T> = (value: T) => void;

class EventEmitter<T> {
  private readonly _listeners: Set<Listener<T>> = new Set();

  subscribe(fn: Listener<T>): () => void {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  emit(value: T): void {
    this._listeners.forEach((fn) => fn(value));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FleetStore
// ─────────────────────────────────────────────────────────────────────────────

export class FleetStore {
  private readonly _drones: Map<number, DroneState> = new Map();
  private readonly _danger: DangerEntry[] = [];
  private readonly _heatmaps: Map<number, HeatmapFrame> = new Map();

  readonly onDroneUpdate = new EventEmitter<DroneState>();
  readonly onDangerUpdate = new EventEmitter<DangerEntry[]>();
  readonly onHeatmapUpdate = new EventEmitter<HeatmapFrame>();

  // ── Drone state ────────────────────────────────────────────────────────────

  updateDrone(state: DroneState): void {
    this._drones.set(state.drone_id, state);
    this.onDroneUpdate.emit(state);
  }

  getDrone(id: number): DroneState | undefined {
    return this._drones.get(id);
  }

  getAllDrones(): DroneState[] {
    return Array.from(this._drones.values());
  }

  get droneCount(): number {
    return this._drones.size;
  }

  // ── Danger grid ────────────────────────────────────────────────────────────

  addDangerEntry(entry: DangerEntry): void {
    this._danger.push(entry);
    // Evict expired entries (TTL-based)
    const now = Date.now() / 1000;
    const live = this._danger.filter(
      (e) => now - e.timestamp_s < e.ttl_s
    );
    this._danger.length = 0;
    this._danger.push(...live);
    this.onDangerUpdate.emit(this.getDangerEntries());
  }

  getDangerEntries(): DangerEntry[] {
    return [...this._danger];
  }

  getDangerInRadius(
    east_m: number,
    north_m: number,
    radius_m: number
  ): DangerEntry[] {
    return this._danger.filter((e) => {
      const de = e.east_m - east_m;
      const dn = e.north_m - north_m;
      return Math.hypot(de, dn) <= radius_m;
    });
  }

  // ── Neural heatmap ─────────────────────────────────────────────────────────

  updateHeatmap(frame: HeatmapFrame): void {
    this._heatmaps.set(frame.drone_id, frame);
    this.onHeatmapUpdate.emit(frame);
  }

  getHeatmap(droneId: number): HeatmapFrame | undefined {
    return this._heatmaps.get(droneId);
  }
}

// Singleton store shared across all components
export const fleetStore = new FleetStore();
