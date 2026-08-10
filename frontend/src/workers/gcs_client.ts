/**
 * GCS WebSocket client with automatic reconnection.
 *
 * Connects to the GCS telemetry WebSocket endpoint and deserialises
 * incoming JSON frames into the FleetStore.
 *
 * Reconnection uses exponential back-off with jitter (max 30 s).
 */

import type { DroneState, DangerEntry, HeatmapFrame } from "../types.js";
import { fleetStore } from "../store/fleet.js";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface GcsClientConfig {
  /** Base URL of the GCS API, e.g. "http://localhost:8000". */
  baseUrl:          string;
  /** Drone IDs to subscribe to.  Empty = subscribe to all. */
  droneIds:         number[];
  /** Initial reconnect delay (ms). */
  reconnectBaseMs:  number;
  /** Maximum reconnect delay (ms). */
  reconnectMaxMs:   number;
}

const DEFAULT_CONFIG: GcsClientConfig = {
  baseUrl:         "http://localhost:8000",
  droneIds:        [],
  reconnectBaseMs: 500,
  reconnectMaxMs:  30_000,
};

// ─────────────────────────────────────────────────────────────────────────────
// GcsClient
// ─────────────────────────────────────────────────────────────────────────────

export class GcsClient {
  private readonly _cfg: GcsClientConfig;
  private _sockets: Map<number, WebSocket> = new Map();
  private _reconnectDelays: Map<number, number> = new Map();
  private _stopped = false;

  constructor(config: Partial<GcsClientConfig> = {}) {
    this._cfg = { ...DEFAULT_CONFIG, ...config };
  }

  /** Connect to telemetry streams for all configured drone IDs. */
  connect(droneIds?: number[]): void {
    const ids = droneIds ?? this._cfg.droneIds;
    if (ids.length === 0) {
      console.warn("[GcsClient] No drone IDs configured — nothing to connect");
      return;
    }
    this._stopped = false;
    ids.forEach((id) => this._connectDrone(id));
  }

  /** Disconnect all WebSocket connections. */
  disconnect(): void {
    this._stopped = true;
    this._sockets.forEach((ws) => ws.close());
    this._sockets.clear();
  }

  /** Whether the client has at least one open connection. */
  get isConnected(): boolean {
    return Array.from(this._sockets.values()).some(
      (ws) => ws.readyState === WebSocket.OPEN
    );
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _connectDrone(droneId: number): void {
    if (this._stopped) return;

    const wsUrl = this._cfg.baseUrl
      .replace(/^http/, "ws")
      .replace(/\/$/, "");
    const url = `${wsUrl}/telemetry/${droneId}`;

    const ws = new WebSocket(url);
    this._sockets.set(droneId, ws);

    ws.onopen = () => {
      this._reconnectDelays.set(droneId, this._cfg.reconnectBaseMs);
      console.info(`[GcsClient] Connected to drone ${droneId}`);
    };

    ws.onmessage = (ev: MessageEvent) => {
      try {
        const frame: unknown = JSON.parse(ev.data as string);
        this._dispatch(frame);
      } catch {
        // Malformed frame — ignore
      }
    };

    ws.onclose = () => {
      this._sockets.delete(droneId);
      if (!this._stopped) {
        this._scheduleReconnect(droneId);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  private _scheduleReconnect(droneId: number): void {
    const delay = this._reconnectDelays.get(droneId) ?? this._cfg.reconnectBaseMs;
    const jitter = Math.random() * delay * 0.2;
    const next = Math.min(delay * 2 + jitter, this._cfg.reconnectMaxMs);
    this._reconnectDelays.set(droneId, next);

    setTimeout(() => this._connectDrone(droneId), delay);
  }

  private _dispatch(frame: unknown): void {
    if (!frame || typeof frame !== "object") return;
    const f = frame as Record<string, unknown>;

    // Route by frame type field
    if ("tier" in f && "position" in f) {
      fleetStore.updateDrone(f as unknown as DroneState);
    } else if ("threat_type" in f && "severity" in f) {
      fleetStore.addDangerEntry(f as unknown as DangerEntry);
    } else if ("activations" in f && "grid_width" in f) {
      fleetStore.updateHeatmap(f as unknown as HeatmapFrame);
    }
  }
}
