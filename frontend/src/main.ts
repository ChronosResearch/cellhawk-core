/**
 * CellHawk Digital Twin — application entry point.
 *
 * Initialises:
 * 1. MapLibre map with drone markers and danger grid
 * 2. Three.js 3D overlay for drone meshes and neural heatmap
 * 3. GCS WebSocket client for live telemetry
 */

import { MapComponent } from "./components/map.js";
import { ThreeOverlay } from "./components/three_overlay.js";
import { GcsClient } from "./workers/gcs_client.js";

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

function bootstrap(): void {
  // Read configuration from URL params or environment
  const params = new URLSearchParams(window.location.search);
  const gcsUrl  = params.get("gcs")    ?? "http://localhost:8000";
  const refLat  = parseFloat(params.get("lat") ?? "40.7128");
  const refLon  = parseFloat(params.get("lon") ?? "-74.0060");
  const droneIds = (params.get("drones") ?? "1,2,3")
    .split(",")
    .map(Number)
    .filter((n) => !isNaN(n));

  // ── Map ────────────────────────────────────────────────────────────────────
  const mapComponent = new MapComponent({
    containerId: "map",
    refLat,
    refLon,
    styleUrl:    "https://demotiles.maplibre.org/style.json",
    initialZoom: 15,
  });
  mapComponent.init();

  // ── Three.js overlay ───────────────────────────────────────────────────────
  const canvas = document.getElementById("overlay-canvas") as HTMLCanvasElement;
  if (canvas) {
    const overlay = new ThreeOverlay({ canvas, metersPerUnit: 0.001 });
    overlay.start();

    window.addEventListener("resize", () => {
      overlay.resize(window.innerWidth, window.innerHeight);
    });
    overlay.resize(window.innerWidth, window.innerHeight);
  }

  // ── GCS WebSocket client ───────────────────────────────────────────────────
  const client = new GcsClient({ baseUrl: gcsUrl, droneIds });
  client.connect();

  // Expose for debugging
  (window as unknown as Record<string, unknown>)["cellhawk"] = { mapComponent, client };
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
