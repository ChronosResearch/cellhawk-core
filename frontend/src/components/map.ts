/**
 * MapLibre GL JS map component for the CellHawk Digital Twin.
 *
 * Renders:
 * - Live drone positions as GeoJSON point features with tier-coloured markers
 * - Navigation tier badge (GNSS / Cellular / SLAM) per drone
 * - Danger Grid hazard circles with threat-type colour coding
 * - Spoofing alert overlay when spoofing_suspected = true
 *
 * The map uses ENU→WGS-84 conversion for all feature coordinates.
 * Reference origin is configurable (default: New York City).
 */

import maplibregl from "maplibre-gl";
import type { Map as MapLibreMap, GeoJSONSource } from "maplibre-gl";
import type { Feature, Point } from "geojson";

import { TIER_COLOURS, TIER_LABELS, THREAT_COLOURS } from "../types.js";
import { fleetStore } from "../store/fleet.js";

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate helpers
// ─────────────────────────────────────────────────────────────────────────────

const M_PER_DEG_LAT = 111_320.0;

function enuToLngLat(
  east_m: number,
  north_m: number,
  refLat: number,
  refLon: number
): [number, number] {
  const mPerDegLon = M_PER_DEG_LAT * Math.cos((refLat * Math.PI) / 180);
  const lat = refLat + north_m / M_PER_DEG_LAT;
  const lon = refLon + east_m / mPerDegLon;
  return [lon, lat];
}

// ─────────────────────────────────────────────────────────────────────────────
// MapComponent
// ─────────────────────────────────────────────────────────────────────────────

export interface MapComponentConfig {
  containerId: string;
  refLat:      number;
  refLon:      number;
  /** MapLibre style URL or style object. */
  styleUrl:    string;
  initialZoom: number;
}

const DEFAULT_MAP_CONFIG: MapComponentConfig = {
  containerId: "map",
  refLat:      40.7128,
  refLon:      -74.0060,
  styleUrl:    "https://demotiles.maplibre.org/style.json",
  initialZoom: 15,
};

export class MapComponent {
  private _map: MapLibreMap | null = null;
  private readonly _cfg: MapComponentConfig;
  private readonly _unsubscribers: Array<() => void> = [];

  constructor(config: Partial<MapComponentConfig> = {}) {
    this._cfg = { ...DEFAULT_MAP_CONFIG, ...config };
  }

  /** Initialise the map and subscribe to store updates. */
  init(): void {
    this._map = new maplibregl.Map({
      container: this._cfg.containerId,
      style:     this._cfg.styleUrl,
      center:    [this._cfg.refLon, this._cfg.refLat],
      zoom:      this._cfg.initialZoom,
    });

    this._map.on("load", () => {
      this._addSources();
      this._addLayers();
      this._subscribeToStore();
    });
  }

  destroy(): void {
    this._unsubscribers.forEach((fn) => fn());
    this._map?.remove();
    this._map = null;
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _addSources(): void {
    if (!this._map) return;

    this._map.addSource("drones", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });

    this._map.addSource("danger", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
  }

  private _addLayers(): void {
    if (!this._map) return;

    // Drone circles — colour by tier
    this._map.addLayer({
      id:     "drone-circles",
      type:   "circle",
      source: "drones",
      paint: {
        "circle-radius": 10,
        "circle-color":  ["get", "tier_colour"],
        "circle-stroke-width": 2,
        "circle-stroke-color": "#ffffff",
        "circle-opacity": 0.9,
      },
    });

    // Drone labels — tier badge + ID
    this._map.addLayer({
      id:     "drone-labels",
      type:   "symbol",
      source: "drones",
      layout: {
        "text-field":  ["concat", ["get", "tier_label"], " #", ["get", "drone_id"]],
        "text-size":   11,
        "text-offset": [0, 1.5],
        "text-anchor": "top",
      },
      paint: {
        "text-color": "#1f2937",
        "text-halo-color": "#ffffff",
        "text-halo-width": 1.5,
      },
    });

    // Danger grid circles
    this._map.addLayer({
      id:     "danger-circles",
      type:   "circle",
      source: "danger",
      paint: {
        "circle-radius":       ["*", ["get", "severity"], 20],
        "circle-color":        ["get", "threat_colour"],
        "circle-opacity":      0.35,
        "circle-stroke-width": 1,
        "circle-stroke-color": ["get", "threat_colour"],
      },
    });
  }

  private _subscribeToStore(): void {
    this._unsubscribers.push(
      fleetStore.onDroneUpdate.subscribe(() => this._refreshDrones()),
      fleetStore.onDangerUpdate.subscribe(() => this._refreshDanger())
    );
  }

  private _refreshDrones(): void {
    const source = this._map?.getSource("drones") as GeoJSONSource | undefined;
    if (!source) return;

    const features: Feature<Point>[] = fleetStore.getAllDrones().map((d) => {
      const [lon, lat] = enuToLngLat(
        d.position.east_m,
        d.position.north_m,
        this._cfg.refLat,
        this._cfg.refLon
      );
      return {
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {
          drone_id:    d.drone_id,
          tier_colour: TIER_COLOURS[d.tier],
          tier_label:  TIER_LABELS[d.tier],
          jnr_db:      d.jnr_db,
          rms_m:       d.rms_position_error_m,
          spoofing:    d.spoofing_suspected,
        },
      };
    });

    source.setData({ type: "FeatureCollection", features });
  }

  private _refreshDanger(): void {
    const source = this._map?.getSource("danger") as GeoJSONSource | undefined;
    if (!source) return;

    const features: Feature<Point>[] = fleetStore.getDangerEntries().map((e) => {
      const [lon, lat] = enuToLngLat(
        e.east_m,
        e.north_m,
        this._cfg.refLat,
        this._cfg.refLon
      );
      return {
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {
          threat_colour: THREAT_COLOURS[e.threat_type],
          severity:      e.severity,
          threat_type:   e.threat_type,
          drone_id:      e.drone_id,
        },
      };
    });

    source.setData({ type: "FeatureCollection", features });
  }
}
