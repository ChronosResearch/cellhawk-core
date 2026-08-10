/**
 * Frontend unit tests — FleetStore, coordinate helpers, GcsClient dispatch.
 *
 * Run with: npm test  (from frontend/)
 * Uses Vitest + jsdom (no browser required).
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { FleetStore } from "../src/store/fleet.js";
import type { DroneState, DangerEntry, HeatmapFrame } from "../src/types.js";
import { TIER_COLOURS, TIER_LABELS, THREAT_COLOURS } from "../src/types.js";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function makeDrone(id: number, tier: 1 | 2 | 3 = 1): DroneState {
  return {
    drone_id:             id,
    timestamp_s:          1000.0,
    position:             { east_m: id * 10, north_m: id * 5, up_m: 50 },
    heading_rad:          0.0,
    jnr_db:               3.0,
    tier,
    rms_position_error_m: 4.5,
    battery_voltage_v:    14.8,
    spoofing_suspected:   false,
  };
}

function makeDanger(id: number): DangerEntry {
  return {
    drone_id:    id,
    east_m:      100.0,
    north_m:     200.0,
    severity:    0.75,
    threat_type: "RF_JAMMING",
    ttl_s:       60.0,
    timestamp_s: Date.now() / 1000,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// FleetStore
// ─────────────────────────────────────────────────────────────────────────────

describe("FleetStore", () => {
  let store: FleetStore;

  beforeEach(() => {
    store = new FleetStore();
  });

  it("starts empty", () => {
    expect(store.droneCount).toBe(0);
    expect(store.getAllDrones()).toHaveLength(0);
    expect(store.getDangerEntries()).toHaveLength(0);
  });

  it("stores and retrieves a drone state", () => {
    const d = makeDrone(1);
    store.updateDrone(d);
    expect(store.droneCount).toBe(1);
    expect(store.getDrone(1)).toEqual(d);
  });

  it("overwrites drone state on update", () => {
    store.updateDrone(makeDrone(1, 1));
    store.updateDrone(makeDrone(1, 3));
    expect(store.getDrone(1)?.tier).toBe(3);
    expect(store.droneCount).toBe(1);
  });

  it("tracks multiple drones independently", () => {
    store.updateDrone(makeDrone(1));
    store.updateDrone(makeDrone(2));
    store.updateDrone(makeDrone(3));
    expect(store.droneCount).toBe(3);
    expect(store.getDrone(2)?.position.east_m).toBe(20);
  });

  it("emits onDroneUpdate on each update", () => {
    const received: DroneState[] = [];
    store.onDroneUpdate.subscribe((d) => received.push(d));
    store.updateDrone(makeDrone(1));
    store.updateDrone(makeDrone(2));
    expect(received).toHaveLength(2);
  });

  it("unsubscribe stops receiving events", () => {
    const received: DroneState[] = [];
    const unsub = store.onDroneUpdate.subscribe((d) => received.push(d));
    store.updateDrone(makeDrone(1));
    unsub();
    store.updateDrone(makeDrone(2));
    expect(received).toHaveLength(1);
  });

  it("stores danger entries and emits update", () => {
    const received: DangerEntry[][] = [];
    store.onDangerUpdate.subscribe((e) => received.push(e));
    store.addDangerEntry(makeDanger(1));
    expect(store.getDangerEntries()).toHaveLength(1);
    expect(received).toHaveLength(1);
  });

  it("getDangerInRadius filters by distance", () => {
    store.addDangerEntry({ ...makeDanger(1), east_m: 0, north_m: 0 });
    store.addDangerEntry({ ...makeDanger(2), east_m: 1000, north_m: 1000 });
    const near = store.getDangerInRadius(0, 0, 50);
    expect(near).toHaveLength(1);
    expect(near[0]?.drone_id).toBe(1);
  });

  it("stores and retrieves heatmap frames", () => {
    const frame: HeatmapFrame = {
      drone_id: 1, timestamp_s: 0,
      activations: [0.1, 0.5, 0.9],
      grid_width: 3, grid_height: 1,
      grid_origin_east_m: 0, grid_origin_north_m: 0,
      cell_size_m: 5,
    };
    store.updateHeatmap(frame);
    expect(store.getHeatmap(1)).toEqual(frame);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Type constants
// ─────────────────────────────────────────────────────────────────────────────

describe("TIER_COLOURS", () => {
  it("has entries for all three tiers", () => {
    expect(TIER_COLOURS[1]).toMatch(/^#/);
    expect(TIER_COLOURS[2]).toMatch(/^#/);
    expect(TIER_COLOURS[3]).toMatch(/^#/);
  });

  it("tier 1 is green (GNSS = safe)", () => {
    expect(TIER_COLOURS[1]).toBe("#22c55e");
  });

  it("tier 3 is red (SLAM only = degraded)", () => {
    expect(TIER_COLOURS[3]).toBe("#ef4444");
  });
});

describe("TIER_LABELS", () => {
  it("labels match paper tier names", () => {
    expect(TIER_LABELS[1]).toBe("GNSS");
    expect(TIER_LABELS[2]).toBe("Cellular");
    expect(TIER_LABELS[3]).toBe("SLAM");
  });
});

describe("THREAT_COLOURS", () => {
  it("has entries for all threat types", () => {
    const types = ["RF_JAMMING", "GPS_SPOOFING", "HUNTER_DRONE", "OBSTACLE", "COMMS_DEGRADED"] as const;
    types.forEach((t) => expect(THREAT_COLOURS[t]).toMatch(/^#/));
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate helper (inline test — no import needed)
// ─────────────────────────────────────────────────────────────────────────────

describe("ENU to LngLat conversion", () => {
  const M_PER_DEG_LAT = 111_320.0;
  const refLat = 40.7128;
  const refLon = -74.0060;

  function enuToLngLat(east_m: number, north_m: number): [number, number] {
    const mPerDegLon = M_PER_DEG_LAT * Math.cos((refLat * Math.PI) / 180);
    return [refLon + east_m / mPerDegLon, refLat + north_m / M_PER_DEG_LAT];
  }

  it("origin maps to reference lat/lon", () => {
    const [lon, lat] = enuToLngLat(0, 0);
    expect(lon).toBeCloseTo(refLon, 9);
    expect(lat).toBeCloseTo(refLat, 9);
  });

  it("100 m north increases latitude", () => {
    const [, lat] = enuToLngLat(0, 100);
    expect(lat).toBeGreaterThan(refLat);
    expect(lat - refLat).toBeCloseTo(100 / M_PER_DEG_LAT, 6);
  });

  it("100 m east increases longitude", () => {
    const [lon] = enuToLngLat(100, 0);
    expect(lon).toBeGreaterThan(refLon);
  });

  it("round-trip is consistent", () => {
    const east = 350.0;
    const north = 220.0;
    const mPerDegLon = M_PER_DEG_LAT * Math.cos((refLat * Math.PI) / 180);
    const [lon, lat] = enuToLngLat(east, north);
    const east_back  = (lon - refLon) * mPerDegLon;
    const north_back = (lat - refLat) * M_PER_DEG_LAT;
    expect(east_back).toBeCloseTo(east, 4);
    expect(north_back).toBeCloseTo(north, 4);
  });
});
