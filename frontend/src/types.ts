/**
 * CellHawk Digital Twin — shared domain types.
 *
 * Mirrors the Protobuf schema (proto/telemetry.proto) and the Python
 * GCS API response shapes.  All positions are ENU metres relative to
 * the mission reference origin unless noted.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────

export type NavigationTier = 1 | 2 | 3;

export interface EnuPosition {
  east_m:  number;
  north_m: number;
  up_m:    number;
}

export interface DroneState {
  drone_id:              number;
  timestamp_s:           number;
  position:              EnuPosition;
  heading_rad:           number;
  jnr_db:                number;
  tier:                  NavigationTier;
  rms_position_error_m:  number;
  battery_voltage_v:     number;
  spoofing_suspected:    boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Danger Grid
// ─────────────────────────────────────────────────────────────────────────────

export type ThreatType =
  | "RF_JAMMING"
  | "GPS_SPOOFING"
  | "HUNTER_DRONE"
  | "OBSTACLE"
  | "COMMS_DEGRADED";

export interface DangerEntry {
  drone_id:    number;
  east_m:      number;
  north_m:     number;
  severity:    number;   // [0, 1]
  threat_type: ThreatType;
  ttl_s:       number;
  timestamp_s: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural Heatmap
// ─────────────────────────────────────────────────────────────────────────────

export interface HeatmapFrame {
  drone_id:             number;
  timestamp_s:          number;
  activations:          number[];   // flattened grid, row-major
  grid_width:           number;
  grid_height:          number;
  grid_origin_east_m:   number;
  grid_origin_north_m:  number;
  cell_size_m:          number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier display helpers
// ─────────────────────────────────────────────────────────────────────────────

export const TIER_LABELS: Record<NavigationTier, string> = {
  1: "GNSS",
  2: "Cellular",
  3: "SLAM",
};

export const TIER_COLOURS: Record<NavigationTier, string> = {
  1: "#22c55e",   // green  — GNSS active
  2: "#f59e0b",   // amber  — cellular primary
  3: "#ef4444",   // red    — SLAM only
};

export const THREAT_COLOURS: Record<ThreatType, string> = {
  RF_JAMMING:     "#ef4444",
  GPS_SPOOFING:   "#f97316",
  HUNTER_DRONE:   "#a855f7",
  OBSTACLE:       "#6b7280",
  COMMS_DEGRADED: "#3b82f6",
};
