"""OSM 3D terrain ingestion for CellHawk simulation (Gap 4).

## Overview

Replaces the flat random-obstacle spawner in `CortexEnvironment` with
real-world building geometry sourced from OpenStreetMap.

## Pipeline

    OSM GeoJSON / Overpass API
        ↓
    OsmTerrainLoader.load_geojson()   — parse building footprints
        ↓
    TerrainBuilding (footprint + height)
        ↓
    TerrainGrid.build()               — spatial index for fast queries
        ↓
    TerrainGrid.obstacles_in_radius() — LiDAR occlusion query
    TerrainGrid.height_at()           — ground elevation query

## Building height estimation

OSM buildings carry optional ``building:height`` or ``building:levels``
tags.  When absent, a default height of 10 m is assumed (2-storey).
``building:levels`` is converted via: height = levels × 3.5 m.

## Coordinate system

All internal coordinates are ENU (East-North-Up) metres relative to a
configurable reference origin (WGS-84 lat/lon).  The ENU conversion uses
the small-angle approximation valid within ±50 km of the reference point.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon, Point, box as shapely_box
from shapely.strtree import STRtree


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_M_PER_DEG_LAT = 111_320.0          # metres per degree latitude
_DEFAULT_HEIGHT_M = 10.0             # assumed height when OSM tag absent
_METRES_PER_LEVEL = 3.5              # floor-to-floor height (m)
_MIN_BUILDING_HEIGHT_M = 3.0         # discard degenerate buildings below this


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TerrainBuilding:
    """A single extruded building obstacle.

    Attributes:
        footprint_enu: Shapely Polygon in ENU metres.
        height_m:      Building height above ground (m).
        osm_id:        OSM way/relation ID (0 if synthetic).
        centroid_east_m, centroid_north_m: Footprint centroid.
        bounding_radius_m: Radius of the bounding circle (for fast rejection).
    """
    footprint_enu:      Polygon
    height_m:           float
    osm_id:             int = 0
    centroid_east_m:    float = 0.0
    centroid_north_m:   float = 0.0
    bounding_radius_m:  float = 0.0

    def __post_init__(self) -> None:
        c = self.footprint_enu.centroid
        self.centroid_east_m  = c.x
        self.centroid_north_m = c.y
        # Bounding radius = half-diagonal of the bounding box
        minx, miny, maxx, maxy = self.footprint_enu.bounds
        self.bounding_radius_m = math.hypot(maxx - minx, maxy - miny) / 2.0

    def contains_point(self, east_m: float, north_m: float) -> bool:
        """True if the 2-D footprint contains the given ENU point."""
        return bool(self.footprint_enu.contains(Point(east_m, north_m)))

    def distance_to(self, east_m: float, north_m: float) -> float:
        """Minimum 2-D distance from the footprint boundary to the point (m)."""
        return float(self.footprint_enu.exterior.distance(Point(east_m, north_m)))


# ─────────────────────────────────────────────────────────────────────────────
# TerrainGrid — spatial index
# ─────────────────────────────────────────────────────────────────────────────

class TerrainGrid:
    """Spatial index over a set of TerrainBuildings.

    Uses a Shapely STRtree (Sort-Tile-Recursive R-tree) for O(log N) queries.

    Args:
        buildings: List of TerrainBuilding objects.
    """

    def __init__(self, buildings: list[TerrainBuilding]) -> None:
        self._buildings = buildings
        # Build STRtree over footprint polygons
        self._tree = STRtree([b.footprint_enu for b in buildings])

    @property
    def building_count(self) -> int:
        return len(self._buildings)

    def obstacles_in_radius(
        self,
        east_m: float,
        north_m: float,
        radius_m: float,
    ) -> list[TerrainBuilding]:
        """Return all buildings whose footprint intersects a circle."""
        query_circle = Point(east_m, north_m).buffer(radius_m)
        indices = self._tree.query(query_circle, predicate="intersects")
        return [self._buildings[int(i)] for i in indices]

    def height_at(self, east_m: float, north_m: float) -> float:
        """Return the building height at a 2-D ENU point (0 if open ground)."""
        pt = Point(east_m, north_m)
        indices = self._tree.query(pt, predicate="intersects")
        if len(indices) == 0:
            return 0.0
        # Filter to buildings that actually contain the point (not just bbox)
        heights = [
            self._buildings[int(i)].height_m
            for i in indices
            if self._buildings[int(i)].footprint_enu.contains(pt)
               or self._buildings[int(i)].footprint_enu.covers(pt)
        ]
        return max(heights) if heights else 0.0

    def to_environment_obstacles(self) -> list[dict[str, float]]:
        """Export buildings as flat obstacle dicts for CortexEnvironment.

        Returns a list of dicts with keys: east_m, north_m, radius_m.
        The radius approximates the building as a circle with the same area.
        """
        result = []
        for b in self._buildings:
            area = b.footprint_enu.area
            radius = math.sqrt(area / math.pi)
            result.append({
                "east_m":   b.centroid_east_m,
                "north_m":  b.centroid_north_m,
                "radius_m": max(radius, 2.0),
            })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def wgs84_to_enu(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
) -> tuple[float, float]:
    """Convert WGS-84 (lat, lon) to ENU (east_m, north_m) relative to reference.

    Uses the small-angle flat-earth approximation, valid within ±50 km.
    """
    m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    north_m = (lat - ref_lat) * _M_PER_DEG_LAT
    east_m  = (lon - ref_lon) * m_per_deg_lon
    return east_m, north_m


# ─────────────────────────────────────────────────────────────────────────────
# OsmTerrainLoader
# ─────────────────────────────────────────────────────────────────────────────

class OsmTerrainLoader:
    """Loads OSM building data and converts it to a TerrainGrid.

    Supports two input formats:
    1. GeoJSON FeatureCollection (from Overpass API or QGIS export).
    2. Synthetic building generation for testing (no OSM data needed).

    Args:
        ref_lat: Reference latitude for ENU conversion.
        ref_lon: Reference longitude for ENU conversion.
    """

    def __init__(self, ref_lat: float = 40.7128, ref_lon: float = -74.0060) -> None:
        self._ref_lat = ref_lat
        self._ref_lon = ref_lon

    def load_geojson(self, geojson: dict[str, Any]) -> TerrainGrid:
        """Parse a GeoJSON FeatureCollection of OSM buildings.

        Each Feature must have geometry type ``Polygon`` or ``MultiPolygon``
        and may carry ``properties.height``, ``properties.building:height``,
        or ``properties.building:levels``.

        Args:
            geojson: Parsed GeoJSON dict (from ``json.loads()``).

        Returns:
            TerrainGrid ready for spatial queries.
        """
        buildings: list[TerrainBuilding] = []

        for feature in geojson.get("features", []):
            geom = feature.get("geometry", {})
            props = feature.get("properties", {}) or {}
            osm_id = int(props.get("id", 0) or props.get("osm_id", 0) or 0)

            height_m = self._extract_height(props)
            if height_m < _MIN_BUILDING_HEIGHT_M:
                continue

            polygons = self._extract_polygons(geom)
            for poly_coords in polygons:
                enu_coords = [
                    self._to_enu(lon, lat)
                    for lon, lat in poly_coords
                ]
                if len(enu_coords) < 3:
                    continue
                try:
                    footprint = Polygon(enu_coords)
                    if not footprint.is_valid or footprint.area < 1.0:
                        continue
                    buildings.append(TerrainBuilding(
                        footprint_enu=footprint,
                        height_m=height_m,
                        osm_id=osm_id,
                    ))
                except Exception:
                    continue

        return TerrainGrid(buildings)

    def load_geojson_string(self, geojson_str: str) -> TerrainGrid:
        """Parse a GeoJSON string."""
        return self.load_geojson(json.loads(geojson_str))

    def generate_synthetic(
        self,
        n_buildings: int,
        arena_east_m: float = 600.0,
        arena_north_m: float = 600.0,
        min_size_m: float = 8.0,
        max_size_m: float = 40.0,
        min_height_m: float = 5.0,
        max_height_m: float = 60.0,
        seed: int = 42,
    ) -> TerrainGrid:
        """Generate a synthetic urban terrain grid for simulation.

        Produces axis-aligned rectangular buildings at random positions.
        Useful for curriculum training without real OSM data.

        Args:
            n_buildings:   Number of buildings to generate.
            arena_east_m:  Arena width (m ENU).
            arena_north_m: Arena depth (m ENU).
            min/max_size_m:   Building footprint size range (m).
            min/max_height_m: Building height range (m).
            seed:          RNG seed for reproducibility.
        """
        import random
        rng = random.Random(seed)
        buildings: list[TerrainBuilding] = []
        margin = max_size_m

        for i in range(n_buildings):
            w = rng.uniform(min_size_m, max_size_m)
            d = rng.uniform(min_size_m, max_size_m)
            cx = rng.uniform(margin, arena_east_m - margin)
            cy = rng.uniform(margin, arena_north_m - margin)
            h = rng.uniform(min_height_m, max_height_m)

            footprint = shapely_box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2)
            buildings.append(TerrainBuilding(
                footprint_enu=footprint,
                height_m=h,
                osm_id=i + 1,
            ))

        return TerrainGrid(buildings)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_enu(self, lon: float, lat: float) -> tuple[float, float]:
        return wgs84_to_enu(lat, lon, self._ref_lat, self._ref_lon)

    @staticmethod
    def _extract_height(props: dict[str, Any]) -> float:
        """Extract building height from OSM properties."""
        # Direct height tag (may include 'm' suffix)
        for key in ("height", "building:height"):
            val = props.get(key)
            if val is not None:
                try:
                    return float(str(val).replace("m", "").strip())
                except ValueError:
                    pass
        # Levels tag
        levels = props.get("building:levels") or props.get("levels")
        if levels is not None:
            try:
                return float(levels) * _METRES_PER_LEVEL
            except ValueError:
                pass
        return _DEFAULT_HEIGHT_M

    @staticmethod
    def _extract_polygons(geom: dict[str, Any]) -> list[list[tuple[float, float]]]:
        """Extract coordinate rings from a GeoJSON geometry."""
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            # coords = [exterior_ring, *hole_rings]
            return [coords[0]] if coords else []
        elif gtype == "MultiPolygon":
            # coords = [[exterior, *holes], ...]
            return [poly[0] for poly in coords if poly]
        return []
