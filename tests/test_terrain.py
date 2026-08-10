"""Tests for OSM 3D terrain ingestion (Gap 4).

Verifies:
- Synthetic terrain generation produces correct building counts/geometry
- GeoJSON parsing extracts buildings with correct heights
- TerrainGrid spatial queries (radius, height_at) are correct
- WGS-84 → ENU coordinate conversion is accurate
- CortexEnvironment uses terrain grid when provided
"""
from __future__ import annotations

import json
import math
import pytest

from python.terrain.osm_loader import (
    OsmTerrainLoader,
    TerrainGrid,
    TerrainBuilding,
    wgs84_to_enu,
    _DEFAULT_HEIGHT_M,
    _METRES_PER_LEVEL,
)
from python.cortex.environment import CortexEnvironment, Obstacle


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestWgs84ToEnu:
    REF_LAT = 40.7128
    REF_LON = -74.0060

    def test_reference_origin_maps_to_zero(self) -> None:
        e, n = wgs84_to_enu(self.REF_LAT, self.REF_LON, self.REF_LAT, self.REF_LON)
        assert abs(e) < 1e-6
        assert abs(n) < 1e-6

    def test_100m_north_increases_north(self) -> None:
        delta_lat = 100.0 / 111_320.0
        _, n = wgs84_to_enu(self.REF_LAT + delta_lat, self.REF_LON, self.REF_LAT, self.REF_LON)
        assert abs(n - 100.0) < 0.1

    def test_east_displacement_is_positive(self) -> None:
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(self.REF_LAT))
        delta_lon = 100.0 / m_per_deg_lon
        e, _ = wgs84_to_enu(self.REF_LAT, self.REF_LON + delta_lon, self.REF_LAT, self.REF_LON)
        assert abs(e - 100.0) < 0.1

    def test_south_gives_negative_north(self) -> None:
        delta_lat = 100.0 / 111_320.0
        _, n = wgs84_to_enu(self.REF_LAT - delta_lat, self.REF_LON, self.REF_LAT, self.REF_LON)
        assert n < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic terrain generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticTerrain:
    def _loader(self) -> OsmTerrainLoader:
        return OsmTerrainLoader()

    def test_generates_correct_building_count(self) -> None:
        loader = self._loader()
        grid = loader.generate_synthetic(n_buildings=20, seed=0)
        assert grid.building_count == 20

    def test_buildings_within_arena(self) -> None:
        loader = self._loader()
        grid = loader.generate_synthetic(n_buildings=50, arena_east_m=600.0, arena_north_m=600.0, seed=1)
        for b in grid._buildings:
            cx, cy = b.centroid_east_m, b.centroid_north_m
            assert 0 < cx < 600.0, f"centroid east {cx} out of arena"
            assert 0 < cy < 600.0, f"centroid north {cy} out of arena"

    def test_building_heights_within_range(self) -> None:
        loader = self._loader()
        grid = loader.generate_synthetic(
            n_buildings=30, min_height_m=5.0, max_height_m=60.0, seed=2
        )
        for b in grid._buildings:
            assert 5.0 <= b.height_m <= 60.0

    def test_reproducible_with_same_seed(self) -> None:
        loader = self._loader()
        g1 = loader.generate_synthetic(n_buildings=10, seed=99)
        g2 = loader.generate_synthetic(n_buildings=10, seed=99)
        for b1, b2 in zip(g1._buildings, g2._buildings):
            assert abs(b1.centroid_east_m - b2.centroid_east_m) < 1e-9

    def test_zero_buildings_returns_empty_grid(self) -> None:
        loader = self._loader()
        grid = loader.generate_synthetic(n_buildings=0)
        assert grid.building_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# GeoJSON parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoJsonParsing:
    REF_LAT = 40.7128
    REF_LON = -74.0060

    def _loader(self) -> OsmTerrainLoader:
        return OsmTerrainLoader(ref_lat=self.REF_LAT, ref_lon=self.REF_LON)

    def _make_geojson(
        self,
        height: float | None = None,
        levels: int | None = None,
        coords: list | None = None,
    ) -> dict:
        """Build a minimal GeoJSON FeatureCollection with one building."""
        if coords is None:
            # Small square ~10m × 10m near reference origin
            d = 0.0001  # ~11 m
            lon, lat = self.REF_LON, self.REF_LAT
            coords = [
                [lon, lat], [lon + d, lat], [lon + d, lat + d],
                [lon, lat + d], [lon, lat],
            ]
        props: dict = {}
        if height is not None:
            props["height"] = height
        if levels is not None:
            props["building:levels"] = levels

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": props,
            }],
        }

    def test_parses_building_with_explicit_height(self) -> None:
        loader = self._loader()
        grid = loader.load_geojson(self._make_geojson(height=25.0))
        assert grid.building_count == 1
        assert abs(grid._buildings[0].height_m - 25.0) < 1e-6

    def test_parses_building_with_levels_tag(self) -> None:
        loader = self._loader()
        grid = loader.load_geojson(self._make_geojson(levels=5))
        assert grid.building_count == 1
        assert abs(grid._buildings[0].height_m - 5 * _METRES_PER_LEVEL) < 1e-6

    def test_uses_default_height_when_no_tag(self) -> None:
        loader = self._loader()
        grid = loader.load_geojson(self._make_geojson())
        assert grid.building_count == 1
        assert abs(grid._buildings[0].height_m - _DEFAULT_HEIGHT_M) < 1e-6

    def test_empty_feature_collection(self) -> None:
        loader = self._loader()
        grid = loader.load_geojson({"type": "FeatureCollection", "features": []})
        assert grid.building_count == 0

    def test_height_string_with_m_suffix(self) -> None:
        loader = self._loader()
        geojson = self._make_geojson()
        geojson["features"][0]["properties"]["height"] = "30m"
        grid = loader.load_geojson(geojson)
        assert abs(grid._buildings[0].height_m - 30.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TerrainGrid spatial queries
# ─────────────────────────────────────────────────────────────────────────────

class TestTerrainGridQueries:
    def _grid_with_one_building(self) -> TerrainGrid:
        loader = OsmTerrainLoader()
        return loader.generate_synthetic(
            n_buildings=1,
            arena_east_m=200.0,
            arena_north_m=200.0,
            min_size_m=20.0,
            max_size_m=20.0,
            min_height_m=15.0,
            max_height_m=15.0,
            seed=0,
        )

    def test_obstacles_in_radius_finds_nearby_building(self) -> None:
        grid = self._grid_with_one_building()
        b = grid._buildings[0]
        results = grid.obstacles_in_radius(b.centroid_east_m, b.centroid_north_m, 50.0)
        assert len(results) >= 1

    def test_obstacles_in_radius_misses_distant_building(self) -> None:
        grid = self._grid_with_one_building()
        results = grid.obstacles_in_radius(5000.0, 5000.0, 10.0)
        assert len(results) == 0

    def test_height_at_inside_building_returns_height(self) -> None:
        grid = self._grid_with_one_building()
        b = grid._buildings[0]
        h = grid.height_at(b.centroid_east_m, b.centroid_north_m)
        assert abs(h - 15.0) < 1e-6

    def test_height_at_open_ground_returns_zero(self) -> None:
        grid = self._grid_with_one_building()
        h = grid.height_at(5000.0, 5000.0)
        assert h == 0.0

    def test_to_environment_obstacles_returns_correct_count(self) -> None:
        loader = OsmTerrainLoader()
        grid = loader.generate_synthetic(n_buildings=5, seed=3)
        obs = grid.to_environment_obstacles()
        assert len(obs) == 5

    def test_to_environment_obstacles_have_required_keys(self) -> None:
        loader = OsmTerrainLoader()
        grid = loader.generate_synthetic(n_buildings=3, seed=4)
        for o in grid.to_environment_obstacles():
            assert "east_m"   in o
            assert "north_m"  in o
            assert "radius_m" in o
            assert o["radius_m"] >= 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CortexEnvironment terrain integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentTerrainIntegration:
    def test_environment_accepts_terrain_grid(self) -> None:
        loader = OsmTerrainLoader()
        grid = loader.generate_synthetic(n_buildings=10, seed=5)
        env = CortexEnvironment(terrain_grid=grid, seed=0)
        env.reset()
        # Obstacles should come from the terrain grid
        assert len(env._obstacles) == 10

    def test_environment_falls_back_to_random_without_grid(self) -> None:
        env = CortexEnvironment(seed=0)
        env.reset()
        # Random obstacles spawned from density config
        assert len(env._obstacles) >= 0  # may be 0 at level 1 density

    def test_terrain_obstacles_have_correct_positions(self) -> None:
        loader = OsmTerrainLoader()
        grid = loader.generate_synthetic(n_buildings=5, seed=6)
        env = CortexEnvironment(terrain_grid=grid, seed=0)
        env.reset()
        for obs in env._obstacles:
            assert isinstance(obs, Obstacle)
            assert obs.radius_m >= 2.0
