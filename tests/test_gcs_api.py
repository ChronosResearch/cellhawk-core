"""GCS FastAPI integration tests.

Uses httpx AsyncClient — no real Redis or Celery broker needed.
The danger_grid and telemetry hub are replaced with lightweight fakes.
"""
from __future__ import annotations

import json
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from python.gcs.main import app
from python.gcs.danger_grid import DangerEntry


# ── Fake danger grid so tests don't need Redis ────────────────────────────────

class _FakeDangerGrid:
    def __init__(self) -> None:
        self._entries: list[DangerEntry] = []

    async def broadcast(self, entry: DangerEntry) -> None:
        self._entries.append(entry)

    async def query_radius(self, east_m: float, north_m: float, radius_m: float) -> list[DangerEntry]:
        return [
            e for e in self._entries
            if ((e.east_m - east_m) ** 2 + (e.north_m - north_m) ** 2) ** 0.5 <= radius_m
        ]

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real Redis-backed grid with the fake for all tests."""
    import python.gcs.main as gcs_main
    gcs_main.grid = _FakeDangerGrid()  # type: ignore[assignment]


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "connected_drones" in body
    assert "timestamp" in body


# ─────────────────────────────────────────────────────────────────────────────
# Fleet
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fleet_empty_on_start(client: AsyncClient) -> None:
    r = await client.get("/fleet")
    assert r.status_code == 200
    assert r.json()["drone_ids"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Waypoint dispatch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_waypoint_dispatch_404_when_drone_not_connected(client: AsyncClient) -> None:
    r = await client.post(
        "/mission/42/waypoint",
        json={"east_m": 100.0, "north_m": 200.0, "altitude_m": 50.0, "speed_m_s": 5.0},
    )
    assert r.status_code == 404
    assert "42" in r.json()["detail"]


@pytest.mark.asyncio
async def test_waypoint_rejects_negative_altitude(client: AsyncClient) -> None:
    r = await client.post(
        "/mission/1/waypoint",
        json={"east_m": 0.0, "north_m": 0.0, "altitude_m": -10.0},
    )
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_waypoint_rejects_excessive_speed(client: AsyncClient) -> None:
    r = await client.post(
        "/mission/1/waypoint",
        json={"east_m": 0.0, "north_m": 0.0, "altitude_m": 50.0, "speed_m_s": 999.0},
    )
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Danger Grid
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_danger_grid_broadcast_and_query(client: AsyncClient) -> None:
    # Broadcast a hazard
    entry = {
        "drone_id": 1,
        "east_m": 100.0,
        "north_m": 100.0,
        "severity": 0.8,
        "threat_type": "RF_JAMMING",
        "ttl_s": 60.0,
        "timestamp_s": 0.0,
    }
    r = await client.post("/danger_grid/broadcast", json=entry)
    assert r.status_code == 200
    assert r.json()["status"] == "broadcast_ok"

    # Query within radius — should find it
    r = await client.post(
        "/danger_grid/query",
        json={"east_m": 100.0, "north_m": 100.0, "radius_m": 50.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["entries"][0]["threat_type"] == "RF_JAMMING"


@pytest.mark.asyncio
async def test_danger_grid_query_outside_radius_returns_empty(client: AsyncClient) -> None:
    entry = {
        "drone_id": 2,
        "east_m": 1000.0,
        "north_m": 1000.0,
        "severity": 0.5,
        "threat_type": "HUNTER_DRONE",
        "ttl_s": 30.0,
        "timestamp_s": 1.0,
    }
    await client.post("/danger_grid/broadcast", json=entry)

    r = await client.post(
        "/danger_grid/query",
        json={"east_m": 0.0, "north_m": 0.0, "radius_m": 100.0},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 0


@pytest.mark.asyncio
async def test_danger_grid_query_radius_validation(client: AsyncClient) -> None:
    r = await client.post(
        "/danger_grid/query",
        json={"east_m": 0.0, "north_m": 0.0, "radius_m": 0.0},  # below ge=1.0
    )
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text(client: AsyncClient) -> None:
    r = await client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text format always starts with a comment or metric name
    assert "cellhawk" in r.text or "#" in r.text
