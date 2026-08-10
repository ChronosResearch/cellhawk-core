"""Danger Grid — Redis GEO-backed collective spatial memory (§6.1).

Any drone encountering a hazard broadcasts a DangerGridEntry.
Subsequent drones query the grid before path planning and re-route
without centralised command intervention.

Redis GEO commands used:
  GEOADD  — store entry with lon/lat
  GEORADIUS — query entries within radius
  EXPIRE  — TTL-based decay
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as aioredis


@dataclass
class DangerEntry:
    drone_id: int
    east_m: float
    north_m: float
    severity: float          # [0, 1]
    threat_type: str
    ttl_s: float
    timestamp_s: float


# Redis key prefix
_GEO_KEY = "cellhawk:danger_grid"
_META_PREFIX = "cellhawk:danger_meta:"


class DangerGrid:
    """Async Redis-backed Danger Grid.

    Args:
        redis_url: Redis connection URL (default localhost).
        ref_lat:   Reference latitude for ENU→WGS84 approximation.
        ref_lon:   Reference longitude.
    """

    # 1 degree latitude ≈ 111,320 m
    _M_PER_DEG_LAT = 111_320.0

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ref_lat: float = 40.7128,
        ref_lon: float = -74.0060,
    ) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        self._ref_lat = ref_lat
        self._ref_lon = ref_lon

    def _enu_to_wgs84(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Approximate ENU → WGS84 for Redis GEO storage."""
        m_per_deg_lon = self._M_PER_DEG_LAT * abs(
            __import__("math").cos(__import__("math").radians(self._ref_lat))
        )
        lat = self._ref_lat + north_m / self._M_PER_DEG_LAT
        lon = self._ref_lon + east_m / m_per_deg_lon
        return lat, lon

    async def broadcast(self, entry: DangerEntry) -> None:
        """Store a hazard entry in the grid with TTL."""
        lat, lon = self._enu_to_wgs84(entry.east_m, entry.north_m)
        member = f"{entry.drone_id}:{entry.timestamp_s}"

        pipe = self._redis.pipeline()
        pipe.geoadd(_GEO_KEY, [lon, lat, member])
        meta_key = f"{_META_PREFIX}{member}"
        pipe.set(meta_key, json.dumps(asdict(entry)), ex=int(entry.ttl_s))
        await pipe.execute()

    async def query_radius(
        self,
        east_m: float,
        north_m: float,
        radius_m: float,
    ) -> list[DangerEntry]:
        """Return all hazard entries within radius_m of the given ENU position."""
        lat, lon = self._enu_to_wgs84(east_m, north_m)
        members: list[str] = await self._redis.georadius(  # type: ignore[attr-defined]
            _GEO_KEY, lon, lat, radius_m, unit="m"
        )
        entries: list[DangerEntry] = []
        for member in members:
            raw = await self._redis.get(f"{_META_PREFIX}{member}")
            if raw:
                data: dict[str, Any] = json.loads(raw)
                entries.append(DangerEntry(**data))
        return entries

    async def close(self) -> None:
        await self._redis.aclose()
