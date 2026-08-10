"""Celery async workers for positioning computation and threat classification.

Workers run on the GCS side, consuming tasks from Redis queue.
"""
from __future__ import annotations

import logging
from celery import Celery

log = logging.getLogger(__name__)

app = Celery(
    "cellhawk",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@app.task(name="gcs.classify_threat", bind=True, max_retries=2)
def classify_threat(self, telemetry_frame: dict) -> dict:
    """Classify threat level from a telemetry frame.

    Returns a dict with keys: threat_type, severity, confidence.
    """
    jnr_db: float = telemetry_frame.get("jnr_db", 0.0)
    tier: int = telemetry_frame.get("tier", 1)

    if jnr_db >= 19.0:
        threat_type, severity = "RF_JAMMING", min(1.0, (jnr_db - 19.0) / 11.0 + 0.8)
    elif jnr_db >= 6.0:
        threat_type, severity = "RF_JAMMING", min(0.8, (jnr_db - 6.0) / 13.0 + 0.3)
    else:
        threat_type, severity = "NONE", 0.0

    return {"threat_type": threat_type, "severity": severity, "tier": tier}


@app.task(name="gcs.compute_swarm_failover", bind=True, max_retries=1)
def compute_swarm_failover(self, fleet_state: dict) -> dict:
    """Elect a new lead node when the current lead is lost (§6.1).

    Selects the drone with the lowest JNR (best navigation quality)
    as the new lead.  Target failover time: 1.2 ms (§5.3).

    Args:
        fleet_state: {drone_id: {jnr_db, tier, battery_v, ...}}

    Returns:
        {new_lead_id, reason}
    """
    if not fleet_state:
        return {"new_lead_id": None, "reason": "empty_fleet"}

    new_lead = min(
        fleet_state.items(),
        key=lambda kv: (kv[1].get("jnr_db", 999.0), -kv[1].get("battery_v", 0.0)),
    )
    return {"new_lead_id": new_lead[0], "reason": "lowest_jnr"}
