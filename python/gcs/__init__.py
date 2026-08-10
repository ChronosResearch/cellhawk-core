"""
cellhawk.gcs
============
Ground Control Station services.

Modules:
    main         — FastAPI application factory
    telemetry    — WebSocket real-time telemetry handler (Protobuf-encoded)
    danger_grid  — Redis GEO-backed collective spatial memory (§6.1)
    workers      — Celery async positioning and threat-classification workers
"""
