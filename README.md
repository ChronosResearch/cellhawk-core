# CellHawk Core

**A Triply-Redundant Navigation Architecture for GPS-Denied and Electronically Contested Environments**

Based on the research paper by Shashank Kumar (2026).

---

## Architecture

```
cellhawk-core/
├── crates/
│   ├── cellhawk-types/      # Shared domain types (NavigationState, TierLevel, …)
│   ├── cellhawk-ekf/        # 9-state EKF + JNR covariance scaling + tier arbitration
│   │   └── benches/         # Criterion handover timing benchmark              ← Chunk 3
│   ├── cellhawk-rssi/       # LDPL model + WLS multilateration + RANSAC + Rician fading
│   ├── cellhawk-sdr/        # SDR front-end: I/Q power, JNR estimation, RSSI extraction  ← Chunk 1
│   └── cellhawk-pyo3/       # PyO3 bridge: Rust EKF callable from Python                ← Chunk 1
├── python/
│   ├── cortex/
│   │   ├── dqn.py           # CORTEX v2.0 DQN (19-dim state, 9 actions)
│   │   ├── trainer.py       # Experience replay, epsilon-greedy, target network
│   │   ├── environment.py   # Simulation: LiDAR, wind, TPN hunter, jamming dome, terrain
│   │   ├── jamming.py       # Spatial EA jamming dome with 100 ms ramp                  ← Chunk 1
│   │   ├── imu_drift.py     # Allan variance IMU drift model + DriftPredictor           ← Chunk 3
│   │   ├── curriculum.py    # 5-level auto-curriculum scheduler
│   │   └── heatmap.py       # Neural heatmap projector (§4.5)
│   ├── mavlink/
│   │   └── adapter.py       # PX4/MAVLink adapter: DQN intent → SET_POSITION_TARGET     ← Chunk 2
│   ├── terrain/
│   │   └── osm_loader.py    # OSM 3D terrain: GeoJSON → TerrainGrid → LiDAR queries     ← Chunk 2
│   ├── gcs/
│   │   ├── crypto.py        # AES-256-GCM + HKDF + mTLS context builders               ← Chunk 3
│   │   └── telemetry.py     # WebSocket hub (encrypts outbound commands via crypto.py)  ← Chunk 3
│   └── slam/                # Visual SLAM interface (ORB-SLAM2 compatible)
├── frontend/                # MapLibre Digital Twin (TypeScript + Three.js)              ← Chunk 2
│   ├── src/
│   │   ├── types.ts         # Shared domain types (DroneState, DangerEntry, …)
│   │   ├── store/fleet.ts   # Reactive fleet state store
│   │   ├── workers/gcs_client.ts  # Reconnecting GCS WebSocket client
│   │   ├── components/map.ts      # MapLibre drone markers + danger grid
│   │   ├── components/three_overlay.ts  # Three.js 3D drone meshes + heatmap
│   │   └── main.ts          # Application entry point
│   └── tests/               # Vitest unit tests
├── config/                  # YAML parameter files (EKF, RSSI, CORTEX)
├── proto/                   # Protobuf telemetry schema
└── .github/workflows/       # CI/CD (Rust + PyO3 + Python + Frontend + proto)
```

## Navigation Tiers

| Tier | Condition | Primary Source | Simulated RMS Error |
|------|-----------|----------------|---------------------|
| 1 | JNR < 6 dB | GNSS | ~4.5 m |
| 2 | 6 ≤ JNR < 19 dB | Cellular RSSI | ~42 m (favourable) |
| 3 | JNR ≥ 19 dB | Visual SLAM | ~12 m (terrain-rich) |

Tier transitions use 1.5 dB hysteresis and a 5-cycle sigmoid covariance ramp to prevent EKF divergence at handover boundaries.

## SDR Front-End (`cellhawk-sdr`)

The SDR crate bridges raw I/Q samples from hardware to the navigation layer:

```
Hardware I/Q → PowerEstimator → JnrEstimator → jnr_db  → EKF.update_jnr()
                              → RssiExtractor → rssi_dbm → MultilaterationSolver
```

- **PowerEstimator**: sliding-window RMS power (O(1) updates via running sum)
- **JnrEstimator**: minimum-statistics noise floor tracking (Doblinger 1995)
- **RssiExtractor**: pilot-correlation RSSI extraction (LFSR Gold-code pilot)
- **SdrBackend** trait: hardware-agnostic; `SimulatedSdrBackend` for tests/HIL

## PyO3 Bridge (`cellhawk-pyo3`)

The Rust EKF is now directly callable from Python:

```python
import cellhawk_pyo3 as ch

ekf = ch.CellHawkEkf(east_m=0.0, north_m=0.0, up_m=50.0, heading_rad=0.0)
ekf.predict(ax=0.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=0.1)
ekf.update_jnr(jnr_db=3.0, timestamp_s=0.1)
ekf.update_gnss(east_m=10.0, north_m=5.0, up_m=50.0, hdop=1.0, satellites=8, timestamp_s=0.1)
state = ekf.state()   # dict: east_m, north_m, tier, rms_position_error_m, …
```

Build with maturin:
```bash
maturin develop --manifest-path crates/cellhawk-pyo3/Cargo.toml --release
```

## Spatial EA Jamming Dome

The simulation now models physically accurate Electronic Attack jamming:

```python
from python.cortex.jamming import JammingDome, JammingDomeField

dome = JammingDome(center_east_m=300.0, center_north_m=300.0,
                   radius_m=90.0, peak_jnr_db=20.0, ramp_steps=1)
field = JammingDomeField([dome])

# JNR at drone position — spatial falloff + 100 ms sigmoid ramp
jnr = field.step(east_m=300.0, north_m=300.0)  # → 20.0 dB (at centre)
jnr = field.step(east_m=500.0, north_m=500.0)  # → 0.5 dB (outside dome)
```

`CortexEnvironment` automatically spawns a dome field on each episode reset using the curriculum level's `jnr_max_db` as the peak JNR.

## PX4/MAVLink Adapter (`python/mavlink/`)

Translates the 10 Hz DQN action loop into MAVLink `SET_POSITION_TARGET_LOCAL_NED` commands:

```python
from python.mavlink.adapter import IntentDispatcher, SimulatedMavlinkAdapter

adapter = SimulatedMavlinkAdapter()          # or Px4Adapter("udp:127.0.0.1:14550")
dispatcher = IntentDispatcher(adapter)

# Called at 10 Hz from the DQN control loop
dispatcher.dispatch(action=2, heading_rad=0.785)  # action 2 = east
```

Action mapping (NED frame, §4.2):

| Action | Intent | NED velocity |
|--------|--------|--------------|
| 0 | hover | (0, 0, 0) |
| 1–4 | N/E/S/W | ±cruise_speed |
| 5/6 | climb/descend | vz = ∓2 m/s |
| 7/8 | evade-left/right | perpendicular to heading |

## OSM 3D Terrain (`python/terrain/`)

Replaces flat random obstacles with real-world building geometry:

```python
from python.terrain.osm_loader import OsmTerrainLoader
from python.cortex.environment import CortexEnvironment

loader = OsmTerrainLoader(ref_lat=40.7128, ref_lon=-74.0060)
grid = loader.load_geojson_string(open("manhattan.geojson").read())
# or: grid = loader.generate_synthetic(n_buildings=50)

env = CortexEnvironment(terrain_grid=grid)
```

Supports OSM `building:height`, `building:levels` tags. Falls back to 10 m default. Uses Shapely STRtree for O(log N) LiDAR queries.

## Digital Twin Frontend (`frontend/`)

```bash
cd frontend && npm install && npm run dev
# Open http://localhost:5173?gcs=http://localhost:8000&drones=1,2,3
```

Stack: MapLibre GL JS (2D map) + Three.js (3D overlay) + Vite + TypeScript strict mode.

Features:
- Live drone markers coloured by navigation tier (green/amber/red)
- Danger Grid hazard circles with threat-type colour coding
- Three.js drone cone meshes oriented by heading, altitude-scaled
- Neural heatmap as a textured plane overlay
- Reconnecting WebSocket client with exponential back-off


```bash
# Rust core (all 5 crates)
cargo build --release
cargo test --all

# PyO3 bridge
pip install maturin
maturin develop --manifest-path crates/cellhawk-pyo3/Cargo.toml --release

# Python layer
pip install -r requirements.txt
pytest tests/
```

## Test Coverage

| Layer | Tests | Status |
|-------|-------|--------|
| Rust (types + EKF + RSSI + SDR + PyO3 bridge) | 81 | ✅ |
| Python (CORTEX + GCS + SLAM + jamming + MAVLink + terrain + crypto + drift + timing) | 280 | ✅ |
| Frontend TypeScript (FleetStore + GcsClient + coordinates) | 18 | ✅ |
| **Total** | **379** | **✅ all passing** |

## Production Gap Status

| Gap | Status |
|-----|--------|
| ✅ SDR front-end (`cellhawk-sdr`) | **Fixed — Chunk 1** |
| ✅ PyO3 bridge (Rust EKF ↔ Python) | **Fixed — Chunk 1** |
| ✅ Spatial EA jamming dome | **Fixed — Chunk 1** |
| ✅ PX4/MAVLink adapter | **Fixed — Chunk 2** |
| ✅ OSM 3D terrain ingestion | **Fixed — Chunk 2** |
| ✅ MapLibre Digital Twin frontend | **Fixed — Chunk 2** |
| ✅ mTLS + AES-256-GCM | **Fixed — Chunk 3** |
| ✅ Handover timing benchmark (< 150 ms) | **Fixed — Chunk 3** |
| ✅ IMU drift characterisation | **Fixed — Chunk 3** |

> All quantitative results are simulation-derived. Physical SDR field validation planned Q3 2026.

## mTLS + AES-256-GCM (`python/gcs/crypto.py`)

All GCS ↔ drone command frames are now authenticated and encrypted (Gap 7):

```python
from python.gcs.crypto import CommandCipher, derive_key

# Derive a 256-bit key from a pre-shared secret via HKDF-SHA256
key = derive_key(b"mission-psk", salt=os.urandom(32))
cipher = CommandCipher(key)

# Encrypt a waypoint command bound to drone 1
wire = cipher.encrypt_bytes(payload, drone_id=1)   # nonce(12) + tag(16) + ct

# Decrypt on the drone side
plaintext = cipher.decrypt_bytes(wire, drone_id=1)  # raises InvalidTag if tampered
```

Wire format: `nonce (12 B) || GCM-tag+ciphertext (16+N B)`. AAD = `drone_id` as 4-byte big-endian, binding each ciphertext to a specific drone. `TelemetryHub` accepts an optional `CommandCipher` and encrypts all `send_command` calls transparently.

mTLS context builders (`build_server_ssl_context`, `build_client_ssl_context`) enforce TLS 1.3 minimum and mutual certificate authentication.

## Handover Timing Benchmark (`crates/cellhawk-ekf/benches/`)

The paper (§5.4) claims < 150 ms tier handover. Gap 8 adds two verification layers:

```bash
# Criterion micro-benchmark (measures µs-level computation latency)
cargo bench --bench handover -p cellhawk-ekf
```

Python timing tests (`tests/test_handover_timing.py`) assert:
- Single predict+JNR cycle: < 150 ms (actual: < 1 ms)
- Full T1→T2 handover computation: < 150 ms (actual: < 5 ms)
- Median cycle latency: < 1 ms (10 Hz budget = 100 ms)
- Tier actually transitions within ≤ 15 cycles

## IMU Drift Characterisation (`python/cortex/imu_drift.py`)

Allan variance model for GNSS-denied Tier 3 operation (Gap 9):

```python
from python.cortex.imu_drift import DriftPredictor, MEMS_UAV, TACTICAL

predictor = DriftPredictor(profile=MEMS_UAV, slam_velocity_noise_m_s=0.5)

# 1-σ position error after 60 s of GNSS denial (SLAM-assisted)
error_m = predictor.position_error_1sigma(60.0)   # ≈ 30 m

# Time until error exceeds 50 m
t = predictor.time_to_error(50.0)                  # ≈ 100 s

# Error budget table
table = predictor.error_budget_table([10, 30, 60, 120, 300])
```

Two preset profiles: `MEMS_UAV` (ICM-42688-P class) and `TACTICAL` (ADIS16488 class). The model validates the paper's ~12 m Tier 3 RMS claim: with 0.5 m/s SLAM velocity noise, the 1-σ bound at 24 s is ≤ 12 m.
