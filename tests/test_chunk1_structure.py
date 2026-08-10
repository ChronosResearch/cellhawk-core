"""
Integration test suite for CellHawk.

Chunk 1: Structural tests — verify the package layout, config files,
and proto schema are all well-formed before any math is implemented.
"""

import pathlib
import yaml
import pytest


REPO_ROOT = pathlib.Path(__file__).parent.parent


class TestConfigFiles:
    """All YAML config files must be parseable and contain required keys."""

    def _load(self, name: str) -> dict:
        path = REPO_ROOT / "config" / name
        assert path.exists(), f"Config file missing: {path}"
        with path.open() as f:
            return yaml.safe_load(f)

    def test_ekf_params_loads(self) -> None:
        cfg = self._load("ekf_params.yaml")
        assert "tier_thresholds" in cfg
        assert "update_rate_hz" in cfg
        assert "initial_covariance_diagonal" in cfg
        assert len(cfg["initial_covariance_diagonal"]) == 9

    def test_ekf_tier_thresholds_are_ordered(self) -> None:
        cfg = self._load("ekf_params.yaml")
        t = cfg["tier_thresholds"]
        assert t["tier1_to_tier2_db"] < t["tier2_to_tier3_db"], (
            "Tier 1→2 threshold must be less than Tier 2→3 threshold"
        )

    def test_rssi_params_loads(self) -> None:
        cfg = self._load("rssi_params.yaml")
        assert "ldpl" in cfg
        assert "multilateration" in cfg
        assert "rician_fading" in cfg

    def test_rssi_min_towers_is_three(self) -> None:
        """Paper §3.2: minimum N=3 towers for a 2-D fix."""
        cfg = self._load("rssi_params.yaml")
        assert cfg["multilateration"]["min_towers"] == 3

    def test_cortex_params_loads(self) -> None:
        cfg = self._load("cortex_params.yaml")
        assert "network" in cfg
        assert "training" in cfg
        assert "curriculum" in cfg

    def test_cortex_state_dim_is_19(self) -> None:
        """Paper §4.1: 19-dimensional state vector."""
        cfg = self._load("cortex_params.yaml")
        assert cfg["network"]["state_dim"] == 19

    def test_cortex_curriculum_has_five_levels(self) -> None:
        """Paper §4.3: five difficulty levels."""
        cfg = self._load("cortex_params.yaml")
        assert len(cfg["curriculum"]["levels"]) == 5


class TestProtoSchema:
    """Protobuf schema file must exist and contain expected message names."""

    def test_proto_file_exists(self) -> None:
        proto = REPO_ROOT / "proto" / "telemetry.proto"
        assert proto.exists()

    def test_proto_contains_navigation_state(self) -> None:
        proto = (REPO_ROOT / "proto" / "telemetry.proto").read_text()
        assert "NavigationStateTelemetry" in proto

    def test_proto_contains_danger_grid(self) -> None:
        proto = (REPO_ROOT / "proto" / "telemetry.proto").read_text()
        assert "DangerGridBroadcast" in proto

    def test_proto_contains_heatmap(self) -> None:
        proto = (REPO_ROOT / "proto" / "telemetry.proto").read_text()
        assert "HeatmapFrame" in proto

    def test_proto_contains_replay_nonce(self) -> None:
        """Paper §8: 96-bit cryptographic nonces on every command dispatch."""
        proto = (REPO_ROOT / "proto" / "telemetry.proto").read_text()
        assert "nonce_96bit" in proto


class TestDirectoryStructure:
    """Repository layout must match the architectural specification."""

    @pytest.mark.parametrize("rel_path", [
        "crates/cellhawk-types/src/lib.rs",
        "crates/cellhawk-ekf/src/lib.rs",
        "crates/cellhawk-rssi/src/lib.rs",
        "python/cortex/__init__.py",
        "python/cortex/dqn.py",
        "python/cortex/trainer.py",
        "python/cortex/environment.py",
        "python/cortex/curriculum.py",
        "python/cortex/heatmap.py",
        "python/gcs/__init__.py",
        "python/gcs/main.py",
        "python/gcs/telemetry.py",
        "python/gcs/danger_grid.py",
        "python/gcs/workers.py",
        "python/slam/__init__.py",
        "python/slam/interface.py",
        "config/ekf_params.yaml",
        "config/rssi_params.yaml",
        "config/cortex_params.yaml",
        "proto/telemetry.proto",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "Cargo.toml",
        "pyproject.toml",
        "requirements.txt",
    ])
    def test_file_exists(self, rel_path: str) -> None:
        assert (REPO_ROOT / rel_path).exists(), f"Missing: {rel_path}"
