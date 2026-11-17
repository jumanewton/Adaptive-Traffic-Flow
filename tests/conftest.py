"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure packages are importable
for pkg_dir in ["packages", "apps", "configs"]:
    pkg_path = PROJECT_ROOT / pkg_dir
    if pkg_path.exists():
        sys.path.insert(0, str(pkg_path))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Return data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def models_dir(project_root: Path) -> Path:
    """Return models directory."""
    return project_root / "models"


# Environment variable fixtures
@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables for each test."""
    # Clear traffic-related env vars
    for key in list(os.environ.keys()):
        if key.startswith(("VISION_", "TRACKING_", "TRAFFIC_", "CONTROL_", "SIM_", "STREAM_", "STORAGE_", "OBS_")):
            monkeypatch.delenv(key, raising=False)


# Markers
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_sumo: Tests requiring SUMO")
    config.addinivalue_line("markers", "requires_kafka: Tests requiring Kafka")