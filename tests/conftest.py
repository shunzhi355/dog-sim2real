from pathlib import Path

import pytest

from dog3588.config import load_config, Calibration
from dog3588.policy import load_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "config/robot.yaml")


@pytest.fixture
def manifest(cfg):
    return load_manifest(cfg["policy_manifest"])


@pytest.fixture
def calibrations(cfg):
    # Synthetic calibration for tests ONLY. Never written into runtime hardware config.
    return [Calibration(j["name"], j["bus"], j["id"], 500, 1, 200., 0., -1., 1.)
            for j in cfg["joints"]]

