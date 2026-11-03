import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def offline() -> bool:
    return os.environ.get("AERODOPPLER_OFFLINE", "").lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(config, items):
    if offline():
        skip = pytest.mark.skip(reason="offline mode: network tests skipped")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)
