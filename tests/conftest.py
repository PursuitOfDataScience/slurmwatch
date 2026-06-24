from __future__ import annotations

import os
from collections.abc import Generator

import pytest

os.environ.setdefault("SLURMWATCH_MOCK", "1")


@pytest.fixture
def mock_slurm_env() -> Generator[None, None, None]:
    old = os.environ.copy()
    os.environ["SLURMWATCH_MOCK"] = "1"
    yield
    os.environ.clear()
    os.environ.update(old)
