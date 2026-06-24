from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_mock_by_default() -> Generator[None, None, None]:
    """Ensure tests don't accidentally run in mock mode unless explicitly set."""
    old = os.environ.pop("SLURMWATCH_MOCK", None)
    yield
    if old is not None:
        os.environ["SLURMWATCH_MOCK"] = old
    else:
        os.environ.pop("SLURMWATCH_MOCK", None)


@pytest.fixture
def mock_slurm_env() -> Generator[None, None, None]:
    old = os.environ.copy()
    os.environ["SLURMWATCH_MOCK"] = "1"
    yield
    os.environ.clear()
    os.environ.update(old)


@pytest.fixture
def fake_cgroup_v2(tmp_path: Path) -> Path:
    """Create a fake cgroup v2 filesystem tree at tmp_path."""
    cg = tmp_path / "sys" / "fs" / "cgroup"
    cg.mkdir(parents=True)
    (cg / "cgroup.controllers").write_text("cpu memory")
    (cg / "cgroup.procs").write_text("")
    return cg


@pytest.fixture
def fake_cgroup_v2_job(fake_cgroup_v2: Path) -> Path:
    """Create a fake job cgroup under the fake v2 hierarchy."""
    job_cg = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope" / "job_12345"
    job_cg.mkdir(parents=True)
    (job_cg / "cgroup.procs").write_text("1000\n1001\n")
    (job_cg / "cpu.stat").write_text("usage_usec 5000000\n")
    (job_cg / "memory.current").write_text(str(2 * 1024**3))
    (job_cg / "memory.max").write_text(str(8 * 1024**3))
    (job_cg / "memory.stat").write_text("inactive_file 104857600\nslab_reclaimable 52428800\n")
    (job_cg / "memory.peak").write_text(str(4 * 1024**3))
    return job_cg
