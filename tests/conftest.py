from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_slurmwatch_env() -> Generator[None, None, None]:
    """Isolate tests from the developer's ambient SLURMWATCH_* environment.

    An exported SLURMWATCH_MOCK, SLURMWATCH_FORMAT=csv, SLURMWATCH_POLL_INTERVAL,
    etc. would otherwise leak into config/CLI tests and make them pass or fail
    for the wrong reason (B-T9). Pop every SLURMWATCH_* for the duration and
    restore the exact prior values afterwards.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("SLURMWATCH_")}
    for key in saved:
        del os.environ[key]
    yield
    for key in [k for k in os.environ if k.startswith("SLURMWATCH_")]:
        del os.environ[key]
    os.environ.update(saved)


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
    """Create a fake job cgroup mirroring Slurm's real cgroup/v2 layout.

    The kernel's no-internal-process constraint means PIDs appear only in
    leaf cgroups (job_X/step_Y/user/task_Z), never at the job or step level.
    """
    job_cg = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope" / "job_12345"
    task_cg = job_cg / "step_0" / "user" / "task_0"
    task_cg.mkdir(parents=True)
    (job_cg / "cgroup.procs").write_text("")
    (job_cg / "step_0" / "cgroup.procs").write_text("")
    (task_cg / "cgroup.procs").write_text("1000\n1001\n")
    (job_cg / "cpu.stat").write_text("usage_usec 5000000\n")
    (job_cg / "memory.current").write_text(str(2 * 1024**3))
    (job_cg / "memory.max").write_text(str(8 * 1024**3))
    (job_cg / "memory.stat").write_text("inactive_file 104857600\nslab_reclaimable 52428800\n")
    (job_cg / "memory.peak").write_text(str(4 * 1024**3))
    return job_cg
