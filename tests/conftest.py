from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# The ambient allocation, when the suite is run from inside a Slurm job (an
# sinteractive shell, or a batch job running the gate). `local_node_name()` prefers
# $SLURMD_NODENAME over socket.gethostname(), so the ~30 tests that patch
# gethostname to name "the local node" were silently overridden there: they
# exercised the OFF-node branch and passed for the wrong reason. Same code, green on
# a login node, red on a compute node. Stripping these makes the gate mean the same
# thing wherever it runs.
_AMBIENT_SLURM_VARS = ("SLURMD_NODENAME", "SLURM_NODELIST", "SLURM_JOB_ID", "SLURM_JOBID")


@pytest.fixture(autouse=True)
def _clean_slurmwatch_env() -> Generator[None, None, None]:
    """Isolate tests from the developer's ambient SLURMWATCH_* environment.

    An exported SLURMWATCH_MOCK, SLURMWATCH_FORMAT=csv, SLURMWATCH_POLL_INTERVAL,
    etc. would otherwise leak into config/CLI tests and make them pass or fail
    for the wrong reason (B-T9). Pop every SLURMWATCH_* for the duration and
    restore the exact prior values afterwards. The surrounding job's own SLURM_*
    identity goes with them (see _AMBIENT_SLURM_VARS).
    """
    # The "already reported this variable" set is process-global by design (one
    # line per stale variable, not one per frame), so it has to be cleared between
    # tests or the second test to look at a knob sees no warning.
    from slurmwatch import config as _config

    _config._warned_env_vars.clear()
    saved = {k: v for k, v in os.environ.items() if k.startswith("SLURMWATCH_")}
    saved.update({k: os.environ[k] for k in _AMBIENT_SLURM_VARS if k in os.environ})
    for key in saved:
        del os.environ[key]
    yield
    for key in [k for k in os.environ if k.startswith("SLURMWATCH_")] + list(_AMBIENT_SLURM_VARS):
        os.environ.pop(key, None)
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _clean_stream_transport_state() -> Generator[None, None, None]:
    """Forget which transport each node's stream used, between tests.

    `remote` remembers per node whether the last stream was the ssh rung or the
    Slurm-step rung, and which nodes' ssh turned out to be refused — process-global
    by design (the answer is a property of the site, not of one launch). Left
    standing, the first test to blacklist "cn9" silently changes the transport every
    later test on that node gets, which is the class of cross-test leak
    `_clean_slurmwatch_env` exists to prevent.
    """
    from slurmwatch import remote as _remote

    _remote.reset_stream_transport_state()
    yield
    _remote.reset_stream_transport_state()


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
