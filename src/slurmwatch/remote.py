"""Sample another node of a multi-node job.

The dashboard collector only reads the node it runs on. To show a *different*
node (the node switcher), run slurmwatch's own ``--once --json`` on that node via
``srun --overlap`` and parse the result back into a snapshot — reusing all of the
real collection logic (cgroup v1/v2, NVML, per-process attribution) rather than
reimplementing it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

from .model import TelemetrySnapshot


def build_sample_command(job_id: str, node: str, python: str | None = None) -> list[str]:
    """The ``srun`` command that snapshots ``node`` once, as JSON on stdout.

    ``--jobid`` targets the running allocation and ``--overlap`` shares it (the
    sampler adds no new resources); ``python -m slurmwatch … --once --json`` runs
    the same install (a shared-filesystem path), so the remote node produces a
    snapshot in the identical format this process parses.
    """
    py = python or sys.executable
    return [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        "-w",
        node,
        "-n1",
        py,
        "-m",
        "slurmwatch",
        job_id,
        "--once",
        "--json",
    ]


def _child_env() -> dict[str, str]:
    # Strip the current step's SLURM_* variables so the nested srun isn't confused
    # by the step context the TUI is already running inside (the hop launched us
    # in a step); --jobid targets the allocation explicitly instead. Never re-hop
    # on the remote side (we're already on a job node), and never mock.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURMWATCH_NO_HOP"] = "1"
    env.pop("SLURMWATCH_MOCK", None)
    return env


async def sample_node(
    job_id: str, node: str, timeout: float = 20.0, python: str | None = None
) -> TelemetrySnapshot | None:
    """Snapshot ``node`` once via ``srun``; ``None`` on any failure/timeout.

    Never raises for a sampling failure (a busy scheduler, a transient srun
    error) — the caller keeps showing the last good snapshot. Propagates
    ``CancelledError`` so a node switch / shutdown can interrupt promptly.
    """
    cmd = build_sample_command(job_id, node, python)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_child_env(),
        )
    except (OSError, ValueError):
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        return None
    except asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise

    if proc.returncode != 0 or not stdout:
        return None

    # slurmwatch prints one JSON object; take the last non-empty line in case a
    # warning slipped onto stdout ahead of it.
    lines = [ln for ln in stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
    for line in reversed(lines):
        with contextlib.suppress(Exception):
            return TelemetrySnapshot.from_json(line)
    return None
