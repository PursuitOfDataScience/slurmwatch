"""Stream another node of a multi-node job.

The dashboard collector only reads the node it runs on. To show a *different*
node (the node switcher), run slurmwatch's own headless logger on that node via
``srun --overlap`` and read its JSONL stream back — reusing all of the real
collection logic (cgroup v1/v2, NVML, per-process attribution) rather than
reimplementing it, and paying the ``srun`` launch cost once per viewed node
instead of on every refresh. Only the node currently on screen is streamed, so
this stays O(1) no matter how many nodes the job has.
"""

from __future__ import annotations

import asyncio
import os
import sys

from .model import TelemetrySnapshot


def build_stream_command(
    job_id: str, node: str, interval: float, python: str | None = None
) -> list[str]:
    """The ``srun`` command that streams ``node``'s snapshots as JSONL on stdout.

    ``--jobid`` targets the running allocation and ``--overlap`` shares it (the
    stream adds no resources); ``-m slurmwatch … --log /dev/stdout`` runs the same
    install's headless logger, which flushes one JSON snapshot per ``interval``.
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
        "--log",
        "/dev/stdout",
        "--interval",
        f"{interval:g}",
    ]


def _child_env() -> dict[str, str]:
    # Strip the current step's SLURM_* variables so the nested srun isn't confused
    # by the step context the TUI already runs inside (the hop launched us in a
    # step); --jobid targets the allocation explicitly instead. Never re-hop on
    # the remote side (we're already on a job node), and never mock.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURMWATCH_NO_HOP"] = "1"
    env.pop("SLURMWATCH_MOCK", None)
    return env


async def open_stream(
    job_id: str, node: str, interval: float = 1.0, python: str | None = None
) -> asyncio.subprocess.Process | None:
    """Launch the streaming ``srun`` for ``node``; ``None`` if it can't start.

    The caller reads ``proc.stdout`` line by line and parses each with
    :meth:`TelemetrySnapshot.from_json`, and must ``kill()``/``wait()`` the
    process when switching away or shutting down.
    """
    try:
        return await asyncio.create_subprocess_exec(
            *build_stream_command(job_id, node, interval, python),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_child_env(),
        )
    except (OSError, ValueError):
        return None


def parse_snapshot_line(line: bytes) -> TelemetrySnapshot | None:
    """A single JSONL line from the stream → snapshot, or ``None`` if unparseable."""
    text = line.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return TelemetrySnapshot.from_json(text)
    except Exception:
        return None
