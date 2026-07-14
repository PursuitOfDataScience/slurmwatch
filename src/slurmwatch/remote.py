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

# Bound step creation for the node-switch stream, same as the login-node hop: a
# stream that requests the GPU on a node whose GPU is held by the job's own step
# would otherwise retry step creation forever and the switch would look stuck.
_STREAM_CONNECT_TIMEOUT = 10
# A GPU a stream step can actually get yields a step in ~1s; cap the "can I get
# it?" probe so the "GPU held by the job's own step" case falls through fast.
_GPU_PROBE_SECONDS = 6


def build_stream_command(
    job_id: str, node: str, interval: float, python: str | None = None, gpu: bool = True
) -> list[str]:
    """The ``srun`` command that streams ``node``'s snapshots as JSONL on stdout.

    ``--jobid`` targets the running allocation and ``--overlap`` shares it (the
    stream adds no resources); ``-m slurmwatch … --log /dev/stdout`` runs the same
    install's headless logger, which flushes one JSON snapshot per ``interval``.

    ``--immediate`` bounds step creation so switching to a node whose GPU is held
    by the job's own step can't hang the stream. When ``gpu`` is False the step
    requests no GPU (``--gres=none``) so it still launches on such a node — the
    remote dashboard then shows live CPU/mem, GPU just unreadable, mirroring the
    login hop rather than leaving the switch stuck.

    ``--input none`` is critical: without it srun connects the *terminal's* stdin
    to the remote task and swallows every keystroke the user types at the live
    dashboard (so, e.g., pressing a node number to switch back never reaches the
    TUI while a remote node is on screen). The remote logger reads no input, so
    detaching stdin costs nothing.
    """
    py = python or sys.executable
    # --overlap shares CPUs, --mem=0 reserves no memory, --gres=none (when the
    # node's GPU is held) requests no GPU — together the stream step launches on
    # any live node no matter what the job's own steps hold.
    gres = [] if gpu else ["--gres=none"]
    return [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        f"--immediate={_STREAM_CONNECT_TIMEOUT}",
        "--mem=0",
        *gres,
        "--input=none",
        "-w",
        node,
        "-n1",
        py,
        "-m",
        "slurmwatch",
        job_id,
        "--log",
        "/dev/stdout",
        # Pin JSON: the parser reads JSONL, so a caller's SLURMWATCH_FORMAT=csv (or
        # any config default) must not turn the stream into CSV and break parsing.
        "--json",
        "--interval",
        f"{interval:g}",
    ]


async def _stream_can_get_gpu(job_id: str, node: str) -> bool:
    """Quietly test whether a stream step can obtain ``node``'s GPU(s).

    Runs a throwaway ``true`` step (output discarded). Success ⇒ stream with the
    GPU (live GPU util); failure (GPU held by the job's own step) ⇒ stream with
    ``--gres=none`` so the switch still shows CPU/mem instead of hanging.
    """
    probe = [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        f"--immediate={min(_STREAM_CONNECT_TIMEOUT, _GPU_PROBE_SECONDS)}",
        "--mem=0",
        "--input=none",
        "-w",
        node,
        "-n1",
        "true",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *probe,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=_child_env(),
        )
        return await proc.wait() == 0
    except (OSError, ValueError):
        return False


def _child_env() -> dict[str, str]:
    # Strip the current step's SLURM_* variables so the nested srun isn't confused
    # by the step context the TUI already runs inside (the hop launched us in a
    # step); --jobid targets the allocation explicitly instead. Never re-hop on
    # the remote side (we're already on a job node), and never mock.
    #
    # KEEP SLURM_CONF: on a cluster that exports it (a non-default slurm.conf path,
    # configless, or multi-cluster/federated), srun needs it to find the config and
    # reach slurmctld. Stripping it made the node-switcher stream's `srun` exit
    # immediately, so the target node never rendered and the switch looked stuck
    # (#51). The login-node hop in cli.py already special-cases SLURM_CONF for the
    # same reason; this keeps the two srun paths consistent.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"}
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
    # Quietly decide whether this node's GPU is reachable from a stream step; if
    # not (held by the job's own step) drop the GPU request so the stream still
    # launches (CPU/mem live) instead of hanging on step creation.
    gpu = await _stream_can_get_gpu(job_id, node)
    try:
        return await asyncio.create_subprocess_exec(
            *build_stream_command(job_id, node, interval, python, gpu=gpu),
            stdin=asyncio.subprocess.DEVNULL,  # never let srun read the terminal's
            stdout=asyncio.subprocess.PIPE,  # stdin — it would steal the user's keys
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
