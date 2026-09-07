"""Stream another node of a multi-node job.

The dashboard collector only reads the node it runs on. To show a *different*
node (the node switcher), run slurmwatch's own headless logger on that node via
``srun --overlap`` and read its JSONL stream back — reusing all of the real
collection logic (cgroup v1/v2, NVML, per-process attribution) rather than
reimplementing it, and paying the ``srun`` launch cost (a GPU probe plus the
stream) once per viewed node instead of on every refresh. Only the node currently
on screen is streamed, so this stays O(1) no matter how many nodes the job has.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import sys

from .model import TelemetrySnapshot


def _kill_quietly(proc: asyncio.subprocess.Process | None) -> None:
    """SIGKILL a child if it's still running, tolerating an already-reaped one.

    asyncio's child watcher reaps the zombie once the process dies, so no ``await``
    is needed — which also makes this safe to call from a ``finally`` while the
    coroutine is being cancelled (an ``await`` there could re-raise immediately and
    skip the kill), the exact path that used to orphan the stream/probe ``srun``
    (N1).
    """
    if proc is not None and proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()


# Bound step creation for the node-switch stream, same as the login-node hop: a
# stream that requests the GPU on a node whose GPU is held by the job's own step
# would otherwise retry step creation forever and the switch would look stuck.
_STREAM_CONNECT_TIMEOUT = 10
# A GPU a stream step can actually get yields a step in ~1s; cap the "can I get
# it?" probe so the "GPU held by the job's own step" case falls through fast.
_GPU_PROBE_SECONDS = 6
# The stream's stdout is read one JSON line per frame, and asyncio's StreamReader
# defaults to a 64 KiB line limit — above which `readline()` raises ValueError and
# DISCARDS the line, so a node whose frames are too big never delivers one.
#
# One frame is O(devices^2), because GpuInterconnect.matrix is the device-by-device
# topology grid: measured 7.4 KB at 8 devices, 14.5 KB at 16, and 65,553 B at 56 —
# one byte over the default — which is exactly 8 GPUs x 7 MIG slices, the "many-slice
# MIG config" the CSV schema already sizes itself for. So the default made the
# BIGGEST nodes, the ones a right-sizing monitor is most wanted on, the ones the node
# switcher could not read. 1 MiB holds ~300 devices; it is a cap on one line, not a
# buffer that is allocated up front.
_STREAM_LINE_LIMIT = 1 << 20


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

    ``--input=none`` is critical: without it srun connects the *terminal's* stdin
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


def build_ssh_stream_command(
    job_id: str, node: str, interval: float, python: str | None = None
) -> list[str]:
    """Stream ``node``'s snapshots over ssh instead of an ``srun`` step.

    Used when a monitor step cannot be granted the job's GPUs — the normal case for
    multi-node training, where an inner ``srun`` holds every GPU and this Slurm
    cannot share GRES between steps. A step that requested no GPU is denied
    ``/dev/nvidiaN`` outright, so it can only ever report "GPU unreadable"; an
    adopted ssh session (pam_slurm_adopt + ``PrologFlags=Contain``) keeps the job's
    cgroups for CPU/memory accounting while the devices controller leaves it in
    ``/user.slice``, so NVML sees the job's GPUs and real utilization/VRAM/power
    stream through.

    One ssh per viewed node for the whole session — the stream is a single
    long-lived process — which matters because every login leaks threads into the
    job's ``.extern`` stepd that are never released.

    No ``-t``: a remote tty would inject control characters into the JSONL. ``ssh
    host cmd`` runs a NON-login shell, so PATH and SLURM_CONF are carried
    explicitly (same reason as the login-node ssh hop), via ``env VAR=val`` rather
    than the inline form csh/tcsh/fish cannot parse.
    """
    py = python or sys.executable
    env_prefix = ["env"]
    for var in ("PATH", "SLURM_CONF"):
        val = os.environ.get(var)
        if val:
            env_prefix.append(f"{var}={val}")
    # Never hop or ssh again from the far side: we are already on a job node.
    env_prefix += ["SLURMWATCH_NO_HOP=1", "SLURMWATCH_ON_NODE=1"]
    inner = [
        *env_prefix,
        py,
        "-m",
        "slurmwatch",
        job_id,
        "--log",
        "/dev/stdout",
        "--json",
        "--interval",
        f"{interval:g}",
    ]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        node,
        " ".join(shlex.quote(tok) for tok in inner),
    ]


# Which transport the last successful open_stream used for a node, and the nodes
# whose ssh transport turned out to be unusable. `shutil.which("ssh")` only proves
# the CLIENT exists — plenty of sites refuse login->compute ssh outright (measured
# on a Booth cluster: `youzhi@mcn57: Permission denied (publickey,gssapi-keyex,…)`,
# rc 255). There the ssh rung was preferred over a step that WOULD have worked, and
# its stderr said "permission denied", which `stream_error_is_permanent` reads as a
# refused Slurm step — so the node switcher gave up for good on a node it could
# still have reached, and blamed Slurm for ssh's answer.
_STREAM_TRANSPORT: dict[str, str] = {}
_SSH_STREAM_BLOCKED: set[str] = set()


def reset_stream_transport_state() -> None:
    """Forget which transport each node used (module state; tests and re-runs)."""
    _STREAM_TRANSPORT.clear()
    _SSH_STREAM_BLOCKED.clear()


def stream_transport(node: str) -> str:
    """``"ssh"`` / ``"step"`` for the last stream opened on ``node``, else ``""``."""
    return _STREAM_TRANSPORT.get(node, "")


def retry_other_stream_transport(node: str) -> bool:
    """The ssh stream on ``node`` died; True if the step transport is still untried.

    Called when a stream ends with an error that looks permanent. If that stream
    was the ssh one, ssh is retired FOR THAT NODE and the caller keeps going — the
    next launch takes the `--gres=none` step, which is what the site actually
    permits. Returns False for a step stream (both rungs have now failed, so giving
    up is correct) and False for an ssh stream on a node already blacklisted, which
    cannot happen but would otherwise loop.
    """
    if _STREAM_TRANSPORT.get(node) != "ssh" or node in _SSH_STREAM_BLOCKED:
        return False
    _SSH_STREAM_BLOCKED.add(node)
    _STREAM_TRANSPORT.pop(node, None)
    return True


def _ssh_stream_allowed(node: str = "") -> bool:
    """Whether the ssh stream transport may be used (env opt-out, ssh present).

    ``node`` also excludes one already proven unreachable by ssh this session, so a
    site that refuses login->compute ssh pays the failed attempt once per node
    instead of on every relaunch.
    """
    val = os.environ.get("SLURMWATCH_NO_SSH")
    if val is not None and val.strip().lower() not in ("", "0", "false", "no", "off"):
        return False
    if node and node in _SSH_STREAM_BLOCKED:
        return False
    return shutil.which("ssh") is not None


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
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *probe,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=_child_env(),
        )
        # --immediate bounds srun's resource wait but NOT the initial slurmctld RPC
        # (see the login hop's probe in cli.py), so cap the whole probe in Python
        # too — else a wedged controller hangs here until the caller's 25s wait_for
        # cancels us, leaving the probe srun orphaned.
        return await asyncio.wait_for(proc.wait(), _GPU_PROBE_SECONDS + 3) == 0
    except (OSError, ValueError, asyncio.TimeoutError):
        return False
    finally:
        # Reap on ANY exit — normal, our timeout, or a CancelledError from the
        # caller's wait_for firing / the user quitting mid-connect. This proc was
        # never handed back, so this is the only place it can be killed (N1).
        _kill_quietly(proc)


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
    # A step that can't get the GPU can only report "unreadable"; ssh can actually
    # read it. Prefer ssh in that case so switching to another node of a multi-node
    # GPU job shows real utilization instead of an explanation.
    use_ssh = not gpu and _ssh_stream_allowed(node)
    if use_ssh:
        cmd = build_ssh_stream_command(job_id, node, interval, python)
    else:
        cmd = build_stream_command(job_id, node, interval, python, gpu=gpu)
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,  # never let srun read the terminal's
            stdout=asyncio.subprocess.PIPE,  # stdin — it would steal the user's keys
            # KEPT, not discarded: this is the only place the reason a launch failed
            # exists. Throwing it away left the dashboard guessing "busy or
            # unreachable" at a permanent failure (see read_stream_error below).
            stderr=asyncio.subprocess.PIPE,
            # Not the 64 KiB default: see _STREAM_LINE_LIMIT.
            limit=_STREAM_LINE_LIMIT,
            env=_child_env(),
        )
        # Remember WHICH rung this is. The reason a stream died is on its stderr,
        # and ssh's wording ("permission denied") is indistinguishable from a
        # refused Slurm step — so the transport has to be recorded here rather than
        # guessed from the text later.
        _STREAM_TRANSPORT[node] = "ssh" if use_ssh else "step"
        return proc
    except (OSError, ValueError):
        _kill_quietly(proc)
        return None
    except asyncio.CancelledError:
        # The caller's 25s wait_for fired, or the user quit mid-connect.
        #
        # `proc` is ALWAYS None here, and this call is a safety net rather than the
        # thing that prevents the orphan. Measured: 5 of 5 cancellations reached
        # this handler with `proc is None`, having killed 0 of the 5 children
        # created. The reason is structural — the only `await` inside this `try` is
        # the `create_subprocess_exec` above, and cancellation is delivered only at
        # an await, so either it lands inside the exec (where the name is still
        # unbound) or the exec has returned and `return proc` runs with no await
        # left to interrupt it.
        #
        # What actually reaps a stream spawned-but-not-handed-back is asyncio's own
        # subprocess transport cleanup, which closes and waits the child when the
        # exec coroutine is cancelled. Measured on CPython 3.11.14: 42 trials
        # cancelling a real `create_subprocess_exec` across a sweep of delays
        # (13 cancelled, 29 won the race) left ZERO surviving children.
        # `test_no_orphan_when_the_stream_exec_is_cancelled` pins that, because it
        # is the guarantee this function actually depends on.
        #
        # Kept anyway: it costs nothing, tolerates an already-reaped child, and
        # becomes load-bearing the moment anything adds a second await inside this
        # try (N1). What it must not do is let a reader believe the orphan
        # protection lives here.
        _kill_quietly(proc)
        raise


def parse_snapshot_line(line: bytes) -> TelemetrySnapshot | None:
    """A single JSONL line from the stream → snapshot, or ``None`` if unparseable."""
    text = line.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return TelemetrySnapshot.from_json(text)
    except Exception:
        return None


# What a stream step says when it can NEVER launch here, however often we retry. Each
# was reported by srun/slurmstepd itself, on the stderr slurmwatch used to discard: an
# install the compute node cannot see (a node-local /tmp, an unshared venv, a container
# path), an allocation we may not join, or a job that has gone.
_PERMANENT_STREAM_ERRORS = (
    "execve()",
    "no such file or directory",
    "permission denied",
    "invalid job id",
    "invalid user",
)
# ...and deliberately NOT "unable to create step": that one clears the moment the job's
# own step releases the CPUs, so it keeps retrying — but it is still SUMMARISED, because
# showing the reason and giving up on it are separate decisions.

# The step launched fine and the REMOTE PYTHON refused the job — the version-skew half
# of the same misdiagnosis, and every token here was measured on this cluster
# (midway3-0200, `srun --overlap` into a live allocation):
#   * the node's python predates this source: a 14-line traceback ending
#     `SyntaxError: future feature annotations is not defined` (the node's system
#     python is 3.6.8; the package floor is 3.10, so a stream launched with a python
#     that resolves differently on the node hits this every time);
#   * a python that cannot import the package: `/usr/bin/python3: No module named
#     slurmwatch` (a venv that is not the one on PATH there, a different conda env);
#   * an OLDER slurmwatch on the node: `slurmwatch: error: unrecognized arguments:
#     --json` under a four-line argparse `usage:` dump.
# All three answer every retry identically, and all three were classified TRANSIENT —
# so the switcher relaunched `srun` on a node that can never serve it for the whole
# session, behind a banner that said it was "still retrying". `importerror` joins them
# because a half-upgraded install ("cannot import name X from slurmwatch.model") is the
# same fact about the far side. NOT the bare traceback header: a remote crash mid-stream
# prints one too, and that one may well clear on the next launch.
_PERMANENT_REMOTE_PYTHON_ERRORS = (
    "no module named",
    "importerror",
    "syntaxerror",
    "unrecognized arguments",
)


def _remote_python_failure_line(text: str) -> str:
    """The line that NAMES a broken remote install, or ``""``.

    Not the first line: a traceback's first line is ``Traceback (most recent call
    last):`` and its cause is ~13 lines down, so the banner's "quote the first line"
    fallback printed the one line of a stderr that says nothing (measured). srun's own
    epilogue (``srun: error: … task 0: Exited with exit code 1``) and the frame list
    (``File "…", line N``) are skipped for the same reason, and because srun's line can
    be interleaved BEFORE the remote's — the informative line is found by what it says,
    not by where it sits.
    """
    for raw in (text or "").splitlines():
        line = raw.strip()
        low = line.lower()
        if not line or low.startswith(("srun:", "slurmstepd:", 'file "')):
            continue
        if any(token in low for token in _PERMANENT_REMOTE_PYTHON_ERRORS):
            return line
    return ""


async def read_stream_error(proc: asyncio.subprocess.Process, limit: int = 2000) -> str:
    """Whatever the stream step wrote to stderr — bounded, and never a long block.

    The step's stderr is the ONLY place the reason lives. Discarding it left the
    dashboard able to say only "it may be busy or unreachable - still retrying", which
    is a guess, and it was the wrong guess in the case that motivated this: `/tmp` is
    node-local on some clusters, so an install there is invisible from the compute node
    and srun reports ``execve(): .../python: No such file or directory``. Permanent, and
    the reader was told to keep waiting.
    """
    if proc.stderr is None:
        return ""
    try:
        data = await asyncio.wait_for(proc.stderr.read(limit), timeout=0.5)
    except (TimeoutError, asyncio.TimeoutError, ValueError, OSError):
        return ""
    return data.decode("utf-8", "replace").strip()


# ssh's own answers when the CONNECTION never came up, all of them from `connect()`
# failing or the hostname not resolving (measured from this login node:
# `ssh: connect to host 10.255.255.1 port 22: Connection timed out`,
# `… port 1: Connection refused`, `ssh: Could not resolve hostname h: Name or service
# not known`). "permission denied" is the one already handled; a site that instead
# DROPS or rejects login->compute ssh, or does not resolve node names, produced none of
# the permanent tokens, so `retry_other_stream_transport` was never consulted and the
# switcher retried ssh — the rung that cannot work there — on every backoff, forever,
# never reaching the `--gres=none` step that does.
#
# Deliberately only the connect-time wordings. A stream that ssh'd in successfully and
# died an hour later must NOT retire ssh for the session: that rung is the only one that
# can read the job's GPUs, so a transient death has to stay a retry, not a demotion.
_PERMANENT_SSH_STREAM_ERRORS = (
    "connection refused",
    "connection timed out",
    "no route to host",
    "name or service not known",
    "host key verification failed",
)


def stream_error_is_permanent(text: str, transport: str = "") -> bool:
    """Whether retrying this stream failure could ever succeed.

    Three families, all terminal: srun/slurmstepd could not launch the step at all
    (``_PERMANENT_STREAM_ERRORS``), it launched and the node's python/slurmwatch refused
    the job (``_PERMANENT_REMOTE_PYTHON_ERRORS``), or the ssh rung never connected
    (``_PERMANENT_SSH_STREAM_ERRORS``, only when ``transport`` says ssh spoke — the
    module's standing lesson is that a rung is a recorded fact, never an inference from
    wording). The last two used to read as transient: a node running a different build,
    and a site that blocks ssh, were both retried for the whole session.

    "Permanent" is per ATTEMPT, not per node: the caller still gives the other rung a
    turn (see :func:`retry_other_stream_transport`) before it retires the node.
    """
    low = (text or "").lower()
    if any(token in low for token in _PERMANENT_STREAM_ERRORS):
        return True
    if transport == "ssh" and any(token in low for token in _PERMANENT_SSH_STREAM_ERRORS):
        return True
    # Judged by the same line the banner quotes, so "what the user is told" and "stop
    # retrying" can never disagree about which line of a traceback matters.
    return bool(_remote_python_failure_line(text))


def summarise_stream_error(
    text: str, node: str = "", ascii_mode: bool = False, transport: str = ""
) -> str:
    """One line naming the cause, for the banner that used to guess at it.

    ``transport`` is which rung produced ``text`` (see :func:`stream_transport`).
    Without it, ssh's "Permission denied (publickey,…)" was reported as *Slurm*
    refusing a step — a confident wrong diagnosis, and the one a reader would act
    on by mailing the wrong support queue.
    """
    dash = "-" if ascii_mode else "\u2014"
    low = (text or "").lower()
    if transport == "ssh" and "permission denied" in low:
        where = f" to {node}" if node else ""
        return f"ssh{where} is not permitted here {dash} falling back to a monitor step"
    if "execve()" in low or "no such file or directory" in low:
        where = f" on {node}" if node else ""
        return (
            f"slurmwatch could not start{where} {dash} this install is not on a "
            "filesystem the compute node can see (a node-local /tmp, an unshared venv)"
        )
    remote_python = _remote_python_failure_line(text)
    if remote_python:
        where = f" on {node}" if node else ""
        return (
            f"slurmwatch could not run{where} {dash} that node's python or slurmwatch "
            f"is not this one: {remote_python[:80]}"
        )
    if "permission denied" in low:
        return f"Slurm refused a step in this allocation {dash} permission denied"
    if "invalid job id" in low:
        return "Slurm no longer knows this job id"
    if "unable to create step" in low:
        return f"Slurm could not create a step here {dash} the allocation may be full"
    first = (text or "").splitlines()[0] if text else ""
    return first[:120]
