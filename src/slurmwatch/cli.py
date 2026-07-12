# ruff: noqa: T201
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any, NoReturn

from ._version import VERSION
from .collector import TelemetryCollector
from .config import SlurmwatchConfig, _parse_bool
from .exceptions import (
    CgroupAccessError,
    CgroupNotFoundError,
    JobNotFoundError,
    JobNotPendingError,
    JobNotRunningError,
    SlurmCommandError,
)
from .model import JobContext, TelemetrySnapshot
from .pending import (
    PendingJob,
    explain_reason,
    partition_fits_now,
    resolve_cluster_partitions,
    resolve_pending_job,
    resolve_queue_counts,
)
from .slurm import resolve_current_jobs, resolve_job_context

logger = logging.getLogger("slurmwatch")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)

HOSTNAME = socket.gethostname().split(".")[0]


class _BufferingLogHandler(logging.Handler):
    """Collect log records instead of writing them to the terminal.

    Used while the TUI owns the alternate screen so a collector warning or
    traceback doesn't splatter across the dashboard (B-C3). Bounded so a long,
    noisy session can't grow without limit; the newest records are kept.
    """

    _MAX_RECORDS = 200

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if len(self.records) > self._MAX_RECORDS:
            del self.records[0]


@contextlib.contextmanager
def _console_logging_suspended() -> Iterator[None]:
    """Divert slurmwatch logging away from the terminal for the duration.

    The module attaches a stderr StreamHandler at import; while the live TUI
    holds the screen, a propagated collector warning/traceback would corrupt it
    (B-C3). Buffer records during the block and replay them to stderr once the
    TUI has released the screen, so the user still sees them — just afterwards.
    """
    buffer = _BufferingLogHandler()
    logger.removeHandler(_handler)
    logger.addHandler(buffer)
    try:
        yield
    finally:
        logger.removeHandler(buffer)
        logger.addHandler(_handler)
        for record in buffer.records:
            _handler.handle(record)


def _bounded_exit(code: int) -> NoReturn:
    """Terminate immediately without joining stuck executor threads.

    A collection that timed out is still running on an executor thread with no
    internal timeout (e.g. a cgroup read on a wedged NFS mount). ``sys.exit``
    would unwind into ``asyncio.run``'s finalizer, which *joins* that thread and
    hangs the process well past the "timeout" (B-C4). ``os._exit`` skips the
    join; flush first so buffered output isn't lost.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _env_disables_hop() -> bool:
    """Whether SLURMWATCH_NO_HOP is set to a value that disables the srun hop.

    A plain truthiness test treated ``SLURMWATCH_NO_HOP=0``/``false`` as "on"
    and wrongly disabled the hop (B-P2); parse it as a boolean instead. An
    unrecognized value is treated as "set" (disable), matching the flag's
    belt-and-suspenders intent.
    """
    val = os.environ.get("SLURMWATCH_NO_HOP")
    if val is None:
        return False
    try:
        return _parse_bool(val)
    except ValueError:
        return True


def _mouse_enabled() -> bool:
    """Whether to let the TUI capture the mouse.

    Off by default so the terminal's own text selection and copy/paste keep
    working — slurmwatch is fully keyboard-driven (c/m/g/v to focus panels,
    arrows/PgUp/PgDn to scroll, q to quit). Set SLURMWATCH_MOUSE=1 to re-enable
    mouse support (e.g. wheel scrolling), at the cost of drag-to-select.
    """
    return os.environ.get("SLURMWATCH_MOUSE", "") == "1"


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from exc
    # Reject inf/nan (which slip past the <= 0 check: `inf <= 0` and `nan <= 0`
    # are both False) so asyncio.sleep() can't be handed a non-finite interval.
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"interval must be a finite number, got {value}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"interval must be positive, got {value}")
    return parsed


def _env_output_format() -> str:
    """Read SLURMWATCH_FORMAT, normalized to 'json'/'csv' (case-insensitive).

    Returns "" when unset/empty. An unrecognized value raises ValueError rather
    than being silently treated as CSV — e.g. SLURMWATCH_FORMAT=JSON used to emit
    CSV because the comparison was exact-lowercase (C4).
    """
    raw = os.environ.get("SLURMWATCH_FORMAT")
    if raw is None or raw.strip() == "":
        return ""
    fmt = raw.strip().casefold()
    if fmt not in ("json", "csv"):
        raise ValueError(f"Invalid value for SLURMWATCH_FORMAT: {raw!r} (expected 'json' or 'csv')")
    return fmt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmwatch",
        description="Live, process-isolated hardware telemetry for active Slurm jobs.",
    )
    parser.add_argument(
        "job_id",
        nargs="?",
        type=str,
        default=None,
        help="Slurm job ID to monitor (supports array tasks like 12345_3). "
        "Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        type=str,
        default=None,
        help="Run headless and write telemetry to FILE (.jsonl or .csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to the --log file instead of overwriting it",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Take a single snapshot and print to stdout, then exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Shorthand for --format json",
    )
    parser.add_argument(
        "--interval",
        metavar="SECONDS",
        type=_positive_float,
        default=None,
        help="Polling interval in seconds (default: 0.5 for TUI, 1.0 for headless)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose diagnostic logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run with simulated demo data (sets SLURMWATCH_MOCK=1)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        default=False,
        help="Use ASCII-only characters (no Unicode block glyphs)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default=None,
        help="Output format for --once and --log "
        "(default for --log: inferred from the file extension, otherwise JSON; "
        "default for --once: CSV)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("slurmwatch").setLevel(logging.DEBUG)

    try:
        config = SlurmwatchConfig.from_env()
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)

    if args.demo:
        os.environ["SLURMWATCH_MOCK"] = "1"
        config.poll_interval = 0.25
        config.headless_interval = 0.25

    if args.ascii:
        config.ascii_mode = True

    if args.interval is not None:
        config.poll_interval = args.interval
        config.headless_interval = args.interval

    # Re-apply the interval floor: a CLI --interval bypasses the clamp that
    # from_env enforces, and --interval 0.0001 would busy-loop the node (B-P1).
    config.clamp()

    job_id = args.job_id
    log_path: str | None = args.log
    headless = log_path is not None
    once = args.once

    if once and headless:
        logger.error("--once and --log are mutually exclusive")
        sys.exit(1)

    try:
        env_fmt = _env_output_format()
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)
    fmt = args.format or ("json" if args.json else "") or env_fmt

    if job_id is None:
        if os.environ.get("SLURMWATCH_MOCK") == "1":
            job_id = "12345"
        else:
            job_id = _auto_discover_job_id(config, interactive=not (once or headless))
            if job_id is None:
                return

    if headless:
        assert log_path is not None
        _run_headless(job_id, config, log_path, fmt, append=args.append)
    elif once:
        _run_once(job_id, config, fmt)
    else:
        _run_interactive(job_id, config, args)


def _auto_discover_job_id(config: SlurmwatchConfig, interactive: bool = True) -> str | None:
    username = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    logger.info("Auto-discovering running jobs for user %s...", username)

    try:
        jobs = resolve_current_jobs(username)
    except Exception as exc:
        logger.error("Failed to query Slurm jobs: %s", exc)
        sys.exit(1)

    if not jobs:
        user_message = (
            f"No running Slurm jobs found for user '{username}'. "
            "Launch a job first or provide a job_id argument."
        )
        print(user_message, file=sys.stderr)
        sys.exit(1)

    if len(jobs) == 1:
        jid: str = str(jobs[0]["job_id"])
        logger.info("Attaching to running job %s", jid)
        return jid

    if not interactive:
        listing = ", ".join(str(j["job_id"]) for j in jobs)
        logger.error(
            "Multiple running jobs found (%s); pass the job_id to monitor.",
            listing,
        )
        sys.exit(1)

    from .tui import SlurmwatchApp

    app = SlurmwatchApp(jobs=jobs, config=config)
    with _console_logging_suspended():
        app.run(mouse=_mouse_enabled())
    if app.return_code:
        sys.exit(app.return_code)
    return None


def _die_on_resolve_error(exc: Exception, job_id: str) -> NoReturn:
    """Map a resolve failure to a clear message + exit(1)."""
    if isinstance(exc, JobNotFoundError):
        logger.error("Job %s does not exist in the Slurm database.", job_id)
    elif isinstance(exc, JobNotRunningError):
        logger.error(str(exc))
    elif isinstance(exc, (CgroupNotFoundError, CgroupAccessError)):
        logger.error(
            "Job %s: %s\n\nTo resolve this:\n"
            "  1. Make sure you are on the compute node running the job\n"
            "  2. Verify the job is in RUNNING state\n"
            "  3. Try: srun --jobid %s --overlap slurmwatch",
            job_id,
            exc,
            job_id,
        )
    elif isinstance(exc, SlurmCommandError):
        logger.error("Slurm command failed: %s", exc)
    else:
        logger.error("Failed to resolve job context: %s", exc)
    sys.exit(1)


def _resolve_or_die(job_id: str) -> JobContext:
    try:
        return resolve_job_context(job_id)
    except Exception as exc:
        _die_on_resolve_error(exc, job_id)


def _resolve_running_or_pending(job_id: str) -> tuple[JobContext | None, PendingJob | None]:
    """Resolve ``job_id`` to a running JobContext, or a PendingJob if it's queued.

    A running job returns ``(ctx, None)``. A PENDING job — which
    ``resolve_job_context`` rejects with ``JobNotRunningError`` — instead returns
    ``(None, pending)`` so the caller can show the why/when/where pending view
    (#60) rather than a dead-end error. Any genuinely non-runnable state
    (completed/failed/not-found) still exits with the usual clear message.
    """
    # Demo hook: in mock mode `resolve_job_context` always returns a RUNNING job,
    # so a sentinel id (`slurmwatch --demo pending`) is the only way to preview the
    # pending view offline.
    if os.environ.get("SLURMWATCH_MOCK") == "1" and job_id.lower() in ("pending", "queued"):
        return None, resolve_pending_job(job_id)
    try:
        return resolve_job_context(job_id), None
    except JobNotRunningError as exc:
        try:
            return None, resolve_pending_job(job_id)
        except (JobNotPendingError, JobNotFoundError, SlurmCommandError):
            _die_on_resolve_error(exc, job_id)
        except Exception:
            _die_on_resolve_error(exc, job_id)
    except Exception as exc:
        _die_on_resolve_error(exc, job_id)


def _run_once(job_id: str, config: SlurmwatchConfig, fmt: str = "") -> None:
    job_ctx, pending = _resolve_running_or_pending(job_id)
    if pending is not None:
        # A queued job has no snapshot to emit. --once is machine-oriented (its
        # stdout is meant to be parseable JSON/CSV), so keep stdout clean: print
        # the human why/when/where report to STDERR and exit non-zero, signalling
        # "no snapshot available" rather than polluting the stream with prose that
        # a downstream jq/CSV reader would choke on (#60 review).
        _print_pending_summary(pending, stream=sys.stderr)
        sys.exit(1)
    assert job_ctx is not None
    collector = TelemetryCollector(job_ctx, config)
    asyncio.run(_once_loop(collector, json_output=fmt == "json", csv_dialect=config.csv_dialect))


async def _once_loop(
    collector: TelemetryCollector, json_output: bool, csv_dialect: str = "excel"
) -> None:
    await collector.start()
    try:
        snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=10.0)
        if json_output:
            print(snapshot.to_json())
        else:
            # Size the CSV GPU columns to this job's actual device count so a
            # >8-GPU node (or a many-slice MIG config) isn't silently clipped (#38).
            max_gpus = max(len(snapshot.gpus), collector.job_ctx.gpu_count_requested)
            writer = csv.writer(sys.stdout, dialect=csv_dialect)
            writer.writerow(TelemetrySnapshot.csv_header(max_gpus))
            writer.writerow(snapshot.to_csv_row(max_gpus))
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for first snapshot")
        # The collection that timed out is still on an executor thread; exit
        # hard so a wedged read can't hang us past the timeout (B-C4).
        _bounded_exit(1)
    finally:
        await collector.stop()


def _fmt_gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _print_remote_summary(job_ctx: JobContext, snap: TelemetrySnapshot) -> None:
    mem = snap.memory
    cpu = snap.cpu
    node = job_ctx.nodelist or "?"
    state = job_ctx.job_state or ""
    print(f"Job {job_ctx.job_id}  {job_ctx.partition}  {state}  on {node}")
    if mem.current_bytes > 0 or cpu.usage_ns > 0:
        if mem.limit_bytes > 0:
            print(
                f"  Memory   peak {_fmt_gib(mem.current_bytes)} / "
                f"{_fmt_gib(mem.limit_bytes)} ({mem.usage_percent:.0f}%)"
            )
        else:
            print(f"  Memory   peak {_fmt_gib(mem.current_bytes)}")
        print(
            f"  CPU      {_fmt_hms(cpu.usage_ns / 1e9)} CPU-time  "
            f"~{cpu.effective_cores:.1f} of {cpu.cores_allocated} cores (avg, running steps)"
        )
    else:
        print("  usage not yet sampled by Slurm (samples ~every 30s) — try again shortly")
    if job_ctx.gpu_count_requested > 0:
        print(
            f"  GPU      {job_ctx.gpu_count_requested} allocated — "
            "run slurmwatch on the compute node for live GPU utilization"
        )
    print("  source: sstat (remote; run on the node for working-set & live GPU util)")


def _run_remote_summary(job_ctx: JobContext, config: SlurmwatchConfig) -> None:
    collector = TelemetryCollector(job_ctx, config)

    async def _run() -> TelemetrySnapshot:
        await collector.start()
        try:
            return await asyncio.wait_for(collector.next_snapshot(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("Timed out fetching remote usage for job %s", job_ctx.job_id)
            # sstat is still running on an executor thread; exit hard rather
            # than let asyncio.run's finalizer join it and hang (B-C4).
            _bounded_exit(1)
        finally:
            await collector.stop()

    snap = asyncio.run(_run())
    _print_remote_summary(job_ctx, snap)


def _hop_to_compute_node(job_ctx: JobContext, args: argparse.Namespace) -> bool:
    """Re-launch the live TUI on the job's compute node via ``srun --overlap``.

    From a login node the cgroups aren't reachable, so instead of degrading to
    a text summary we attach to the job's allocation and run the full dashboard
    where the data actually lives. Returns ``True`` if the hop ran to
    completion (caller should exit), ``False`` if it wasn't possible (caller
    should fall back to the remote summary).
    """
    # SLURMWATCH_NO_HOP is set on the relaunched process (belt-and-suspenders
    # against any loop) and lets a user opt out of the behavior entirely.
    if _env_disables_hop():
        return False
    # A TUI needs a terminal; when piped/redirected the summary is more useful.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    srun = shutil.which("srun")
    if srun is None:
        return False
    node = job_ctx.nodelist_resolved[0] if job_ctx.nodelist_resolved else None
    if not node:
        return False

    # `srun --jobid=` only accepts the numeric JobId; the user-facing form
    # ("12345_3" / "123+1") is rejected, so use the raw id there. The inner
    # positional keeps the user's form — scontrol on the node re-resolves it.
    raw_id = job_ctx.raw_job_id or job_ctx.job_id
    # Relaunch *this* interpreter's slurmwatch by absolute path so it resolves
    # over the shared filesystem without depending on the compute node's PATH.
    inner = [sys.executable, "-m", "slurmwatch", job_ctx.job_id]
    if args.ascii:
        inner.append("--ascii")
    if args.interval is not None:
        inner += ["--interval", str(args.interval)]
    cmd = [
        srun,
        f"--jobid={raw_id}",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        "--pty",
        *inner,
    ]
    # Start from our env but drop the surrounding allocation's SLURM_* sizing
    # vars (e.g. if launched from inside another salloc) so the step's request
    # comes only from the explicit flags; keep SLURM_CONF, which srun needs.
    child_env = {
        k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"
    }
    child_env["SLURMWATCH_NO_HOP"] = "1"
    # A short, "loading"-style line (a spinner glyph + the node) instead of the
    # old verbose two-clause sentence; srun's inner TUI clears the screen once it
    # attaches, so this shows only during the brief connect. ASCII-safe glyph
    # when --ascii, so a non-UTF-8 terminal doesn't show a stray box.
    spin = "..." if args.ascii else "⠿"
    dots = "..." if args.ascii else "…"
    print(f"{spin}  connecting to {node} {dots}  (Ctrl-C to cancel)", file=sys.stderr)
    start = time.monotonic()
    try:
        result = subprocess.run(cmd, env=child_env)
    except KeyboardInterrupt:
        sys.exit(130)
    except OSError as exc:
        logger.debug("srun hop did not run: %s", exc)
        return False
    rc = result.returncode
    elapsed = time.monotonic() - start
    # rc == 0 is a clean quit; rc == 130 is the user hitting Ctrl-C inside the
    # live TUI. In both cases the dashboard was shown, so don't dump the text
    # summary on top of the session they already saw.
    if rc in (0, 130):
        return True
    # Any other non-zero exit means no clean TUI session. A fast failure is
    # almost always srun refusing to attach (--overlap denied, node gone). The
    # earlier "ran >=3s so it must have worked" heuristic silently swallowed a
    # *slow* attach failure, leaving a blank screen (B-P8); now every non-clean
    # exit falls back to the remote summary, with a message that says why.
    if elapsed >= 3.0:
        print(
            f"slurmwatch: the session on {node} exited with code {rc}; "
            "showing the remote summary instead.",
            file=sys.stderr,
        )
    else:
        print(
            f"slurmwatch: couldn't attach on {node} (srun exit {rc}); "
            "showing the remote summary instead.",
            file=sys.stderr,
        )
    return False


def _fmt_wait(seconds: int) -> str:
    """A compact wait duration: ``45s`` / ``3m`` / ``1h 5m`` / ``2d 3h``."""
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        h, m = divmod(seconds, 3600)
        return f"{h}h {m // 60}m"
    d, rem = divmod(seconds, 86400)
    return f"{d}d {rem // 3600}h"


def _print_pending_summary(pending: PendingJob, stream: Any = None) -> None:
    """Plain-text 'why / when / where' report for a PENDING job (non-TUI paths)."""
    out = stream if stream is not None else sys.stdout

    def emit(line: str) -> None:
        print(line, file=out)

    now = time.time()
    emit(f"Job {pending.job_id}  {pending.partition}  PENDING")
    reason = pending.reason or "None"
    emit(f"  Why    {reason} — {explain_reason(pending.reason)}")
    est = pending.start_time_estimate
    if est is not None and est >= now - 1:
        rel = _fmt_wait(int(est - now))
        when = time.strftime("%a %H:%M", time.localtime(est))
        emit(f"  When   estimated start {when} (in ~{rel}; scheduler estimate, may change)")
    else:
        emit("  When   not yet estimated by the scheduler")
    if pending.submit_time is not None:
        sub = time.strftime("%b %d %H:%M", time.localtime(pending.submit_time))
        emit(
            f"         submitted {sub} · waiting {_fmt_wait(int(now - pending.submit_time))} so far"
        )
    req = f"{pending.req_nodes} node(s), {pending.req_cpus} CPU"
    if pending.req_mem_bytes > 0:
        req += f", {_fmt_gib(pending.req_mem_bytes)}"
    if pending.req_gpus > 0:
        req += f", {pending.req_gpus}x {pending.req_gpu_type or 'GPU'}"
    emit(f"  Needs  {req}")
    try:
        running, waiting = resolve_queue_counts(pending.partition)
        emit(f"         queue on {pending.partition}: {running} running · {waiting} pending")
    except Exception:
        pass
    parts = resolve_cluster_partitions(pending.partition)
    if parts:
        emit("  Where  cluster capacity right now:")
        alts = []
        for p in parts[:10]:
            fits = partition_fits_now(pending, p)
            if fits and not p.is_current:
                alts.append(p)
            marker = "FITS NOW" if fits else ("down" if not p.available else "full")
            cur = " (current)" if p.is_current else ""
            gpus = ",".join(p.gpu_types[:3]) or "-"
            emit(
                f"           {p.name:<16} {p.free_nodes:>3} free / {p.cpus_idle:>5} idle CPU · "
                f"{gpus:<14} {marker}{cur}"
            )
        if alts:
            best = alts[0]
            emit(
                f"  Tip    {best.name} has room for this request now — requeue with: "
                f"scontrol update JobId={pending.job_id} Partition={best.name}"
            )
        else:
            emit("  Tip    no partition currently has free capacity for this request; it")
            emit("         will start once resources free up (the estimate above is Slurm's).")
    emit("  source: scontrol/sinfo/squeue (a queue estimate; actual start is up to the scheduler)")


def _run_pending(pending: PendingJob, config: SlurmwatchConfig, args: argparse.Namespace) -> None:
    """Show the pending-job view: the live TUI on a real terminal, else text."""
    interactive = not (args.once or args.log) and sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        _print_pending_summary(pending)
        return
    with _console_logging_suspended():
        try:
            from .tui import PendingApp

            app = PendingApp(pending, config)
            app.run(mouse=_mouse_enabled())
            return
        except Exception as exc:
            logger.error("TUI error: %s", exc)
    _print_pending_summary(pending)


def _run_interactive(job_id: str, config: SlurmwatchConfig, args: argparse.Namespace) -> None:
    job_ctx, pending = _resolve_running_or_pending(job_id)
    if pending is not None:
        # A queued job has no telemetry to show — surface why/when/where instead
        # of the dead-end "only running jobs can be monitored" error (#60).
        _run_pending(pending, config, args)
        return
    assert job_ctx is not None
    if job_ctx.remote:
        # Off the compute node: try to hop onto it for the real live TUI, and
        # only fall back to the sstat-derived text summary if that's not doable.
        if _hop_to_compute_node(job_ctx, args):
            return
        _run_remote_summary(job_ctx, config)
        return
    collector = TelemetryCollector(job_ctx, config)
    # Buffer slurmwatch logging while the TUI owns the screen so a transient
    # collector warning/traceback can't corrupt the dashboard; replayed on exit.
    with _console_logging_suspended():
        try:
            from .tui import SlurmwatchApp

            app = SlurmwatchApp(job_ctx=job_ctx, collector=collector, config=config)
            app.run(mouse=_mouse_enabled())
        except Exception as exc:
            logger.error("TUI error: %s", exc)
            sys.exit(1)
        finally:
            # stop_sync() sets the stop event and shuts NVML down synchronously;
            # the background task is torn down when the app's event loop closes.
            collector.stop_sync()
    if app.return_code:
        sys.exit(app.return_code)


def _infer_use_json(fmt: str, log_path: str) -> bool:
    """Whether headless ``--log`` output should be JSON.

    An explicit, already-validated ``fmt`` ("json"/"csv") always wins; otherwise
    the format is inferred from the file extension and defaults to JSON. The
    extension test is case-insensitive, so ``out.CSV`` infers CSV — matching the
    case-folding already applied to ``SLURMWATCH_FORMAT`` (#53).
    """
    if fmt in ("json", "csv"):
        return fmt == "json"
    return not log_path.lower().endswith(".csv")


def _csv_max_gpus_from_header(log_path: str, dialect: str) -> int | None:
    """The GPU-column width already established by an existing CSV log's header.

    Counts the ``gpu_<N>_index`` columns on the first line so an ``--append`` run
    reuses the file's layout instead of re-deriving a (possibly different) width
    from its own job — which would misalign the appended rows (#62). Returns
    ``None`` when the file is missing/empty or isn't a slurmwatch CSV (no
    ``timestamp`` column), so the caller falls back to snapshot-based sizing.
    """
    try:
        with open(log_path, newline="") as f:
            first = f.readline()
    except OSError:
        return None
    if not first.strip():
        return None
    try:
        cols = next(csv.reader([first], dialect=dialect))
    except (csv.Error, StopIteration):
        return None
    if "timestamp" not in cols:
        return None
    return sum(1 for c in cols if c.startswith("gpu_") and c.endswith("_index"))


def _run_headless(
    job_id: str,
    config: SlurmwatchConfig,
    log_path: str,
    fmt: str = "",
    append: bool = False,
) -> None:
    job_ctx, pending = _resolve_running_or_pending(job_id)
    if pending is not None:
        # A queued job has no telemetry to log yet — report why/when/where on
        # stderr and exit without creating an empty log file (#60).
        print(f"slurmwatch: job {job_id} is PENDING — nothing to log yet.", file=sys.stderr)
        _print_pending_summary(pending, stream=sys.stderr)
        return
    assert job_ctx is not None

    config.poll_interval = config.headless_interval
    print(
        f"slurmwatch: logging job {job_id} to {log_path} (PID {os.getpid()})",
        file=sys.stderr,
    )

    asyncio.run(_headless_loop(job_ctx, config, log_path, fmt, append))


# Grace period given to an in-flight log write to finish after a SIGINT/SIGTERM
# before we conclude the sink is wedged and hard-exit (B-C6). A module constant
# so tests can shorten it.
_HEADLESS_STUCK_WRITE_GRACE_SECONDS = 2.0


async def _headless_loop(
    job_ctx: JobContext,
    config: SlurmwatchConfig,
    log_path: str,
    fmt: str = "",
    append: bool = False,
) -> None:
    collector = TelemetryCollector(job_ctx, config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    # fmt already folds in a validated, normalized SLURMWATCH_FORMAT (see the
    # caller); an explicit format always wins, the extension is only a fallback.
    use_json = _infer_use_json(fmt, log_path)

    try:
        await collector.start()

        # When appending to an existing CSV, its header fixes the column layout for
        # the whole file, so reuse that width — otherwise a run whose job has a
        # different GPU count writes rows that don't line up under the existing
        # header (a regression the per-run #38 sizing introduced). None when the
        # file is new/empty/JSON, in which case we size from the first snapshot.
        forced_max_gpus = (
            _csv_max_gpus_from_header(log_path, config.csv_dialect)
            if append and not use_json
            else None
        )

        mode = "a" if append else "w"
        with open(log_path, mode) as f:
            csv_writer: Any = None
            # Sized to the job's actual GPU count from the first snapshot, then
            # fixed for the file's lifetime so every row lines up under the one
            # header (a >8-GPU node isn't clipped at 8, #38).
            csv_max_gpus = 0
            # Skip the CSV header when appending to a non-empty file. A pipe or
            # /dev/stdout isn't seekable (tell() raises) — treat it as fresh so
            # `--log /dev/stdout` can stream to another process (the node switcher
            # reads exactly this).
            try:
                header_needed = f.tell() == 0
            except (OSError, ValueError):
                header_needed = True

            def _write(snap: TelemetrySnapshot) -> None:
                nonlocal csv_writer, csv_max_gpus
                if use_json:
                    f.write(snap.to_json() + "\n")
                else:
                    if csv_writer is None:
                        csv_max_gpus = (
                            forced_max_gpus
                            if forced_max_gpus is not None
                            else max(len(snap.gpus), job_ctx.gpu_count_requested)
                        )
                        csv_writer = csv.writer(f, dialect=config.csv_dialect)
                        if header_needed:
                            csv_writer.writerow(TelemetrySnapshot.csv_header(csv_max_gpus))
                    csv_writer.writerow(snap.to_csv_row(csv_max_gpus))
                f.flush()

            while not shutdown_event.is_set():
                try:
                    snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No frame this second: the job may have ended (the collector
                    # stops enqueuing then). Exit cleanly instead of spinning
                    # forever writing nothing (#28).
                    if collector.job_ended:
                        print("slurmwatch: job ended", file=sys.stderr)
                        break
                    continue

                # Write on a worker thread, and RACE it against shutdown so a
                # stalled sink (a full pipe whose reader stopped, a hung NFS /
                # scratch mount) can't wedge the loop past a SIGINT/SIGTERM — a
                # plain `await` on the write would never return and the handler's
                # event could never be re-checked (B-C6).
                write_fut = loop.run_in_executor(None, _write, snapshot)
                shutdown_fut = asyncio.ensure_future(shutdown_event.wait())
                race: set[asyncio.Future[Any]] = {write_fut, shutdown_fut}
                try:
                    await asyncio.wait(race, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    shutdown_fut.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await shutdown_fut
                if not write_fut.done():
                    # Shutdown fired mid-write. Give the in-flight write a brief
                    # grace to finish; if the sink is genuinely stuck, hard-exit
                    # rather than hang forever joining the wedged writer thread
                    # (asyncio.run's finalizer would otherwise block on it, B-C4).
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(write_fut),
                            timeout=_HEADLESS_STUCK_WRITE_GRACE_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        print(
                            "slurmwatch: shutting down — the log sink is not draining",
                            file=sys.stderr,
                        )
                        _bounded_exit(0)
                    # A real write error propagates to the outer `except OSError`.
                write_fut.result()  # surface any write error from the executor
                if collector.job_ended:
                    print("slurmwatch: job ended", file=sys.stderr)
                    break

    except OSError as exc:
        # Any open()/write failure — not just a missing parent dir: a directory
        # target (IsADirectoryError) or an unwritable path (PermissionError) are
        # sibling OSErrors, and used to escape a FileNotFoundError-only handler as
        # a raw traceback instead of this clean message (#52).
        logger.error("Cannot write log file: %s", exc)
        sys.exit(1)
    finally:
        await collector.stop()
        print("slurmwatch: monitoring stopped", file=sys.stderr)
