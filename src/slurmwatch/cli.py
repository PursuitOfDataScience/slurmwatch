# ruff: noqa: T201
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import errno
import fcntl
import io
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any, NoReturn

from ._version import VERSION
from .aio import reap_cancelled
from .collector import TelemetryCollector
from .config import (
    MIN_INTERVAL,
    MIN_REMOTE_INTERVAL,
    SlurmwatchConfig,
    _parse_bool,
    warn_unusable_env,
)
from .exceptions import (
    CgroupAccessError,
    CgroupNotFoundError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from .model import (
    CPU_UNDERUSE_ADVICE,
    JobContext,
    TelemetrySnapshot,
    _csv_text,
    cpu_is_underused,
)
from .pending import (
    _MAX_WHERE_ROWS,
    PendingJob,
    available_node_count,
    blocker_is_permanent,
    capacity_is_irrelevant,
    explain_reason,
    fit_blocker,
    format_gpu_types,
    is_held_like,
    is_usage_capped,
    largest_node_cpus,
    requeue_could_help,
    resolve_cluster_partitions,
    resolve_pending_job,
    resolve_priority_rank,
    resolve_queue_counts,
)
from .slurm import (
    SLURM_CMD_TIMEOUT,
    _job_owner_differs,
    acct_gather_disabled,
    current_username,
    is_job_active,
    resolve_array_task_counts,
    resolve_current_jobs,
    resolve_job_context,
)
from .units import format_bytes, mem_pair, per_node_suffix

# The first snapshot on a remote (login-node) context is an sstat call bounded by
# SLURM_CMD_TIMEOUT; the wait wrapping it must exceed that plus margin, or a
# slow-but-healthy controller trips the outer timeout while the data is in flight
# (A3). Used by both --once and the interactive remote summary so they agree.
_FIRST_SNAPSHOT_TIMEOUT = SLURM_CMD_TIMEOUT + 5.0

logger = logging.getLogger("slurmwatch")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)


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

    The flush is itself best-effort: on a closed pipe it raises the very
    ``BrokenPipeError`` some callers are here to handle, which would escape the
    handler and turn a clean exit into an unhandled traceback.
    """
    with contextlib.suppress(BrokenPipeError, ValueError, OSError):
        sys.stdout.flush()
    with contextlib.suppress(BrokenPipeError, ValueError, OSError):
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
        warn_unusable_env(
            "SLURMWATCH_NO_HOP", val, "not a boolean", "the safe reading: hop disabled"
        )
        return True


def _env_disables_ssh() -> bool:
    """Whether ``SLURMWATCH_NO_SSH`` disables the ssh-to-node fallback transport.

    Parsed as a boolean (like ``SLURMWATCH_NO_HOP``), so ``0``/``false`` keep it
    enabled; an unrecognized value is treated as "set" (disable). Also set on the
    relaunched remote process to stop any chance of a re-ssh loop.
    """
    val = os.environ.get("SLURMWATCH_NO_SSH")
    if val is None:
        return False
    try:
        return _parse_bool(val)
    except ValueError:
        warn_unusable_env(
            "SLURMWATCH_NO_SSH", val, "not a boolean", "the safe reading: ssh disabled"
        )
        return True


def _mouse_enabled(config: SlurmwatchConfig | None = None) -> bool:
    """Whether to let the TUI capture the mouse.

    Off by default so the terminal's own text selection and copy/paste keep
    working — slurmwatch is fully keyboard-driven (c/m/g/v to focus panels,
    arrows/PgUp/PgDn to scroll, q to quit). Set SLURMWATCH_MOUSE=1 to re-enable
    mouse support (e.g. wheel scrolling), at the cost of drag-to-select.

    Reads the CONFIG, which parsed the variable through the same validator every
    other knob uses: a bare `== "1"` here made `SLURMWATCH_MOUSE=7` silently false
    while `SLURMWATCH_ASCII=maybe` was a startup error (SW-17).
    """
    if config is not None:
        return config.mouse
    return _parse_bool(os.environ.get("SLURMWATCH_MOUSE", "0").strip() or "0")


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


def _help_color() -> bool:
    """Whether to colourise --help: only on a real terminal, honouring NO_COLOR."""
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )


class _ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Colourise the help so sections/examples are scannable at a glance.

    Colour is applied ONLY when stdout is a real terminal (respects NO_COLOR and
    TERM=dumb), so piped/redirected help and captured test output stay plain text.
    Option-invocation columns are deliberately left uncoloured — injecting ANSI
    there would throw off argparse's alignment maths.
    """

    @staticmethod
    def _paint(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if _help_color() else text

    def start_section(self, heading: str | None) -> None:  # section titles -> bold amber
        super().start_section(self._paint("1;33", heading) if heading else heading)

    def _format_usage(self, usage, actions, groups, prefix):  # type: ignore[no-untyped-def]
        text = super()._format_usage(usage, actions, groups, prefix)
        return text.replace("usage:", self._paint("1", "usage:"), 1)

    def _format_text(self, text: str) -> str:
        out = super()._format_text(text)
        if not _help_color():
            return out
        painted: list[str] = []
        for raw in out.splitlines(keepends=True):
            body, nl = raw.rstrip("\n"), ("\n" if raw.endswith("\n") else "")
            stripped = body.lstrip()
            if stripped == "examples:":  # match the argparse section headings
                body = self._paint("1;33", body)
            elif stripped.startswith("sw "):  # an example command line
                m = re.match(r"^(\s*)(\S.*?)(\s{2,}.*)?$", body)
                if m:
                    body = f"{m.group(1)}\033[32m{m.group(2)}\033[0m{m.group(3) or ''}"
            elif stripped.startswith("'sw'") or stripped.startswith("In the dashboard"):
                body = f"\033[2m{body}\033[0m"
            painted.append(body + nl)
        return "".join(painted)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmwatch",
        formatter_class=_ColorHelpFormatter,
        description=(
            "Live, process-isolated CPU / memory / GPU telemetry for your running\n"
            "Slurm jobs. Run with no arguments to pick from your running jobs; with\n"
            "--once/--log (or any non-terminal stdout) a lone job is attached\n"
            "directly."
        ),
        epilog=(
            "examples:\n"
            "  sw                    watch your running job live (auto-detected)\n"
            "  sw 1234567            watch a specific job  (array task: sw 1234567_3)\n"
            "  sw --demo             explore the dashboard with simulated data\n"
            "  sw --once             print a single snapshot and exit\n"
            "  sw --log run.jsonl    run headless, stream telemetry to a file\n"
            "\n"
            "'sw' is the short alias for 'slurmwatch' — the same command either way.\n"
            "In the dashboard the bottom bar lists the keys; press q to quit.\n"
            "\n"
            "how it reaches the node (from a login node it relocates itself):\n"
            "  srun --overlap monitor step, or ssh when a step cannot be granted\n"
            "  the job's GPUs — which is the usual case for multi-node training,\n"
            "  where an inner srun holds them all and Slurm cannot share GRES.\n"
            "  One ssh login per node per session; set SLURMWATCH_NO_SSH=1 to stay\n"
            "  on srun only (GPU numbers then read as unavailable), or\n"
            "  SLURMWATCH_NO_HOP=1 to not relocate at all. SLURMWATCH_HOP_TIMEOUT\n"
            "  (seconds, 2-120) raises the wait when step creation is slow."
        ),
    )
    parser.add_argument(
        "job_id",
        nargs="?",
        type=str,
        default=None,
        help="Slurm job ID to monitor. Takes the forms Slurm's own tools PRINT, "
        "since that is what gets pasted: 12345, an array task 12345_3, a pending "
        "array's range 12345_[1-9%%3] (resolved to the array job), a step 12345.0 "
        "(resolved to its job), or a het component 123+1. Auto-discovers if omitted.",
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
        help=(
            "Polling interval in seconds (default: 0.5 for TUI, 1.0 for headless; "
            "raised to a 0.1s floor on the node, 1.0s off it)"
        ),
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
        config.requested_interval = args.interval
        config.poll_interval = args.interval
        config.headless_interval = args.interval

    # Re-apply the interval floor: a CLI --interval bypasses the clamp that
    # from_env enforces, and --interval 0.0001 would busy-loop the node (B-P1).
    config.clamp()

    # Normalize an empty/whitespace id to None so it takes the auto-discover path. ""
    # is not None, so it used to skip discovery and run `scontrol show job -d ""`, which
    # means "every job" -- slurmwatch then silently monitored whatever job the resolver
    # landed on (the caller's own $SLURM_JOB_ID allocation) while every emitted record
    # kept the empty id, so a --log CSV had a blank job_id primary key on every row.
    # Reached by the ordinary `sw "$JOBID" --once` in a script where JOBID is unset.
    job_id = (args.job_id or "").strip() or None
    if job_id is not None:
        # A `<job>.<step>` id can never resolve (scontrol has no such form), so
        # rewrite it to its job before any Slurm call rather than fail with a reason
        # that isn't true (SW-14).
        job_id = _job_id_without_step(job_id)
        # Likewise a bracketed array RANGE — what squeue prints for a pending array —
        # names no single task, so resolve it to the array job it belongs to.
        job_id = _job_id_without_array_range(job_id)
    log_path: str | None = args.log
    headless = log_path is not None
    once = args.once

    if once and headless:
        logger.error("--once and --log are mutually exclusive")
        sys.exit(1)
    # --json is shorthand for --format json; a contradicting --format is an error
    # (don't silently drop --json), mirroring the --once/--log guard above.
    if args.json and args.format and args.format != "json":
        logger.error("--json conflicts with --format %s (choose one)", args.format)
        sys.exit(1)

    fmt = args.format or ("json" if args.json else "")
    # SLURMWATCH_FORMAT only affects machine output (--once/--log). Validate/consume
    # it ONLY on those paths, so a stale/bogus value exported in the shell can't
    # block the interactive TUI (which never uses an output format).
    if once or headless:
        try:
            fmt = fmt or _env_output_format()
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)

    # Record the machine format for the paths that die during job RESOLUTION: that is
    # where three of the four no-telemetry outcomes are decided, and they ran before
    # any snapshot existed, so they had only prose on stderr (SW-27). `--once`'s
    # documented default is csv; a `--log` file's format comes from its extension.
    global _MACHINE_FORMAT, _MACHINE_CSV_DIALECT
    if once:
        _MACHINE_FORMAT = fmt or "csv"
    elif headless:
        _MACHINE_FORMAT = "json" if _infer_use_json(fmt, args.log or "") else "csv"
    _MACHINE_CSV_DIALECT = config.csv_dialect

    # Flags that only affect machine output are no-ops on the interactive TUI; warn
    # rather than silently dropping them (the interactive path ignores fmt/append).
    if args.append and not headless:
        logger.warning("--append has no effect without --log; ignoring")
    # --json is a no-op ONLY on the real live TUI (both stdin and stdout ttys) — a
    # RUNNING job on a redirected/piped stdout instead degrades to one snapshot
    # (see `_run_interactive`) and DOES honour it, so warning "ignoring" there would
    # contradict what actually happens a moment later. A PENDING job's plain-text
    # report ignores it either way, but that's the safer direction to miss the
    # warning in — never the direction that warns "ignored" and then uses it.
    if args.json and not (once or headless) and sys.stdin.isatty() and sys.stdout.isatty():
        logger.warning("--json has no effect without --once/--log; ignoring")

    if job_id is None:
        if os.environ.get("SLURMWATCH_MOCK") == "1":
            job_id = "12345"
        else:
            # The tty test belongs HERE, not only inside the paths below: without
            # it, `slurmwatch > log` and `slurmwatch | tee` with no job id built
            # the Textual app anyway, entered the alternate screen, drew the job
            # picker into the pipe and waited forever for a keypress that cannot
            # arrive — 73 KB of escape sequences and a process to kill. Non-tty
            # falls through to the headless discovery branch, which attaches a lone
            # job (the path that already works) and names the ids otherwise. SW-10.
            job_id = _auto_discover_job_id(
                config,
                interactive=(not (once or headless) and sys.stdin.isatty() and sys.stdout.isatty()),
            )
            if job_id is None:
                return

    # Ctrl-C is a normal way to stop any of these, so report it as one. Without this a
    # SIGINT during a slow scontrol or while waiting on the first snapshot escaped as a
    # six-line KeyboardInterrupt traceback; the hop and ssh paths already did this.
    try:
        if headless:
            assert log_path is not None
            _run_headless(job_id, config, log_path, fmt, append=args.append)
        elif once:
            _run_once(job_id, config, fmt)
        else:
            _run_interactive(job_id, config, args)
    except KeyboardInterrupt:
        sys.exit(130)


def _auto_discover_job_id(config: SlurmwatchConfig, interactive: bool = True) -> str | None:
    # Resolved from the uid first: under cron/systemd/`env -i` there is no $USER to
    # read, and `squeue -u ""` answers "no jobs" for a user whose job is running
    # (SW-11).
    username = current_username()
    logger.info("Auto-discovering running/pending jobs for user %s...", username)

    try:
        jobs = resolve_current_jobs(username)
    except Exception as exc:
        logger.error("Failed to query Slurm jobs: %s", exc)
        sys.exit(1)

    if not jobs:
        user_message = (
            f"No running or pending Slurm jobs found for user '{username}'. "
            "Launch a job first or provide a job_id argument."
        )
        print(user_message, file=sys.stderr)
        sys.exit(1)

    if not interactive:
        # Headless (--once/--log): no picker possible. A lone job attaches directly;
        # multiple is ambiguous, so require an explicit job_id.
        if len(jobs) == 1:
            jid: str = str(jobs[0]["job_id"])
            logger.info("Attaching to job %s", jid)
            return jid
        listing = ", ".join(str(j["job_id"]) for j in jobs)
        # Name the reason a picker isn't an option, since on a pipe/redirect the
        # caller never asked for --once/--log and would otherwise read this as
        # slurmwatch refusing for no reason (SW-10).
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            logger.error(
                "%d jobs found (%s) and stdout is not a terminal, so there is no "
                "picker; pass a job id.",
                len(jobs),
                listing,
            )
            sys.exit(1)
        logger.error("Multiple jobs found (%s); pass the job_id to monitor.", listing)
        sys.exit(1)

    # Interactive: ALWAYS show the picker — even for a single job — so `sw` behaves
    # consistently and the user sees the job (state / elapsed) and confirms before
    # diving into the dashboard, rather than being dropped straight in.
    from .tui import SlurmwatchApp

    # Pass a refresh callable so the picker keeps its list live (adds newly-submitted
    # jobs, drops finished ones) while it's open — same username the initial list used.
    app = SlurmwatchApp(jobs=jobs, config=config, refresh=lambda: resolve_current_jobs(username))
    with _console_logging_suspended():
        app.run(mouse=_mouse_enabled(config))
    if app.return_code:
        sys.exit(app.return_code)
    return None


def _raw_job_id(job_id: str) -> str:
    """The numeric id `srun --jobid=` accepts: an array task's or het component's
    user-facing form ("12345_3" / "123+1") is rejected there."""
    return re.split(r"[_+]", job_id, maxsplit=1)[0]


# A Slurm job id is numeric, optionally an array task (`12345_3`), an array RANGE as
# squeue prints one for a pending array (`12345_[1-9%3]`, throttle and all) or a het
# component (`123+1`); `.step` and `_[range]` are normalised away before this. Anything
# else was never an id, so "does not exist in the Slurm database" is a claim about the
# database when the real problem is the FORM — the same wrong-diagnosis shape as
# SW-7/SW-19/SW-22, and the likely intent (a job NAME, a path, a node) has a different
# remedy.
_JOB_ID_FORM = re.compile(r"^\d+(?:_(?:\d+|\[[\d,\-%]+\]))?(?:\+\d+)?$")


def _die_on_resolve_error(exc: Exception, job_id: str) -> NoReturn:
    """Map a resolve failure to a clear message + exit(1)."""
    if isinstance(exc, JobNotFoundError) and not _JOB_ID_FORM.match(job_id.strip()):
        _emit_no_telemetry_facts(job_id, "bad_job_id", f"{job_id!r} is not a job id")
        logger.error(
            "%r is not a job id (expected 12345, an array task 12345_3, an array "
            "range 12345_[1-9], or a het component 123+1). If that was a job NAME, "
            "find its id with "
            "`squeue --me -o '%%i %%j'` (running) or `sacct --name=%s -o JobID,State` "
            "(finished).",
            job_id,
            job_id,
        )
    elif isinstance(exc, JobNotFoundError):
        _emit_no_telemetry_facts(job_id, "job_unknown", "job does not exist in the Slurm database")
        logger.error("Job %s does not exist in the Slurm database.", job_id)
    elif isinstance(exc, JobNotRunningError):
        # A finished job is a normal outcome, not a malfunction — but it is a
        # DIFFERENT one from "never existed", and the two shared an exit code and were
        # separable only by reading the prose (SW-27).
        _emit_no_telemetry_facts(job_id, "job_finished", str(exc))
        logger.error(str(exc))
    elif isinstance(exc, (CgroupNotFoundError, CgroupAccessError)):
        # Print the command the HOP actually runs, not a shorter-looking one: this
        # interpreter by absolute path (so it resolves over the shared filesystem
        # instead of depending on the compute node's PATH — bare `slurmwatch` there
        # fails with execve(): No such file or directory) and an explicit job id (or
        # the inner process auto-discovers, finds nothing, and exits 1). Both are
        # things `_hop_to_compute_node` already gets right; only this string didn't.
        # SW-22.
        logger.error(
            "Job %s: %s\n\nTo resolve this:\n"
            "  1. Make sure you are on the compute node running the job\n"
            "  2. Verify the job is in RUNNING state\n"
            "  3. Try: srun --jobid=%s --overlap %s -m slurmwatch %s",
            job_id,
            exc,
            _raw_job_id(job_id),
            sys.executable,
            job_id,
        )
    elif isinstance(exc, SlurmCommandError):
        # Machine-readable here too. Every other no-telemetry outcome emits its facts
        # row, so `--once --json` on a host without Slurm wrote NOTHING to stdout while
        # a finished job on the same command wrote a full object — a consumer could not
        # tell "no Slurm here" from a crash. The token comes from the exception's own
        # classification, never from grepping its prose: the three causes want
        # different actions from the reader — reinstall/module-load, report a
        # slurmwatch bug, or wait — and matching my own wording would break the moment
        # the wording improved.
        token = {
            "unavailable": "slurm_unavailable",
            "unsupported": "slurm_query_unsupported",
        }.get(getattr(exc, "kind", "transient"), "slurm_error")
        _emit_no_telemetry_facts(job_id, token, str(exc))
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
        except Exception:
            # Any pending-resolution failure (JobNotPending/NotFound/SlurmCommand or
            # anything else) means we can't classify the job — report the original
            # not-running error with the usual clear message.
            _die_on_resolve_error(exc, job_id)
    except Exception as exc:
        _die_on_resolve_error(exc, job_id)


def _run_once(job_id: str, config: SlurmwatchConfig, fmt: str = "") -> None:
    job_ctx, pending = _resolve_running_or_pending(job_id)
    if pending is not None:
        # A queued job has no snapshot to emit, and prose must not go to stdout — a
        # downstream jq/CSV reader would choke on it (#60 review). But "keep stdout
        # clean" was implemented as "leave stdout EMPTY", which is SW-27's finding on
        # the fifth and commonest no-telemetry outcome: a poller asking what its job
        # is doing got zero bytes for the state a job spends most of its life in, and
        # had to parse English on stderr to learn why. The facts go to stdout in the
        # requested format; the human report still goes to stderr, so piping to jq
        # gets both without them mixing.
        _emit_facts_payload(_pending_facts(pending), fmt or "csv", config)
        _print_pending_summary(pending, stream=sys.stderr, ascii_mode=config.ascii_mode)
        sys.exit(1)
    assert job_ctx is not None
    if job_ctx.remote and _job_owner_differs(job_ctx):
        # Another user's job: sstat is permission-denied, so the collector would
        # emit an all-zero row a consumer misreads as "using nothing". Print the
        # honest read-only summary to stderr and exit non-zero instead (M2) — the
        # interactive path already gates this; the machine paths did not.
        #
        # A machine format was still REQUESTED, so answer in it: `--once --json | jq`
        # used to get prose it can't parse, when the facts Slurm does hand out
        # cross-user (owner, state, placement, request) are perfectly expressible as
        # JSON with the measured fields null (SW-8).
        #
        # `--once` alone counts as a machine format: its documented default IS csv, so
        # `fmt or "csv"` — it was falling through to the prose branch and emitting zero
        # bytes on stdout while --json got a full payload from the same event (SW-27).
        _emit_facts_payload(_foreign_facts(job_ctx), fmt or "csv", config)
        # rc 0, matching the DEFAULT path on the identical event. The two used to
        # disagree about whether a colleague's job is an error: `slurmwatch <job>` said
        # success on stdout, `--once` said failure on stderr, for the same 453 bytes.
        # A complete payload was produced; `telemetry_available: false` inside it is
        # what says there is no telemetry, and that is machine-readable where an exit
        # code shared with "job does not exist" is not (SW-27).
        sys.exit(0)
    # Off-node, sstat sees only the job's TRACKED process tree. A job whose work
    # runs in detached workers (R multisession/PSOCK, nohup, setsid) reads ~0.1 of 8
    # cores while saturating all 8 — measured 79-100x low on CPU and 27x on memory,
    # and this is the interface right-sizing decisions are actually made from. The
    # on-node reading is correct and reachable: run ourselves there over
    # `srun --overlap` and pass the child's output through unchanged. Falls back to
    # sstat when no step can be created, so today's behaviour is the floor. SW-23.
    if job_ctx.remote and _once_on_node(job_ctx, config, fmt):
        return
    # Only now is the sampling path known, so this is where the floor belongs: the
    # interactive and headless paths both applied it and `--once` did not, so
    # `--interval 0.001` was accepted in silence here while being raised (with a
    # notice) everywhere else. The window it produces is real: measured on one idle
    # job, 0.1 cores at 0.001s against 0.3 at the default — the same job, three times
    # the reading, decided by a flag the tool elsewhere calls pathological. SW-13's
    # fix, on the interface it missed.
    _apply_sampling_floor(config, job_ctx.remote)
    collector = TelemetryCollector(job_ctx, config)
    asyncio.run(_once_loop(collector, json_output=fmt == "json", csv_dialect=config.csv_dialect))


# Said once per process: the step is brief, but a caller in a loop should know a
# step is being created in their allocation at all.
_MONITOR_STEP_NOTED = False


def _once_on_node(job_ctx: JobContext, config: SlurmwatchConfig, fmt: str) -> bool:
    """Take one snapshot ON the job's node via ``srun --overlap``; True if it worked.

    Unlike the interactive hop this needs no terminal — it captures a subprocess's
    stdout — so the tty gate that (correctly) guards the TUI does not apply. The
    child is this same interpreter by absolute path with an explicit raw job id, the
    two things round 32 established a relaunch needs, and `SLURMWATCH_ON_NODE`/
    `SLURMWATCH_NO_HOP` on its environment make a second hop impossible.
    """
    if _env_disables_hop() or _env_says_already_on_node():
        return False
    srun = shutil.which("srun")
    node = job_ctx.nodelist_resolved[0] if job_ctx.nodelist_resolved else None
    if srun is None or not node:
        return False
    raw_id = job_ctx.raw_job_id or job_ctx.job_id
    inner = [sys.executable, "-m", "slurmwatch", job_ctx.job_id, "--once"]
    if fmt:
        inner += ["--format", fmt]
    if config.ascii_mode:
        inner.append("--ascii")
    # Forward the interval the user actually asked for, as both interactive hops do.
    # Dropped here, it was silently replaced by the child's default — and the child is
    # the process that takes the measurement, so the flag had no effect at all. It
    # applies its own (on-node) floor, and announces that itself if it raises.
    if config.poll_interval != SlurmwatchConfig().poll_interval:
        inner += ["--interval", f"{config.poll_interval:g}"]
    child_env = {
        k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"
    }
    child_env["SLURMWATCH_ON_NODE"] = "1"
    child_env["SLURMWATCH_NO_HOP"] = "1"
    cmd = [
        srun,
        f"--jobid={raw_id}",
        "--overlap",
        f"--immediate={_hop_connect_timeout()}",
        "--mem=0",
        "--gres=none",  # never contend for the job's GPUs just to read a counter
        "--input=none",
        "-w",
        node,
        "-n1",
        *inner,
    ]
    # Creating a step is not free: while it lives, a NORMAL `srun` inside the same
    # allocation is refused ("Requested nodes are busy") — measured on this cluster,
    # and `--overlap`/`--exact -c1` do not change it, which is why the dashboard has
    # a contention note at all. The step here lives ~2-4s, but this is the SCRIPTED
    # path, so a sizing loop creates one per call. Correct numbers are worth it (the
    # alternative reads 0.00 of 6 cores on an R multisession job), but not silently.
    global _MONITOR_STEP_NOTED
    if not _MONITOR_STEP_NOTED:
        _MONITOR_STEP_NOTED = True
        logger.warning(
            "taking a brief monitor step on %s to read the job's real CPU/memory "
            "(sstat cannot see detached workers). While it runs, a plain `srun` in "
            "that allocation is refused; export SLURMWATCH_NO_HOP=1 to read sstat "
            "instead.",
            node,
        )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=child_env, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("on-node --once did not run (%s); falling back to sstat", exc)
        return False
    if result.returncode != 0 or not result.stdout.strip():
        logger.debug("on-node --once failed (rc=%s); falling back to sstat", result.returncode)
        return False
    # Pass the child's snapshot through verbatim — it is already in the requested
    # format, measured where the numbers are real.
    sys.stdout.write(result.stdout)
    with contextlib.suppress(BrokenPipeError):
        sys.stdout.flush()
    return True


# Liveness comes from polling squeue/sstat, not from a notification, so the last row
# or two of a log can post-date the job's exit: a `sleep 35` job's final snapshot read
# `elapsed_seconds: 43` with `cpu.usage_percent: 0.0` and 2.3 MB of memory — post-exit
# residue, not the job. The reporter (round 6) filed it as "worth knowing if the last
# row is used as a job's final reading: prefer the max over the series, not the tail",
# which is exactly right and is what the peak columns already are, so say so where the
# reader is looking rather than leaving it to be rediscovered.
_JOB_ENDED_NOTE = (
    "slurmwatch: job ended (the last row may post-date the exit — for a final "
    "reading use the peak columns, or the max over the series, not the tail)"
)


async def _once_loop(
    collector: TelemetryCollector, json_output: bool, csv_dialect: str = "excel"
) -> None:
    await collector.start()
    try:
        snapshot = await asyncio.wait_for(
            collector.next_snapshot(), timeout=_FIRST_SNAPSHOT_TIMEOUT
        )
        if not snapshot.usage_sampled:
            # NOT a measurement. Off-node, Slurm samples accounting every ~30s, so a
            # young job has no sample yet and every metric reads 0. The plain-text
            # summary has always said "usage not yet sampled by Slurm"; this channel
            # published `usage_ns: 0`, `limit_bytes: 0` and `source: "sstat"` — which a
            # right-sizing consumer reads as "this job uses nothing" and acts on by
            # shrinking --mem and --cpus-per-task to the floor. Emit the same
            # no-telemetry shape every other empty-handed outcome uses, with a token
            # that says which, and a non-zero code so a script does not record it as a
            # completed reading.
            _emit_facts_payload(
                _no_telemetry_facts(
                    collector.job_ctx.job_id,
                    "usage_not_sampled",
                    "Slurm has not sampled this job's usage yet (accounting samples "
                    "roughly every 30s) — nothing to measure from off the node. Try "
                    "again shortly, or run on the compute node.",
                    collector.job_ctx,
                ),
                "json" if json_output else "csv",
                SlurmwatchConfig(csv_dialect=csv_dialect),
            )
            sys.stdout.flush()
            sys.exit(1)
        if json_output:
            print(snapshot.to_json())
        else:
            # Size the CSV GPU columns to this job's actual device count so a
            # >8-GPU node (or a many-slice MIG config) isn't silently clipped (#38).
            max_gpus = max(len(snapshot.gpus), collector.job_ctx.gpu_count_requested)
            writer = csv.writer(sys.stdout, dialect=csv_dialect)
            writer.writerow(TelemetrySnapshot.csv_header(max_gpus))
            writer.writerow(snapshot.to_csv_row(max_gpus))
        # Flush INSIDE the try. Piped stdout is block-buffered, so a payload smaller
        # than the buffer never touches the pipe here and EPIPE surfaced only at
        # interpreter-shutdown flush — outside this handler, which is why the guard
        # below looked correct but never fired: `--once --json | <early-closing reader>`
        # exited 120 with "Exception ignored ... BrokenPipeError" on stderr instead of 0.
        sys.stdout.flush()
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for first snapshot")
        # The collection that timed out is still on an executor thread; exit
        # hard so a wedged read can't hang us past the timeout (B-C4).
        _bounded_exit(1)
    except BrokenPipeError:
        # A downstream reader closed the pipe (e.g. `sw --once --json | head`): exit
        # quietly, not with a BrokenPipeError traceback. os._exit (via _bounded_exit)
        # skips the final stdout flush that would otherwise re-raise it (N6).
        _bounded_exit(0)
    finally:
        await collector.stop()


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _name_suffix(name: str) -> str:
    """``  name `<job name>``` for a text-summary header, or "" when Slurm reports none.

    The name is the one identity field the USER chose, so it answers "which experiment
    is this" where the id only answers "which record". Every TUI surface, --json and CSV
    carried it; all three plain-text summaries dropped it, which is exactly the report a
    user redirects to a file from a login node. Capped like every other name render.
    """
    if not name:
        return ""
    shown = name if len(name) <= 40 else name[:39] + "..."
    return f"  name `{shown}`"


def _print_remote_summary(
    job_ctx: JobContext, snap: TelemetrySnapshot, config: SlurmwatchConfig | None = None
) -> None:
    mem = snap.memory
    cpu = snap.cpu
    node = job_ctx.nodelist_display or "?"
    state = job_ctx.job_state or ""
    # This summary never consulted ascii_mode, so every em dash below reached a
    # terminal that had explicitly asked for none — the same leak class as the three
    # already fixed in the rendered views, on the surface a reader falls back to when
    # they cannot have the dashboard at all.
    dash = "-" if (config or SlurmwatchConfig()).ascii_mode else "\u2014"
    print(
        f"Job {job_ctx.job_id}  {job_ctx.partition}  {state}  on {node}"
        f"{_name_suffix(job_ctx.job_name)}"
    )
    if mem.current_bytes > 0 or cpu.usage_ns > 0:
        if mem.limit_bytes > 0:
            # units.mem_pair, not a local :.1f GiB — a --mem=400M job read
            # "0.0 GiB / 0.4 GiB" here long after the dashboard gauge was fixed,
            # because this renderer had its own copy of the arithmetic (SW-4).
            used_txt, limit_txt = mem_pair(mem.current_bytes, mem.limit_bytes)
            print(f"  Memory   peak {used_txt} / {limit_txt} ({mem.usage_percent:.0f}%)")
            # Only where it changes what the reader DOES. This figure is what the
            # off-node OOM guard fires on, and it can overstate (see the collector's
            # note on MaxRSS being a per-process sum), so "88% of --mem" must not be
            # the last word before someone raises --mem on a job using half of it.
            if mem.current_bytes > mem.limit_bytes:
                # Provable rather than possible: a footprint truly above the limit
                # would already have been OOM-killed, so the sum is the explanation.
                print(
                    "  note     that peak is ABOVE the limit, which a real footprint\n"
                    f"           could not be {dash} MaxRSS sums each process's RSS, counting\n"
                    "           a shared page once per process. Read it as a ceiling."
                )
            elif mem.oom_guard_warning:
                print(
                    "  note     that peak is what the memory warning fires on, and it can\n"
                    f"           overstate {dash} confirm on the node before raising --mem."
                )
        else:
            print(f"  Memory   peak {format_bytes(mem.current_bytes)}")
        print(
            f"  CPU      {_fmt_hms(cpu.usage_ns / 1e9)} CPU-time  "
            f"~{cpu.effective_cores:.1f} of {cpu.cores_allocated} cores (avg, running steps)"
        )
        # The underuse advisory belongs here too. This view is what a reader gets
        # when they CANNOT have the live dashboard — a cluster that forbids step
        # creation, or a redirect — which is precisely when "you asked for 8x what
        # you are using" is worth saying, and the number it needs is on the line
        # above. Same rule and same wording as the dashboard's insight line (SW-18).
        threshold = (config or SlurmwatchConfig()).cpu_underuse_threshold
        # NEVER off-node. sstat sees only the job's tracked process tree, so a job
        # whose work runs in detached workers (R multisession/PSOCK, nohup, setsid)
        # reads ~0.1 of 8 cores while saturating all 8 — measured 79-100x low. Adding
        # the advisory here (SW-18) would turn a silently wrong number into actively
        # wrong advice, on the one path least able to support it (SW-23).
        if not snap.remote and cpu_is_underused(cpu, threshold):
            print(
                f"  Advice   only ~{cpu.effective_cores:.1f} of {cpu.cores_allocated} cores "
                f"are doing work {dash} {CPU_UNDERUSE_ADVICE}."
            )
    elif acct_gather_disabled():
        # NOT "try again shortly": with JobAcctGatherType=none there is no sample to
        # wait for, and telling a reader to retry sends them round that loop forever.
        print(
            "  usage not sampled: this cluster's Slurm has JobAcctGatherType=none, so\n"
            "          sstat reports nothing for a running job. Run slurmwatch ON the\n"
            "          compute node (or let --once hop there) for real figures."
        )
    else:
        print(f"  usage not yet sampled by Slurm (samples ~every 30s) {dash} try again shortly")
    if job_ctx.gpu_count_requested > 0:
        print(
            f"  GPU      {job_ctx.gpu_count_requested} allocated {dash} "
            "run slurmwatch on the compute node for live GPU utilization"
        )
    # Name the risk that actually bites. The old wording ("working-set & live GPU")
    # sat directly under a CPU figure that can be 100x low and a memory figure that
    # can omit 96% of the job: sstat sees only the TRACKED process tree, so any work
    # in detached workers is invisible to it. SW-23.
    print(
        f"  source: sstat {dash} covers only the job's tracked process tree, so a job whose\n"
        "          work runs in detached workers (R multisession/PSOCK, nohup, setsid)\n"
        "          can read far lower than reality on BOTH cpu and memory. Memory can\n"
        "          also read HIGHER than reality: MaxRSS sums each process's RSS, so a\n"
        "          shared page counts once per process. Run on the node (or let --once\n"
        "          hop there) for the true figures; live GPU utilization is on-node only."
    )


def _run_remote_summary(job_ctx: JobContext, config: SlurmwatchConfig) -> None:
    collector = TelemetryCollector(job_ctx, config)

    async def _run() -> TelemetrySnapshot:
        await collector.start()
        try:
            return await asyncio.wait_for(
                collector.next_snapshot(), timeout=_FIRST_SNAPSHOT_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error("Timed out fetching remote usage for job %s", job_ctx.job_id)
            # sstat is still running on an executor thread; exit hard rather
            # than let asyncio.run's finalizer join it and hang (B-C4).
            _bounded_exit(1)
        finally:
            await collector.stop()

    snap = asyncio.run(_run())
    _print_remote_summary(job_ctx, snap, config)


# The machine format in force for this process, so the paths that die during job
# RESOLUTION can answer in it. They run before any snapshot exists and had only prose
# on stderr, which left `--once --json` emitting zero bytes for three of the four
# no-telemetry outcomes and one exit code to tell them apart (SW-27).
_MACHINE_FORMAT = ""
# ...and the dialect to write it in, for the same reason: the resolution-failure paths
# have no config in hand.
_MACHINE_CSV_DIALECT = "excel"


def _facts_csv_row(facts: dict[str, object]) -> list[str]:
    """A facts dict as CSV cells: ``None`` blanked, free text formula-guarded.

    `job_name` is arbitrary user text, and the telemetry CSV has run it through
    `_csv_text` ever since that was found — but these facts writers used a raw
    `csv.writer`, so the SAME field was guarded on one CSV surface and a live
    spreadsheet formula on the other. It is worse here than on the telemetry path: a
    FOREIGN job's name was chosen by somebody else, so the hostile case is the
    ordinary one (`sbatch -J '=cmd|"/bin/sh"!A1'` is a DDE cell, not a label).
    """
    return ["" if v is None else _csv_text(str(v)) for v in facts.values()]


def _no_telemetry_facts(
    job_id: str, token: str, prose: str, job_ctx: JobContext | None = None
) -> dict[str, object]:
    """The same key set as :func:`_foreign_facts`, for an outcome with no context.

    ONE schema across every no-telemetry outcome: a poller running `--once --json` in
    a loop gets the same object shape whether the job belongs to someone else, has
    finished, or never existed, and switches on ``telemetry_unavailable_reason``
    rather than on the wording of an error (SW-27).
    """
    if job_ctx is not None:
        facts = _foreign_facts(job_ctx)
        facts["telemetry_unavailable_reason"] = token
        facts["reason"] = prose
        return facts
    # No context at all: emit the shape with the measured AND the requested fields
    # empty, which is honest — nothing about this job was readable. The key list is
    # DERIVED from _foreign_facts against an empty context rather than duplicated, so
    # the two payloads cannot drift into different schemas.
    empty = JobContext(
        job_id=job_id,
        username="",
        partition="",
        nodelist="",
        hostname="",
        cpus_allocated=0,
        mem_limit_bytes=0,
        gpu_count_requested=0,
        gpu_indices=[],
    )
    facts = dict.fromkeys(_foreign_facts(empty), None)
    facts["timestamp"] = time.time()
    facts["job_id"] = job_id
    facts["telemetry_available"] = False
    facts["telemetry_unavailable_reason"] = token
    facts["reason"] = prose
    facts["source"] = "scontrol/squeue"
    return facts


def _pending_facts(pending: PendingJob) -> dict[str, object]:
    """A queued job as data, in the same shape every other no-telemetry outcome uses.

    PENDING is the fifth no-telemetry outcome and by far the commonest — and it was
    the one still emitting zero bytes on stdout with 528 bytes of prose on stderr, so
    a poller asking "what is my job doing" got nothing machine-readable for the state
    a job spends most of its life in. Everything below is already parsed into
    `PendingJob`; only the transport was missing (SW-27's argument, fifth case).

    The foreign schema fits without inventing keys, because its "requested, not used"
    fields are exactly what a queued job has: `cpus_allocated` and friends are the
    REQUEST here, and every measured field stays None.
    """
    facts = _no_telemetry_facts(pending.job_id, "job_pending", explain_reason(pending.reason))
    facts["job_name"] = pending.name
    facts["owner"] = pending.username
    facts["state"] = "PENDING"
    facts["partition"] = pending.partition
    facts["time_limit_seconds"] = pending.time_limit_seconds
    facts["cpus_allocated"] = pending.req_cpus
    facts["mem_limit_bytes"] = pending.req_mem_bytes or None
    facts["gpu_count_requested"] = pending.req_gpus
    facts["source"] = "scontrol/squeue"
    # The schema promises these and the foreign payload fills them, so leaving them
    # None here meant a consumer grouping a log by array_job_id silently dropped every
    # PENDING task — while the id it already had (`54222358_1`) says both.
    array = _ARRAY_ID_RE.match(pending.job_id.strip())
    if array:
        facts["array_job_id"] = array.group("base")
        facts["array_task_id"] = array.group("task")
    return facts


def _emit_no_telemetry_facts(job_id: str, token: str, prose: str) -> None:
    """Write the no-telemetry row to stdout if a machine format was requested."""
    if _MACHINE_FORMAT not in ("json", "csv"):
        return
    facts = _no_telemetry_facts(job_id, token, prose)
    if _MACHINE_FORMAT == "json":
        print(json.dumps(facts, default=str, allow_nan=False))
    else:
        # The configured dialect, like every other CSV writer here: hardcoding
        # "excel" made this one row disagree with the telemetry rows a consumer had
        # set SLURMWATCH_CSV_DIALECT for.
        writer = csv.writer(sys.stdout, dialect=_MACHINE_CSV_DIALECT)
        writer.writerow(list(facts))
        writer.writerow(_facts_csv_row(facts))
    with contextlib.suppress(BrokenPipeError):
        sys.stdout.flush()


def _foreign_facts(job_ctx: JobContext) -> dict[str, object]:
    """What is knowable about another user's job, as data rather than prose.

    The facts Slurm hands out cross-user (owner, state, placement, what was
    REQUESTED) are filled in; every field that would have to be MEASURED is
    ``None``, never 0 — "can't be measured here" and "measured zero" are different
    claims, and a consumer that averages a 0 is silently wrong. ``reason`` says why,
    so a script doesn't have to infer it. SW-8.
    """
    elapsed = (
        max(0.0, time.time() - job_ctx.job_start_time)
        if job_ctx.job_start_time is not None
        else None
    )
    counts = resolve_array_task_counts(job_ctx.array_job_id) if job_ctx.array_job_id else None
    return {
        "timestamp": time.time(),
        "job_id": job_ctx.job_id,
        "job_name": job_ctx.job_name,
        "owner": job_ctx.username,
        "state": job_ctx.job_state,
        "partition": job_ctx.partition,
        "nodelist": job_ctx.nodelist_display or job_ctx.nodelist,
        "array_job_id": job_ctx.array_job_id or None,
        "array_task_id": job_ctx.array_task_id or None,
        "array_running": counts[0] if counts else None,
        "array_pending": counts[1] if counts else None,
        "elapsed_seconds": None if elapsed is None else int(elapsed),
        "time_limit_seconds": job_ctx.time_limit_seconds,
        # Requested, not used: these come off scontrol, so they are facts.
        "cpus_allocated": job_ctx.cpus_allocated,
        "mem_limit_bytes": job_ctx.mem_limit_bytes or None,
        "gpu_count_requested": job_ctx.gpu_count_requested,
        # Measured fields, none of which Slurm will let us read for another user.
        "cpu_percent": None,
        "cpu_effective_cores": None,
        "mem_working_set_bytes": None,
        "mem_peak_bytes": None,
        "gpu_utilization_percent": None,
        "telemetry_available": False,
        # A STABLE token, not just the prose below. A poller had to match English on
        # stderr to tell "not mine, ask qhauck" from "the job is gone" — which breaks
        # the first time the wording improves, and the wording has improved twice in
        # this exercise already. SW-27.
        "telemetry_unavailable_reason": "foreign_owner",
        "reason": "another user's job — Slurm limits job-step access and sstat to the owner",
        "source": "scontrol/squeue",
    }


def _write_facts_row(
    facts: dict[str, object], config: SlurmwatchConfig, log_path: str, fmt: str
) -> None:
    """Append one facts row to ``log_path`` in the log's own format, best effort.

    Best effort on purpose: the caller is already exiting non-zero with the human
    summary on stderr, so an unwritable path here must not become a second, noisier
    failure.
    """
    use_json = _infer_use_json(fmt, log_path)
    try:
        with open(log_path, "a", newline="" if not use_json else None) as handle:
            if use_json:
                handle.write(json.dumps(facts, default=str, allow_nan=False) + "\n")
            else:
                writer = csv.writer(handle, dialect=config.csv_dialect)
                if handle.tell() == 0:
                    writer.writerow(list(facts))
                writer.writerow(_facts_csv_row(facts))
    except OSError as exc:
        logger.debug("could not write the facts row to %s (%s)", log_path, exc)


def _emit_facts_payload(facts: dict[str, object], fmt: str, config: SlurmwatchConfig) -> None:
    """Write any facts dict to stdout in the format that was asked for."""
    if fmt == "json":
        print(json.dumps(facts, default=str, allow_nan=False))
    else:
        writer = csv.writer(sys.stdout, dialect=config.csv_dialect)
        writer.writerow(list(facts))
        writer.writerow(_facts_csv_row(facts))
    with contextlib.suppress(BrokenPipeError):
        sys.stdout.flush()


# `12345_3` — one task of an array, as scontrol/squeue name it. The bracketed forms
# (`12345_[1-9%3]`) are a RANGE, not one task, so they deliberately do not match: there
# is no single task id to report for them.
_ARRAY_ID_RE = re.compile(r"^(?P<base>\d+)_(?P<task>\d+)$")


# `<job>.<step>` — the id form BOTH `sacct` and `squeue -s` print. The base must
# look like a real job id (with an optional array task / het component) so an id
# that merely contains a dot isn't rewritten into something else.
_STEP_ID_RE = re.compile(r"^(?P<job>\d+(?:_\d+)?(?:\+\d+)?)\.(?P<step>[\w.+-]+)$")


_ARRAY_RANGE_RE = re.compile(r"^(?P<base>\d+)_\[(?P<range>[\d,\-%]+)\]$")


def _job_id_without_array_range(job_id: str) -> str:
    """``54222358_[1-9%3]`` → ``54222358``, saying so on stderr; anything else unchanged.

    This is the id ``squeue`` PRINTS for a pending array — one row for the whole
    unstarted range, with an optional ``%N`` concurrency throttle — and pasting what
    squeue printed is the entire way anyone arrives here. It was rejected as "not a
    job id", and the advice attached to that refusal was to go find the id with
    ``squeue -o '%i %j'``, which prints the very same string: a closed loop.

    A range names no single task, and an unstarted array has no per-task telemetry
    anyway, so the useful target is the array's own job — whose pending reason,
    request and queue position are exactly what the reader was asking about. Fourth
    instance of the family SW-7, RD-2 and SW-14 belong to: a lookup that failed for a
    reason that was not true, throwing away the real explanation.
    """
    match = _ARRAY_RANGE_RE.match(job_id)
    if match is None:
        return job_id
    base = match.group("base")
    print(
        f"slurmwatch: {job_id} names an array range; monitoring array job {base} "
        f"(pass {base}_<task> for one task once it starts)",
        file=sys.stderr,
    )
    return base


def _job_id_without_step(job_id: str) -> str:
    """``48819348.0`` → ``48819348``, saying so on stderr; anything else unchanged.

    A step id is exactly what someone pastes, because `sacct -j 48819348.0` and
    `squeue -s` are where they read it — and `scontrol` doesn't accept the form, so
    resolution failed with "Job 48819348.0 does not exist in the Slurm database".
    That is a false statement about the accounting database (the step is right
    there in `sacct`), and it sends the reader off to check whether their job was
    purged or their id was wrong.

    slurmwatch is a job-level monitor: it reports the whole job's footprint across
    all its steps, and the job-level cgroup follows them live. So attach to the job
    the step belongs to — the thing the reader was after — and say that is what
    happened, rather than refusing with a reason that isn't true. Third instance of
    this family, after SW-7 and rapidu's RD-2: a failed lookup whose real
    explanation was thrown away. SW-14.
    """
    match = _STEP_ID_RE.match(job_id)
    if match is None:
        return job_id
    base = match.group("job")
    print(
        f"slurmwatch: {job_id} names one step; monitoring job {base} "
        f"(the whole job's footprint, across all its steps)",
        file=sys.stderr,
    )
    return base


def _write_record(fd: int, payload: bytes) -> None:
    """One record, one ``write()`` — the atomicity the log format depends on.

    Its own function so the sink can be wedged in a test (a full pipe whose reader
    stopped, a hung NFS mount) without patching ``os.write`` process-wide, and so
    there is exactly one place that must stay a single syscall. SW-16.
    """
    os.write(fd, payload)


def _claim_log_file(fd: int, log_path: str, append: bool = False) -> None:
    """Claim a ``--log`` target: refuse a second TRUNCATING writer, warn on a second
    appender.

    Two telemetry writers on one path is nearly always a logger forgotten in
    another terminal, and the cost lands much later: a file that parses fine at the
    start and raises ``JSONDecodeError`` partway through. Saying so now is cheaper
    than a silently corrupted dataset. SW-16.

    The refusal is narrowed to the case that cannot be made right, on the same rule
    SW-25 settled: refuse when correctness is unreachable, allow when it is not. A
    truncating open destroys the other writer's data outright — nothing recovers that,
    so it is refused. ``--append`` is a different matter: records are single atomic
    ``write()`` calls, so rows stay whole, and the reporter measured exactly this —
    two writers, one path, 1 Hz, "31 data rows from two writers, one header, 0 rows
    with a wrong field count", interleaved rather than clobbering. Refusing that
    forbids a genuinely useful pattern (one aggregate log for several jobs, told apart
    by ``job_id``), so it proceeds — but it says so, because the forgotten-logger case
    is the common one and silence is what made it expensive.

    Only a REGULAR file is claimed. ``--log /dev/stdout`` is how the node switcher
    streams a remote node's snapshots back (see remote.py), and a pipe/tty/FIFO has
    no shared file offset to protect. A filesystem that can't lock (some NFS mounts)
    is not a reason to refuse to log either.

    The lock is a POSIX RECORD lock (``lockf``/``F_SETLK``), not ``flock``, because
    the guarantee has to hold where ``--log`` paths actually live: a shared
    filesystem. Measured on this cluster's GPFS ``/home`` between two nodes of the
    same user's allocation — a lock held on one node was granted AGAIN on the other
    with ``flock``, and correctly refused with ``lockf``:

        flock:  held on midway3-0200  ->  GRANTED on beagle3-0009   (no exclusion)
        lockf:  held on midway3-0200  ->  refused (EAGAIN)          (excludes)

    So the refusal this function exists to print was node-local, silently, on exactly
    the cross-cluster shared-home setup the portability exercise is about. ``flock``
    remains the fallback for a filesystem whose ``F_SETLK`` is unsupported, so nothing
    that works today stops working. One behaviour difference to know: record locks are
    per-PROCESS, so a second claim from within the same process is granted (``flock``
    would refuse it) — irrelevant here, because the case being guarded is two
    slurmwatch processes.
    """
    busy = (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
    except OSError as exc:
        logger.debug("could not stat %s (%s); logging anyway", log_path, exc)
        return
    # BOTH mechanisms, because they exclude different things and neither is a superset:
    # lockf reaches across nodes (above), flock reaches across open file descriptions
    # within one node — including two opens in one process, which lockf grants because
    # record locks are per-PROCESS. Either answering "busy" is a second writer.
    held = False
    for acquire in (fcntl.lockf, fcntl.flock):
        try:
            acquire(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except OSError as exc:
            if exc.errno in busy:
                if append:
                    logger.warning(
                        "another slurmwatch is already logging to %s — appending "
                        "alongside it. Rows stay whole (one atomic write each) and "
                        "carry their own job_id, but this file will hold both "
                        "streams; use a different --log path to keep them apart.",
                        log_path,
                    )
                    return
                logger.error(
                    "another slurmwatch is already logging to %s and this run would "
                    "TRUNCATE it, destroying its data; use a different --log path, or "
                    "--append to write alongside it.",
                    log_path,
                )
                sys.exit(1)
            # Unsupported on this filesystem: try the other mechanism rather than
            # refusing to log at all.
            logger.debug("%s unavailable on %s (%s)", acquire.__name__, log_path, exc)
    if not held:
        logger.debug("could not lock %s with either mechanism; logging anyway", log_path)


def _apply_sampling_floor(config: SlurmwatchConfig, remote: bool) -> None:
    """Raise a pathologically small interval to this sampling path's floor, and SAY so.

    ``--interval 0.001`` is one misplaced character from ``0.1`` and used to be
    accepted in silence: the loop ran at the old 0.05s floor — ~19 samples a second
    — and the cost (a --log file growing a thousand times too fast, a cgroup or
    sstat read that often) landed somewhere other than on the person who typed it.
    Off the node each sample is a subprocess plus a slurmdbd query, so the floor
    there is an order of magnitude higher. Announced on stderr rather than applied
    quietly, because a tool that ignores a flag without a word is the actual
    failure. SW-13.
    """
    floor = MIN_REMOTE_INTERVAL if remote else MIN_INTERVAL
    # Judge against what was TYPED, not against the already-clamped value: `clamp()`
    # runs on the override right after parsing, so an on-node `--interval 0.001` was
    # raised to 0.1 before reaching here and this function saw nothing to announce —
    # silence being the one outcome SW-13 exists to prevent. Off-node it was worse
    # than silent: the notice quoted its own clamped 0.1 back at a user who typed
    # 0.001.
    asked = (
        config.requested_interval
        if config.requested_interval is not None
        else min(config.poll_interval, config.headless_interval)
    )
    # Announce a floor only where it overrode something the user ASKED for. The
    # default poll interval (0.5) sits below the sstat floor, so keying purely on
    # "did the number move" made every off-node run open with
    # "slurmwatch: --interval 0.5 raised to 1s" — naming a flag the reader never
    # typed, about a default they never chose. SW-13's rule is that a TYPED value is
    # never silently ignored, which this keeps: --interval and
    # SLURMWATCH_POLL_INTERVAL both set requested_interval.
    raised = asked < floor and config.requested_interval is not None
    config.poll_interval = max(config.poll_interval, floor)
    config.headless_interval = max(config.headless_interval, floor)
    if raised:
        print(
            f"slurmwatch: --interval {asked:g} raised to {floor:g}s "
            f"({'sstat' if remote else 'cgroup'} sampling floor)",
            file=sys.stderr,
        )


def _run_foreign_summary(job_ctx: JobContext, config: SlurmwatchConfig, stream: Any = None) -> None:
    """Read-only summary for a job owned by *another* user.

    Slurm restricts job-step creation and ``sstat`` usage data to a job's owner
    (or root), so slurmwatch can neither attach a live dashboard nor read live
    CPU/mem/GPU for someone else's job from a login node (see
    :func:`_job_owner_differs`). Rather than attempt the doomed ``srun`` hop —
    which just leaks ``srun``'s "Access/permission denied" and then a misleading
    "usage not yet sampled" line — show the ``scontrol``/``squeue`` facts we *can*
    see cross-user and say plainly why there's no live telemetry.
    """
    out = stream if stream is not None else sys.stdout
    ascii_mode = config.ascii_mode
    dash = "-" if ascii_mode else "—"
    sep_ch = "-" if ascii_mode else "·"
    node = job_ctx.nodelist_display or "?"
    state = job_ctx.job_state or ""
    owner = job_ctx.username or "another user"
    print(
        f"Job {job_ctx.job_id}  {job_ctx.partition}  {state}  on {node}  (owner: {owner})"
        f"{_name_suffix(job_ctx.job_name)}",
        file=out,
    )

    if job_ctx.array_job_id:
        line = f"  Array    {job_ctx.array_job_id} {dash} task {job_ctx.array_task_id}"
        counts = resolve_array_task_counts(job_ctx.array_job_id)
        if counts is not None:
            running, pending = counts
            line += f" ({running} running, {pending} pending)"
        print(line, file=out)

    if job_ctx.job_start_time is not None:
        elapsed = max(0.0, time.time() - job_ctx.job_start_time)
        if job_ctx.time_limit_seconds is not None:
            print(
                f"  Time     running {_fmt_hms(elapsed)} of up to "
                f"{_fmt_hms(job_ctx.time_limit_seconds)} (wall-clock limit)",
                file=out,
            )
        else:
            print(f"  Time     running {_fmt_hms(elapsed)} (no wall-clock limit)", file=out)
    # WHAT IT ASKED FOR. Nothing about a foreign job can be measured, so the request
    # is the only resource fact there is — and it is the question the reader has when
    # they run this on someone else's job ("what is this node busy with, and how
    # much of it did they take?"). The GPU count alone used to be printed here while
    # the cores and the memory, sitting in the same job_ctx, were dropped; the TUI's
    # foreign card had shown all of them all along.
    alloc: list[str] = []
    n_nodes = len(job_ctx.nodelist_resolved) or 1
    per = per_node_suffix(n_nodes)
    alloc.append(f"{n_nodes} node" if n_nodes == 1 else f"{n_nodes} nodes")
    if job_ctx.cpus_allocated:
        alloc.append(f"{job_ctx.cpus_allocated} CPU{per}")
    if job_ctx.mem_limit_bytes > 0:
        alloc.append(f"{format_bytes(job_ctx.mem_limit_bytes)}{per}")
    if job_ctx.gpu_count_requested > 0:
        alloc.append(f"{job_ctx.gpu_count_requested}x GPU")
    print(f"  Request  {f'  {sep_ch}  '.join(alloc)}", file=out)

    print(
        f"  live CPU / memory / GPU usage isn't available for another user's job {dash} Slurm",
        file=out,
    )
    print(
        f"  limits job-step access and sstat to the job's owner. Ask {owner} to run slurmwatch",
        file=out,
    )
    print("  on the node, or watch one of your own jobs.", file=out)
    print(
        f"  source: scontrol/squeue (facts only {dash} no live telemetry across users)",
        file=out,
    )


def _run_foreign(job_ctx: JobContext, config: SlurmwatchConfig, args: argparse.Namespace) -> None:
    """Show another user's job: the styled read-only TUI on a real terminal, else text.

    Mirrors :func:`_run_pending` — a queued job and someone else's job are both
    "facts, no live telemetry" views, so both get a Textual screen interactively
    and fall back to the plain-text summary when piped/redirected (or if the TUI
    can't start).
    """
    interactive = not (args.once or args.log) and sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        _run_foreign_summary(job_ctx, config)
        return
    with _console_logging_suspended():
        try:
            from .tui import ForeignJobApp

            app = ForeignJobApp(job_ctx, config)
            app.run(mouse=_mouse_enabled(config))
            return
        except Exception as exc:
            logger.error("TUI error: %s", exc)
    _run_foreign_summary(job_ctx, config)


# How long to let ``srun`` try to create the monitor step before giving up. A
# normal attach reserves its step in well under a second; a longer wait means a
# scarce resource in the allocation — classically the GPU — is fully held by the
# job's *own* step, and since ``--overlap`` shares CPUs but NOT GRES, a second
# step can never get it. Without a bound srun retries that forever (the login
# node just shows a frozen "connecting …"); ``--immediate`` turns the dead wait
# into a clean fall-through to the remote summary. Overridable for slow clusters.
_HOP_CONNECT_TIMEOUT_DEFAULT = 10


def _hop_connect_timeout() -> int:
    """Seconds to give the srun hop before falling back; env-overridable, clamped."""
    raw = os.environ.get("SLURMWATCH_HOP_TIMEOUT")
    if raw is None:
        return _HOP_CONNECT_TIMEOUT_DEFAULT
    try:
        # int(float("inf")) raises OverflowError (and NaN raises ValueError); a
        # bogus SLURMWATCH_HOP_TIMEOUT must fall back to the default, not crash.
        val = int(float(raw))
    except (ValueError, OverflowError):
        warn_unusable_env(
            "SLURMWATCH_HOP_TIMEOUT",
            raw,
            "not a number",
            f"the {_HOP_CONNECT_TIMEOUT_DEFAULT}s default",
        )
        return _HOP_CONNECT_TIMEOUT_DEFAULT
    clamped = max(2, min(val, 120))
    if clamped != val:
        warn_unusable_env("SLURMWATCH_HOP_TIMEOUT", raw, "outside the 2-120s range", f"{clamped}s")
    return clamped


# How long the GPU-attachability probe waits. A GPU a monitor step can actually
# get yields a step in ~1s; contention (the GPU held by the job's own step) makes
# step creation retry until it's bounded. 6s is generous headroom for a real
# attach while keeping the "can't get it" case quick, capped by the hop timeout.
_GPU_PROBE_SECONDS = 6

# Braille spinner frames for the connect animation (ASCII fallback under --ascii).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spin_loading(stop: threading.Event, node: str, ascii_mode: bool) -> None:
    """Animate a 'loading' line on stderr until ``stop`` is set.

    Used only during the silent connect probe — the probe's output is suppressed,
    so the terminal is ours until the ``--pty`` session takes the screen; the
    caller clears this line before that happens. (A live spinner during the
    ``--pty`` session itself would repaint over the attached dashboard, which is
    why this is scoped to the probe.)
    """
    frames = "|/-\\" if ascii_mode else _SPINNER_FRAMES
    dots = "..." if ascii_mode else "…"
    i = 0
    while not stop.is_set():
        sys.stderr.write(f"\r\033[K{frames[i % len(frames)]} loading {node} {dots}")
        sys.stderr.flush()
        i += 1
        stop.wait(0.1)


def _srun_can_get_gpu(
    srun: str, raw_id: str, node: str, timeout: int, child_env: dict[str, str]
) -> bool:
    """Quietly test whether an ``--overlap`` step can obtain the job's GPU(s).

    Runs a throwaway ``true`` step with output suppressed, so srun's "Requested
    nodes are busy" (when the GPU is held by the job's own step) never reaches the
    user. Success ⇒ the real attach can request the GPU and show live util; failure
    ⇒ attach without one so the dashboard still opens on CPU/mem.

    ``--mem=0`` (share the job's memory, reserve none) isolates the test to GPU
    contention: without it a job that also holds all its memory would fail the
    probe for the wrong reason and needlessly drop the GPU.
    """
    probe = [
        srun,
        f"--jobid={raw_id}",
        "--overlap",
        f"--immediate={min(timeout, _GPU_PROBE_SECONDS)}",
        "--mem=0",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        "true",
    ]
    try:
        result = subprocess.run(
            probe,
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # --immediate only bounds srun's resource-wait, NOT the initial
            # slurmctld RPC: a wedged/slow controller can ignore it and hang the
            # probe indefinitely, freezing `sw <jobid>` behind the loading spinner.
            # Bound it in Python too (kills the child on expiry). "Couldn't decide
            # in time" == "can't get the GPU" → the hop still attaches with
            # --gres=none, mirroring the TUI's wait_for(25s) fallback.
            timeout=min(timeout, _GPU_PROBE_SECONDS) + 3,
        )
    except subprocess.TimeoutExpired:
        logger.debug("gpu probe timed out; treating as 'no GPU'")
        return False
    except KeyboardInterrupt:
        sys.exit(130)
    except OSError as exc:
        logger.debug("gpu probe did not run: %s", exc)
        return False
    return result.returncode == 0


# What the srun hop did, so the caller can tell "the user (or this environment)
# said no" from "we tried and it didn't work". They deserve different follow-ups:
# the ssh login rung is far more invasive than a step — on some sites each
# interactive login leaks threads into the job's .extern stepd that are never freed
# (SW-21) — so it must not be taken on behalf of a user who opted OUT of the
# gentler transport.
_HOP_RAN = "ran"
_HOP_DECLINED_POLICY = "declined"
_HOP_NO_SRUN = "no-srun"
_HOP_FAILED = "failed"


# Leave the alt-screen, show the cursor, end synchronized-update/bracketed paste,
# reset colours. The --pty session held STDOUT (the hop requires a tty there; stderr
# may be redirected), so prefer stdout — writing to a redirected stderr would leave
# the real terminal garbled. Idempotent: sending it when the terminal is already
# restored costs nothing, which is why no caller guards on "was it needed".
# How long the outer process waits for the --pty child to tear its own screen down
# after a forwarded signal. Short: the child handles SIGTERM and Textual's teardown is
# fast, and a user who sent a signal is waiting on their prompt.
_PTY_CHILD_SIGNAL_GRACE_SECONDS = 2.0

_TERMINAL_RESET = "\033[?1049l\033[?25h\033[?2026l\033[?2004l\033[0m\r"


def _restore_terminal() -> None:
    """Undo the inner ``--pty`` TUI's terminal state from the OUTER process.

    The inner process installs its own SIGTERM handler and Textual restores the
    screen as it exits — measured clean on-node (SIGINT/SIGTERM/q all leave ECHO and
    ICANON set). But a signal a user sends goes to the OUTER process in their shell,
    which had no equivalent: it left through ``sys.exit(130)`` or the ``returncode in
    (0, 130)`` shortcut, both ABOVE the reset that already existed, so the terminal
    stayed in the alternate screen with ECHO and ICANON cleared — no echo, no line
    editing, ``reset``/``stty sane`` to recover. Reachable by ``kill``/``kill -INT``
    from another shell, SIGHUP when a tmux pane or IDE terminal dies, Ctrl-C during
    the startup window before ``--pty`` owns the screen, and any wrapper that signals
    its children. SW-26.
    """
    stream = sys.stdout if sys.stdout.isatty() else (sys.stderr if sys.stderr.isatty() else None)
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.write(_TERMINAL_RESET)
        stream.flush()


class _TerminalGuard:
    """Restore the terminal on a SIGTERM/SIGHUP to the OUTER hop process.

    Installed for the WHOLE hop, not just around the ``--pty`` child: measured in a
    real pty, a SIGHUP 9 s in still killed the process outright, because that landed
    in the startup window — the `_srun_can_get_gpu` probe with its own multi-second
    timeout — before any handler existed. The reporter's accounting names that window
    explicitly, and it is the same window a user's Ctrl-C can reach.

    While a child exists the signal is forwarded to it first and reaped with a bounded
    grace (it handles SIGTERM and tears its own screen down); before that there is
    nothing to forward to, so we just restore and leave. Previous handlers are put
    back on exit, because a hop that could not start falls through to the sstat path
    in this same process.
    """

    def __init__(self) -> None:
        self.child: subprocess.Popen[bytes] | None = None
        self._previous: list[tuple[int, Any]] = []

    def __enter__(self) -> _TerminalGuard:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with contextlib.suppress(ValueError, OSError, AttributeError):
                self._previous.append((signum, signal.signal(signum, self._on_signal)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signum, handler)
        self._previous.clear()

    def spawn(self, cmd: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
        """Start the child with the guarded signals BLOCKED until it is registered.

        `Popen(...)` returning and `self.child = ...` are two statements, and a signal
        delivered between them ran the handler with no child to forward to: the parent
        restored the terminal and exited while the freshly-spawned srun/ssh kept the
        tty and the step. Small window, but it is the exact harm this guard exists to
        prevent. Blocking makes the signal pending instead; it fires on unblock, by
        which time there is a child to reap.
        """
        blocked = {signal.SIGTERM, signal.SIGHUP}
        previous: set[int] | None = None
        with contextlib.suppress(AttributeError, OSError, ValueError):
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            self.child = subprocess.Popen(cmd, env=env)
            return self.child
        finally:
            if previous is not None:
                with contextlib.suppress(OSError, ValueError):
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def _on_signal(self, signum: int, _frame: object) -> None:
        child = self.child
        if child is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                child.send_signal(signum)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=_PTY_CHILD_SIGNAL_GRACE_SECONDS)
        _restore_terminal()
        # os._exit: the interpreter is mid-signal with a torn-down screen, and a
        # normal exit would run handlers that write to it.
        os._exit(128 + signum)


def _run_pty_child(
    cmd: list[str], child_env: dict[str, str], guard: _TerminalGuard | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run the ``srun --pty`` child, letting ``guard`` forward a signal to it."""
    # Registered by the guard itself, with the signals blocked across the spawn, so
    # there is no window in which a signal exits and leaves the child running.
    proc = (
        guard.spawn(cmd, child_env) if guard is not None else subprocess.Popen(cmd, env=child_env)
    )
    try:
        rc = proc.wait()
    finally:
        if guard is not None:
            guard.child = None
    return subprocess.CompletedProcess(cmd, rc)


def _hop_to_compute_node(job_ctx: JobContext, args: argparse.Namespace) -> str:
    """Re-launch the live TUI on the job's compute node via ``srun --overlap``.

    From a login node the cgroups aren't reachable, so instead of degrading to a
    text summary we attach to the job's allocation and run the full dashboard
    where the data actually lives — and the UI comes up in *all* cases:

    - First quietly probe whether a monitor step can get the job's GPU(s). If it
      can (the common case — a GPU held by the ``.batch`` step still lets a new
      overlapping step read it), attach requesting the GPU so live GPU util shows.
    - If it can't (the GPU is held by the job's *own* step; ``--overlap`` shares
      CPUs, NOT GRES on this Slurm), attach with ``--gres=none`` instead — that
      step always attaches, so the full live dashboard (CPU/mem/processes) still
      opens; the GPU panel just notes it isn't readable from a monitor step.

    The probe is silent, so no "busy"/"held by your step" noise ever surfaces.
    Returns one of the ``_HOP_*`` outcomes; ``_HOP_RAN`` means a dashboard ran and
    the caller should exit.
    """
    # SLURMWATCH_NO_HOP is set on the relaunched process (belt-and-suspenders
    # against any loop) and lets a user opt out of the behavior entirely.
    if _env_says_already_on_node():
        return _HOP_DECLINED_POLICY
    if _env_disables_hop():
        # A perfectly VALID opt-out, honoured silently until now — and silence is
        # the problem: an exported SLURMWATCH_NO_HOP (from an earlier experiment, a
        # site module file, a .bashrc carried between clusters) leaves the dashboard
        # on sstat-quality data while a working transport sits unused, with nothing
        # on screen or in the log to say so. Name it once (SW-20/round 25).
        logger.warning(
            "SLURMWATCH_NO_HOP is set, so slurmwatch will not attach to the job's "
            "node; expect coarser (sstat) numbers than an on-node run."
        )
        return _HOP_DECLINED_POLICY
    # A TUI needs a terminal; when piped/redirected the summary is more useful.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _HOP_DECLINED_POLICY
    srun = shutil.which("srun")
    if srun is None:
        return _HOP_NO_SRUN
    node = job_ctx.nodelist_resolved[0] if job_ctx.nodelist_resolved else None
    if not node:
        return _HOP_NO_SRUN

    # Everything below can be interrupted, and from here on the terminal is at
    # stake: the guard covers the GPU probe too, because a SIGHUP measured 9 s in
    # still killed the process outright when the handlers only wrapped the --pty
    # child (SW-26).
    with _TerminalGuard() as guard:
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
        timeout = _hop_connect_timeout()

        # Start from our env but drop the surrounding allocation's SLURM_* sizing
        # vars (e.g. if launched from inside another salloc) so the step's request
        # comes only from the explicit flags; keep SLURM_CONF, which srun needs.
        child_env = {
            k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"
        }
        child_env["SLURMWATCH_NO_HOP"] = "1"
        child_env["SLURMWATCH_ON_NODE"] = "1"
        # Mark the relaunched process as the hop's monitor step so its dashboard warns
        # that it's holding a job step (which blocks a NEW srun/mpirun the user starts)
        # and runs the best-effort stalled-launch detector (see tui.MonitorNote).
        child_env["SLURMWATCH_MONITOR_STEP"] = "1"

        # Decide the GPU flags quietly (request the GPU only when a monitor step can
        # actually get it, else attach without one so the dashboard still opens),
        # animating a "loading" spinner while the probe runs. The probe's output is
        # suppressed, so the terminal is ours to animate on until the --pty session
        # starts; stop and erase the line before that takes the screen.
        # Only animate when stderr is a real terminal — the spinner writes cursor/
        # erase control codes (\r\033[K), which would garble a redirected `2>err.log`.
        animate = sys.stderr.isatty()
        stop = threading.Event()
        spinner: threading.Thread | None = None
        if animate:
            spinner = threading.Thread(
                target=_spin_loading, args=(stop, node, args.ascii), daemon=True
            )
            spinner.start()
        try:
            gpu_ok = _srun_can_get_gpu(srun, raw_id, node, timeout, child_env)
        finally:
            stop.set()
            if spinner is not None:
                spinner.join(timeout=1.0)
                sys.stderr.write("\r\033[K")  # erase the spinner line
                sys.stderr.flush()
        # A monitor step that cannot get the GPUs would attach BLIND to them — and on a
        # multi-node training job that is the normal case, not an edge case: the job
        # launches one torchrun per node through an inner `srun`, that step holds every
        # GPU, and this Slurm cannot share GRES between steps. A GPU monitor that shows
        # no GPU numbers for the most common way of running multi-node training is not
        # worth much, so before settling for a blind step, try ssh.
        #
        # ssh works where the step cannot: an adopted session (pam_slurm_adopt +
        # PrologFlags=Contain) is placed in the job's cgroups for cpuset/memory
        # accounting — so CPU and memory still read the job, verified identical to the
        # step transport — while the DEVICES controller leaves it in /user.slice,
        # unrestricted. NVML therefore enumerates the job's GPUs and full utilization,
        # VRAM and power appear. This costs exactly one login for the whole session
        # (the dashboard is a single long-lived process), which matters because each
        # login leaks threads into the job's .extern stepd that are never freed.
        if not gpu_ok and (job_ctx.gpu_count_requested or job_ctx.gpu_indices):
            if _ssh_to_compute_node(job_ctx, args):
                return _HOP_RAN
            # ssh unavailable/not permitted here — fall through to the blind step, which
            # still gives live CPU/memory and an honest "why" for the GPU row.
            logger.debug("ssh GPU transport unavailable; attaching a blind monitor step")
        # --overlap shares CPUs, --mem=0 reserves no memory, --gres=none (when the GPU
        # is held) requests no GPU, and --immediate bounds it: together these make the
        # monitor step launchable on *any* live allocation (sbatch / srun / salloc /
        # sinteractive / het / array) — no resource it could contend for can block it.
        gpu_flags: tuple[str, ...] = () if gpu_ok else ("--gres=none",)
        cmd = [
            srun,
            f"--jobid={raw_id}",
            "--overlap",
            f"--immediate={timeout}",
            "--mem=0",
            *gpu_flags,
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={node}",
            "--pty",
            *inner,
        ]
        try:
            result = _run_pty_child(cmd, child_env, guard)
        except KeyboardInterrupt:
            # SIGINT to the OUTER process (a `kill -INT`, or Ctrl-C during the startup
            # window before --pty owns the screen) surfaces here. Restore before leaving:
            # this exit sat ABOVE the reset block below. SW-26.
            _restore_terminal()
            sys.exit(130)
        except OSError as exc:
            logger.debug("srun hop did not run: %s", exc)
            return _HOP_FAILED
        # rc 0 = clean quit, 130 = Ctrl-C inside the TUI: either way the dashboard ran.
        if result.returncode in (0, 130):
            # Restore on 130 too: the shortcut assumed the inner TUI had tidied up on its
            # way out, which is exactly the assumption the reset below exists because it
            # does not always hold. Harmless on a clean rc 0 (SW-26).
            _restore_terminal()
            return _HOP_RAN
        # Abnormal exit. If the step was signal-killed (scancel SIGTERMs it, rc 143; a
        # SIGKILL is 137) the inner --pty TUI may not have restored the terminal, so
        # reset it (leave alt-screen, show cursor, end synchronized-update/bracketed
        # paste, reset colors) before writing anything — otherwise our text lands in a
        # garbled screen. The --pty session held STDOUT (the hop requires stdout to be
        # a tty; stderr may be redirected), so send the reset there — writing it to a
        # redirected stderr would leave the real terminal garbled.
        _restore_terminal()
        # A signal-terminated step means the job was cancelled/timed-out/preempted or
        # the node went away (143 = SIGTERM from scancel/timeout, 137 = SIGKILL) — not
        # a dashboard crash. Right after a cancel the job is briefly COMPLETING (which
        # `is_job_active` counts as alive), so key off the signal code, not squeue: exit
        # cleanly instead of dumping a stale "RUNNING" summary on the torn-down screen.
        if result.returncode in (137, 143):
            print(
                f"slurmwatch: monitoring stopped — job {job_ctx.job_id} was cancelled or ended.",
                file=sys.stderr,
            )
            return _HOP_RAN
        # Otherwise: if squeue confirms the job is gone, say so cleanly; else the
        # session failed while the job is alive (e.g. the on-node collector crashed) —
        # fall back to the remote summary.
        if is_job_active(raw_id) is False:
            print(f"slurmwatch: job {job_ctx.job_id} has ended.", file=sys.stderr)
            return _HOP_RAN
        # Job is still alive but the attach failed — either srun couldn't create the
        # monitor step within --immediate (resources busy) or the on-node dashboard
        # exited. Word it for both rather than claiming the dashboard "exited".
        print(
            f"slurmwatch: couldn't run the live dashboard on {node} (rc={result.returncode}); "
            "showing the remote summary instead.",
            file=sys.stderr,
        )
        return _HOP_FAILED


def _env_says_already_on_node() -> bool:
    """Whether this process is a hop's relaunched copy, already on the job's node.

    Recursion prevention and the user's "never use ssh" preference used to share
    ``SLURMWATCH_NO_SSH``: the ssh hop set it on the child purely so the child
    wouldn't ssh to itself. That silently disabled the ssh transport the node
    SWITCHER needs — a different node, so no recursion is possible — leaving the
    on-node dashboard unable to read any other node's GPUs. Kept separate so
    ``SLURMWATCH_NO_SSH`` means only what the user meant by it.
    """
    val = os.environ.get("SLURMWATCH_ON_NODE")
    return val is not None and val.strip().lower() not in ("", "0", "false", "no", "off")


def _ssh_to_compute_node(job_ctx: JobContext, args: argparse.Namespace) -> bool:
    """Fallback transport: SSH to the job's node and run the live TUI there.

    When the ``srun`` hop can't create a monitor step — nested-srun hangs on GPU
    nodes, GRES/step-creation policy, or a site that rejects ``--gres`` — most
    Slurm clusters still let a user SSH into a node where they have a running job
    and adopt that session into the job's cgroup (``pam_slurm_adm`` +
    ``PrologFlags=Contain``). slurmwatch's own ``/proc/self/cgroup`` discovery
    then finds the cgroup, so the full on-node dashboard renders with **no Slurm
    step at all** — the universal off-node path that sidesteps every srun-step
    limitation. Returns ``True`` if a dashboard ran (caller should exit).

    Kept a strict *fallback* to the srun hop (not a replacement): the hop is the
    Slurm-native transport and works on most sites; ssh covers the ones where a
    step can't be created but node login is permitted.
    """
    if _env_disables_ssh() or _env_says_already_on_node():
        return False
    # A TUI needs a real terminal on both ends; when piped/redirected the sstat
    # summary is the better output.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    ssh = shutil.which("ssh")
    if ssh is None:
        return False
    node = job_ctx.nodelist_resolved[0] if job_ctx.nodelist_resolved else None
    if not node:
        return False

    # Run THIS interpreter's slurmwatch by absolute path (resolves over the shared
    # filesystem, no dependency on the node's PATH), with hop AND ssh disabled on
    # the relaunched process so it can never loop back out.
    inner = [sys.executable, "-m", "slurmwatch", job_ctx.job_id]
    if args.ascii:
        inner.append("--ascii")
    if args.interval is not None:
        inner += ["--interval", str(args.interval)]
    # `ssh host cmd` runs a NON-login shell that doesn't source /etc/profile.d, so
    # module-provided Slurm binaries aren't on PATH and a configless controller
    # isn't reachable — carry PATH and SLURM_CONF through (the srun hop preserves
    # both via child_env). Use `env VAR=val` (external command), not the inline
    # `VAR=val cmd` form, which csh/tcsh/fish don't parse (F1/F2).
    env_prefix = ["env"]
    for var in ("PATH", "SLURM_CONF"):
        val = os.environ.get(var)
        if val:
            env_prefix.append(f"{var}={val}")
    env_prefix += ["SLURMWATCH_NO_HOP=1", "SLURMWATCH_ON_NODE=1"]
    remote = " ".join(shlex.quote(tok) for tok in [*env_prefix, *inner])
    cmd = [
        ssh,
        "-t",  # allocate a remote tty so the TUI can render
        "-o",
        "BatchMode=yes",  # never block on a password prompt — fail fast if not permitted
        "-o",
        "ConnectTimeout=10",
        node,
        remote,
    ]
    # Same guard as the srun hop: this is the OTHER transport, and it had the same
    # three unguarded exits. Measured on midway3, which prefers ssh, a SIGHUP left the
    # terminal in the alternate screen with ECHO/ICANON cleared while SIGTERM (handled
    # by the inner TUI, rc 143) was already clean — so fixing only the srun path would
    # have left the defect live on every cluster that takes this one. SW-26.
    with _TerminalGuard() as guard:
        try:
            result = _run_pty_child(cmd, dict(os.environ), guard)
        except KeyboardInterrupt:
            _restore_terminal()
            sys.exit(130)
        except OSError as exc:
            logger.debug("ssh hop did not run: %s", exc)
            return False
    # rc 0 = clean quit, 130 = Ctrl-C inside the TUI: either way the dashboard ran.
    if result.returncode in (0, 130):
        _restore_terminal()
        return True
    # 255 = ssh transport failure (host unreachable, login not permitted, no
    # key/host-based auth): fall through to the remote sstat summary.
    if result.returncode == 255:
        logger.debug("ssh to %s failed (rc=255); falling back to remote summary", node)
        return False
    # A signal-killed remote TUI (scancel/timeout/preempt: 143=SIGTERM, 137=SIGKILL)
    # may not have restored the terminal; reset it (leave alt-screen, show cursor,
    # end sync-update/bracketed-paste, reset colours) before printing, then say so
    # cleanly — mirrors the srun hop so a torn-down screen isn't left garbled (F4).
    if result.returncode in (137, 143):
        _restore_terminal()
        print(
            f"slurmwatch: monitoring stopped — job {job_ctx.job_id} was cancelled or ended.",
            file=sys.stderr,
        )
        return True
    # ssh connected but the remote slurmwatch exited nonzero. If the job has ended,
    # say so cleanly; otherwise let the caller fall back to the summary.
    if is_job_active(job_ctx.raw_job_id or job_ctx.job_id) is False:
        print(f"slurmwatch: job {job_ctx.job_id} has ended.", file=sys.stderr)
        return True
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


def _print_pending_summary(
    pending: PendingJob, stream: Any = None, ascii_mode: bool = False
) -> None:
    """Plain-text 'why / when / where' report for a PENDING job (non-TUI paths)."""
    out = stream if stream is not None else sys.stdout
    # Honour --ascii here too (a non-UTF-8 terminal / pipe): no stray Unicode.
    dash = "-" if ascii_mode else "—"
    dot = "-" if ascii_mode else "·"
    dots = "..." if ascii_mode else "…"

    def emit(line: str) -> None:
        print(line, file=out)

    now = time.time()
    emit(f"Job {pending.job_id}  {pending.partition}  PENDING{_name_suffix(pending.name)}")
    reason = pending.reason or "None"
    emit(f"  Why    {reason} {dash} {explain_reason(pending.reason, ascii_mode)}")
    held = is_held_like(pending.reason)
    est = pending.start_time_estimate
    if est is not None and est >= now - 1:
        rel = _fmt_wait(int(est - now))
        # Absolute date (not weekday-only) so an estimate >6 days out isn't ambiguous.
        when = time.strftime("%b %d %H:%M", time.localtime(est))
        emit(f"  When   estimated start {when} (in ~{rel}; scheduler estimate, may change)")
    elif est is not None and est >= now - 900:
        # Backfill stamps StartTime at its last cycle → a few min in the past means
        # imminent, not "no estimate".
        emit("  When   estimated start imminent (scheduler estimate)")
    elif held:
        # A blocked job isn't being scheduled — "calculating" would be misleading.
        emit("  When   not scheduled while blocked (see the reason above)")
    else:
        emit(
            f"  When   calculating{dots} "
            "(the scheduler estimates a start once the job has waited a few minutes)"
        )
    if pending.submit_time is not None:
        sub = time.strftime("%b %d %H:%M", time.localtime(pending.submit_time))
        waited = _fmt_wait(int(now - pending.submit_time))
        emit(f"         submitted {sub} {dot} waiting {waited} so far")
    if not held:
        # Held jobs have priority 0 → a bogus "everyone ahead" position; skip it.
        try:
            rank = resolve_priority_rank(pending.partition, pending.priority)
        except Exception:
            rank = None
        if rank is not None:
            ahead = max(0, rank[0] - 1)
            emit(f"         in line #{rank[0]} of {rank[1]} {dash} {ahead} ahead by priority")
    # Spell out that these are whole-job totals (not per-node), and say "whole
    # nodes" for an --exclusive job (it gets every core, so the CPU floor isn't
    # the whole story).
    node_word = "whole node" if pending.exclusive else "node"
    node_txt = f"{pending.req_nodes} {node_word}" + ("" if pending.req_nodes == 1 else "s")
    req = f"{node_txt}, {pending.req_cpus} CPU"
    if pending.req_mem_bytes > 0:
        # units.format_bytes: this renderer's own GiB helper read a --mem=20M
        # request as "0.0 GiB" (SW-4's shape, in the pending report).
        req += f", {format_bytes(pending.req_mem_bytes)}"
    if pending.req_gpus > 0:
        req += f", {pending.req_gpus}x {pending.req_gpu_type or 'GPU'}"
    emit(f"  Needs  {req}  (job totals)")
    counts = resolve_queue_counts(pending.partition)
    if counts is not None:
        running, waiting = counts
        emit(f"         queue on {pending.partition}: {running} running {dot} {waiting} pending")
    # None = squeue unavailable (busy controller); omit rather than print a
    # fabricated "0 running / 0 pending".
    parts = resolve_cluster_partitions(pending.partition, pending.account, pending.username)
    # The verdicts feed BOTH the table and the tip, so they are computed once here
    # rather than inside the table branch — nesting them there is what made the tip
    # unreachable for a held-like job once the table was suppressed.
    blocker = {p.name: fit_blocker(pending, p) for p in parts}
    fits = {p.name: (not p.is_current and blocker[p.name] == "") for p in parts}
    kept = [p for p in parts if p.is_current or fits[p.name]]
    for p in parts:
        if len(kept) >= _MAX_WHERE_ROWS:
            break
        if p not in kept:
            kept.append(p)
    dropped = len(parts) - len(kept)
    alts = [p for p in kept if fits[p.name]]
    if parts and capacity_is_irrelevant(pending.reason):
        # The same argument the `When` line and the `Tip` already apply: a held /
        # dependency / begin-time / reservation job is not waiting on capacity, so
        # "can run now?" is a question about something that is not the constraint.
        # Answering it 51 times, in the largest element on screen, contradicted the two
        # lines bracketing it — most starkly for a BeginTime job deliberately deferred
        # 24 hours, whose screen said yes fifty-one times. SW-29.
        emit(f"  Where  capacity is not the constraint {dash} not shown (see the reason above)")
    elif parts:
        emit("  Where  cluster capacity right now:")
        # Labelled header so every number is self-explanatory. A whole-node
        # (--exclusive) job needs fully-EMPTY nodes; so does a GPU job when we can't
        # see per-node free GPUs (gpu_detail False). But WITH gpu_detail a GPU job
        # can also land on a mixed node with enough GPUs spare (see `4e91d55`), so
        # those aren't "empty" — call it "free" like a plain job (mirrors the TUI).
        needs_empty = pending.exclusive or (
            pending.req_gpus > 0 and not any(p.gpu_detail for p in parts)
        )
        node_hdr = "empty nodes" if needs_empty else "free nodes"
        # "can run now?" claims capacity AND permission; only say that when the
        # association list was actually readable. Otherwise this column measured
        # room, and a partition with room can still reject the job (SW-2).
        verdict_hdr = "can run now?" if all(p.assoc_verified for p in parts) else "has room now?"
        emit(
            f"           {'partition':<16} {node_hdr:>11}  {'idle cores':>10}   "
            f"{'gpu':<14} {verdict_hdr}"
        )
        # Fit-first selection (mirror the TUI): keep the current partition + every
        # FITTING alternative, then fill to the cap with the rest — else a fitting
        # partition sorted beyond the cap is dropped and we'd falsely say "no room".
        # The blocker string ("" = fits) also names WHY a partition can't take it.
        for p in kept:
            if p.is_current:
                marker = "waiting (current)"
            elif not blocker[p.name]:
                marker = "FITS NOW"
            else:
                # The specific blocker (no GPU / time limit / node too small / down /
                # no room) instead of a blanket "no room" next to idle cores.
                marker = blocker[p.name]
            gpus = format_gpu_types(p.gpu_types, 14, ascii_mode=ascii_mode, has_gpus=p.has_gpus)
            navail = available_node_count(pending, p)
            # Elide with an ellipsis (not a silent hard cut) so two long names that
            # share a 16-char prefix don't render identically.
            pname = p.name if len(p.name) <= 16 else p.name[:13] + "..."
            emit(f"           {pname:<16} {navail:>11}  {p.cpus_idle:>10}   {gpus:<14} {marker}")
        if dropped > 0:
            # Same cap as the TUI's WHERE table (PendingView._MAX_ROWS) — say so
            # instead of silently cutting the list, matching its "... and N more".
            emit(f"           {dots} and {dropped} more partition(s)")
    # The TIP is decided independently of whether the TABLE was shown. Nesting it
    # inside the table branch meant SW-29's suppression for a held-like job silently
    # took the tip with it — and the tip is the actionable line round 56 called
    # correct ("moving to another partition won't start this job"). Found by testing
    # a comment that claimed this ladder mirrors the TUI's: both renderers agreed,
    # and both were wrong the same way.
    if is_usage_capped(pending.reason):
        # A usage cap, not a shortage: saying "won't help" overclaims (these
        # limits are often keyed per partition, so a move CAN change which one
        # applies) and saying "X has room" misdiagnoses it. Name the cap and the
        # one command that answers the question for this site (SW-29 follow-up).
        emit(f"  Tip    a usage limit is capping this job, not free capacity {dash} see the")
        emit("         reason above. A partition change may alter which limit applies;")
        emit("         check with: sacctmgr show assoc user=$USER format=Partition,QOS,GrpTRES")
    elif not requeue_could_help(pending.reason):
        # Held / dependency / begin-time / reservation: a partition change can't
        # start it, so don't suggest one.
        emit(f"  Tip    moving to another partition won't start this job {dash} it isn't")
        emit("         waiting on free capacity (see the reason above).")
    elif alts:
        best = alts[0]
        emit(
            f"  Tip    {best.name} has room for this request now {dash} requeue with: "
            f"scontrol update JobId={pending.job_id} Partition={best.name}"
        )
    elif not any(blocker[p.name] == "" for p in parts if p.is_current):
        # None of the job's own partition(s) can take it right now either. Test
        # the BLOCKER, not `fits` — `fits` is forced False for the current
        # partition (so the table never prints a self-contradictory "FITS NOW
        # (current)"), which would make this an unconditional claim. A job can sit
        # PENDING with its own partition genuinely able to hold it (Reason=Priority,
        # a QOS/assoc limit, a dependency), and saying "no partition has enough free
        # capacity" then directly contradicts the free-node and idle-core columns
        # printed above (mirrors the TUI's PendingView).
        if parts and all(blocker_is_permanent(blocker[p.name]) for p in parts):
            # Every partition is blocked by something waiting cannot change, so
            # "it will start once resources free up" would promise an event that
            # cannot happen. Point at the REQUEST instead of the queue. SW-28.
            biggest = largest_node_cpus(parts)
            emit("  Tip    no partition on this cluster can ever hold this request")
            emit(
                f"         (largest node: {biggest} CPU); it will not start as submitted."
                if biggest
                else "         ; it will not start as submitted."
            )
        else:
            emit("  Tip    no partition currently has free capacity for this request; it")
            emit("         will start once resources free up (the estimate above is Slurm's).")
    # else: the job's own partition could already take it — Slurm just hasn't
    # scheduled it yet (priority/QOS/dependency) — so there's nothing to suggest.
    emit("  source: scontrol/sinfo/squeue (a queue estimate; actual start is up to the scheduler)")


def _run_pending(pending: PendingJob, config: SlurmwatchConfig, args: argparse.Namespace) -> None:
    """Show the pending-job view: the live TUI on a real terminal, else text."""
    interactive = not (args.once or args.log) and sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        if not sys.stdout.isatty():
            # Redirected or piped: stdout is a data stream, and on a RUNNING job
            # this same invocation puts a CSV/JSON snapshot there (see
            # `_run_interactive`). Writing the human report to it instead meant
            # `sw "$JOBID" --json > out.json` silently produced prose whenever the
            # job happened to still be queued, and `jq` failed with a parse error
            # indistinguishable from a real one. Same rule and same exit status as
            # `--once`, which is the machine-oriented path this one degrades into:
            # prose to stderr, the non-zero status telling a script "queued" apart
            # from "broken" (#91) — and, since SW-27's fifth outcome, the FACTS on
            # stdout in the format that was asked for. `--once` started answering a
            # queued job in the requested format; this path degrades into `--once`,
            # so leaving `sw "$JOBID" --json > out.json` with an empty file made the
            # two disagree about the same event. Found by auditing a comment that
            # claimed the rule they shared, after the rule changed.
            fmt = args.format or ("json" if args.json else "csv")
            _emit_facts_payload(_pending_facts(pending), fmt, config)
            print(
                f"slurmwatch: job {pending.job_id} is PENDING — no snapshot to emit yet.",
                file=sys.stderr,
            )
            _print_pending_summary(pending, stream=sys.stderr, ascii_mode=config.ascii_mode)
            sys.exit(1)
        # stdout is a terminal but stdin is not (`echo | sw JOBID`), so there is
        # nobody to drive a TUI but the screen is still where a report belongs.
        _print_pending_summary(pending, ascii_mode=config.ascii_mode)
        return
    with _console_logging_suspended():
        try:
            from .tui import PendingApp

            app = PendingApp(pending, config)
            app.run(mouse=_mouse_enabled(config))
            return
        except Exception as exc:
            logger.error("TUI error: %s", exc)
    _print_pending_summary(pending, ascii_mode=config.ascii_mode)


def _run_interactive(job_id: str, config: SlurmwatchConfig, args: argparse.Namespace) -> None:
    job_ctx, pending = _resolve_running_or_pending(job_id)
    if pending is not None:
        # A queued job has no telemetry to show — surface why/when/where instead
        # of the dead-end "only running jobs can be monitored" error (#60).
        _run_pending(pending, config, args)
        return
    assert job_ctx is not None
    _apply_sampling_floor(config, job_ctx.remote)
    if job_ctx.remote:
        # Another user's job: Slurm won't let us create a step in their allocation
        # (the srun hop) or read their sstat usage, so there's no live telemetry to
        # be had. Skip the doomed hop and show an honest read-only facts view
        # instead of leaking srun's "Access/permission denied".
        if _job_owner_differs(job_ctx):
            _run_foreign(job_ctx, config, args)
            return
        # Our own job, off the compute node: climb a transport ladder for the real
        # live TUI, most-native first, and only fall back to the sstat-derived text
        # summary when no transport can reach the node.
        #   1. srun --overlap --pty hop (Slurm-native; works on most sites)
        #   2. ssh to the node (step-free; works where step creation is blocked or
        #      the GPU/GRES contends — the universal rung)
        #   3. sstat remote summary (coarse, but works wherever accounting is on)
        hop = _hop_to_compute_node(job_ctx, args)
        if hop == _HOP_RAN:
            return
        # The ssh rung is the INVASIVE one: some sites prohibit interactive logins
        # to compute nodes because each leaks threads into the job's .extern stepd
        # that are never freed, and past a few hundred the stepd livelocks and the
        # allocation has to be abandoned. `srun --overlap` costs none and reaches the
        # same node. So only climb to ssh when the gentler transport was TRIED and
        # failed — never when this environment declined it (SLURMWATCH_NO_HOP, no
        # terminal), because a user who opted out of a step did not opt in to a
        # login. SW-21.
        if hop in (_HOP_FAILED, _HOP_NO_SRUN) and _ssh_to_compute_node(job_ctx, args):
            return
        if hop == _HOP_DECLINED_POLICY and shutil.which("ssh") and not _env_disables_ssh():
            logger.info(
                "not falling back to an interactive login on the job's node "
                "(the srun hop was disabled here, not attempted); showing the "
                "remote summary instead."
            )
        _run_remote_summary(job_ctx, config)
        return
    collector = TelemetryCollector(job_ctx, config)
    # A TUI needs a terminal. Redirected or piped (`sw $SLURM_JOB_ID >> mon.log` in a
    # batch script, cron, a CI job), the live dashboard has nobody to draw for and no
    # keypress can ever quit it: it ran forever, wrote nothing to stdout, and pumped
    # ANSI redraw traffic into stderr at ~320 KB per 20 s. Emit one snapshot instead and
    # say how to keep sampling — the same degradation the pending, hop and ssh paths
    # already make. `--once`/`--log` never reach here, so those stay unaffected.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "stdout is not a terminal — emitting a single snapshot instead of the live "
            "dashboard.\nUse `--log FILE` to record continuously, or `--once` for exactly "
            "this one sample.",
            file=sys.stderr,
        )
        # Same format rule as --once: CSV unless json was asked for explicitly, with
        # SLURMWATCH_FORMAT as the same fallback --once/--log get. Unlike those paths,
        # an invalid value is never fatal here — the whole point of this branch is a
        # graceful degradation, and dying on a stale env var would defeat that.
        fmt = getattr(args, "format", "") or ("json" if getattr(args, "json", False) else "")
        if not fmt:
            try:
                fmt = _env_output_format()
            except ValueError:
                # Still not fatal — this branch exists to degrade gracefully, and a
                # stale variable from a site module file shouldn't kill `sw $JOBID |
                # tee`. But SLURMWATCH_FORMAT is load-bearing here (=csv really does
                # produce CSV), so an unusable value must not be dropped in SILENCE
                # the way it was. Same wording as every other tolerated env value
                # (warn_unusable_env), so the family has ONE shape for "couldn't use
                # this, here is what I did instead" (SW-17).
                warn_unusable_env(
                    "SLURMWATCH_FORMAT",
                    os.environ.get("SLURMWATCH_FORMAT", ""),
                    "expected 'json' or 'csv'",
                    "csv",
                )
        asyncio.run(
            _once_loop(collector, json_output=fmt == "json", csv_dialect=config.csv_dialect)
        )
        return
    # Buffer slurmwatch logging while the TUI owns the screen so a transient
    # collector warning/traceback can't corrupt the dashboard; replayed on exit.
    with _console_logging_suspended():
        try:
            from .tui import SlurmwatchApp

            app = SlurmwatchApp(job_ctx=job_ctx, collector=collector, config=config)
            app.run(mouse=_mouse_enabled(config))
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


def _csv_existing_header(log_path: str, dialect: str) -> list[str] | None:
    """An existing CSV log's header row.

    ``None`` when the file is missing/empty or isn't a slurmwatch CSV (no
    ``timestamp`` column), so callers fall back to their own sizing.
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
    return cols if "timestamp" in cols else None


def _csv_max_gpus_from_header(log_path: str, dialect: str) -> int | None:
    """The GPU-column width already established by an existing CSV log's header.

    Counts the ``gpu_<N>_index`` columns on the first line so an ``--append`` run
    reuses the file's layout instead of re-deriving a (possibly different) width
    from its own job — which would misalign the appended rows (#62). Returns
    ``None`` when there's no usable header, so the caller falls back to
    snapshot-based sizing.
    """
    cols = _csv_existing_header(log_path, dialect)
    if cols is None:
        return None
    return sum(1 for c in cols if c.startswith("gpu_") and c.endswith("_index"))


def _csv_conform_map(existing: list[str], current: list[str]) -> list[int | None]:
    """For each column of ``existing``, the index to read it from a ``current`` row.

    ``None`` marks a column the file has and this build no longer produces; it is
    written blank. Matching is by NAME, which is the only thing that survives a
    schema change — a positional append is exactly what shifted every field after
    the insertion point.
    """
    pos = {name: i for i, name in enumerate(current)}
    return [pos.get(name) for name in existing]


def _conform_csv_row(row: list[str], index_map: list[int | None]) -> list[str]:
    """Reorder one row into the file's column order, blank-filling what we lack."""
    return ["" if i is None or i >= len(row) else row[i] for i in index_map]


def _csv_append_layout(
    log_path: str, dialect: str, max_gpus: int, job_gpu_count: int
) -> list[int | None] | None:
    """The column mapping ``--append`` must write through, or None when it matches.

    Says on stderr when ``--append``'s target was written by a slurmwatch with a
    different CSV schema, or has fewer GPU columns than this job needs.

    Reusing the file's GPU-column width keeps the per-device groups lined up, but the
    FIXED columns come from this build — so a log written before a column was added
    (``cpu_peak_effective_cores``, ``mem_working_set_percent``, …) got rows WIDER than
    its own header, and every named column after the insertion point read shifted:
    `csv.DictReader` returned a number belonging to a different field, silently, for
    exactly the half of the file written after the append (SW-25 / round 39).

    Warning was not enough — it goes to stderr, which the batch and cron use that
    ``--append`` exists to serve discards, and mixed row widths are not an error to
    most readers. So the rows are now written in the FILE's column order, matched by
    name: the file stays internally consistent and every value lands under its own
    heading. Columns this build adds are announced and omitted (the old header cannot
    be retro-fitted); columns the file has and this build dropped are left blank.

    This is deliberately NOT SW-16's outright refusal, and the difference is whether
    correctness is reachable: two concurrent writers interleave records into a file
    nothing can reconstruct, so the only safe answer is to refuse. A width mismatch
    is recoverable — the file tells us its own layout — so refusing would break a
    working cron log across an upgrade for a case we can simply get right.

    Reusing the width is also, by itself, lossy whenever ``job_gpu_count`` exceeds
    ``max_gpus``: this run's real per-device columns beyond the file's width are
    never written at all (only the fixed ``gpu_count`` column still reflects them),
    and the schema-drift check above can never catch that on its own — it compares
    the CURRENT build's header sized to the SAME forced ``max_gpus``, so the two
    trivially agree on GPU-column count even though the data being dropped has
    nothing to do with a schema version.
    """
    existing = _csv_existing_header(log_path, dialect)
    if existing is None:
        return None
    if job_gpu_count > max_gpus:
        print(
            f"slurmwatch: {log_path} has {max_gpus} GPU column(s) but this job has "
            f"{job_gpu_count} — appended rows will drop the extra GPUs' detail columns "
            "(only gpu_count will still reflect them). Log to a new file, or drop "
            "--append to rewrite it.",
            file=sys.stderr,
        )
    current = TelemetrySnapshot.csv_header(max_gpus)
    if existing == current:
        return None
    unwritable = [c for c in current if c not in existing]
    blanked = [c for c in existing if c not in current]
    print(
        f"slurmwatch: {log_path} was written with a different CSV schema "
        f"({len(existing)} columns, this build writes {len(current)}) — appending in "
        "the FILE's column order so every value stays under its own heading"
        + (
            f"; {len(unwritable)} column(s) this build produces are not in that header "
            f"and will be omitted: {', '.join(unwritable[:4])}"
            + ("..." if len(unwritable) > 4 else "")
            + ". Log to a new file to capture them"
            if unwritable
            else ""
        )
        + (
            f"; {len(blanked)} column(s) in the file are not produced by this build "
            f"and will be blank: {', '.join(blanked[:4])}" + ("..." if len(blanked) > 4 else "")
            if blanked
            else ""
        )
        + ".",
        file=sys.stderr,
    )
    return _csv_conform_map(existing, current)


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
        # stderr and exit without creating an empty log file (#60). Non-zero, like
        # every other machine path with no data to hand back: exiting 0 having
        # written no file is indistinguishable from a completed recording, which
        # is precisely the ambiguity a caller of `--log` has to resolve (#91).
        print(f"slurmwatch: job {job_id} is PENDING — nothing to log yet.", file=sys.stderr)
        # One facts row rather than no file, exactly as the foreign-job branch below
        # does: a pipeline appending to this path otherwise gets no schema and no
        # explanation, and has to read English on stderr. rc stays non-zero — this mode
        # promised a recording and there is none (SW-27, fifth outcome).
        _write_facts_row(_pending_facts(pending), config, log_path, fmt)
        _print_pending_summary(pending, stream=sys.stderr, ascii_mode=config.ascii_mode)
        sys.exit(1)
    assert job_ctx is not None
    if job_ctx.remote and _job_owner_differs(job_ctx):
        # Another user's job: no live telemetry is readable cross-user, so don't
        # write a log of all-zero rows — print the honest summary and stop (M2).
        #
        # But write ONE facts row into the log first, in the log's own format. The
        # file was left empty, so a pipeline appending to it got no schema and no
        # explanation, and had to read English on stderr to find out why: "a row with
        # holes is far more useful than no row, and it keeps the schema stable for
        # whoever is appending to a file" (SW-27). The exit stays non-zero, because
        # unlike --once this mode promised a RECORDING and there is none.
        _write_facts_row(_foreign_facts(job_ctx), config, log_path, fmt)
        _run_foreign_summary(job_ctx, config, stream=sys.stderr)
        sys.exit(1)

    _apply_sampling_floor(config, job_ctx.remote)
    config.poll_interval = config.headless_interval
    # Prove the path is writable BEFORE announcing it. The banner used to print
    # first, so `--log /etc/nope.jsonl` said "logging job … to /etc/nope.jsonl" and
    # was then immediately followed by its own "Cannot write log file: Permission
    # denied" — a success line above its failure (round-3 nit). Mode "a" so this
    # neither truncates a file --append means to extend nor writes anything of its
    # own; the loop creates the file a moment later anyway.
    try:
        with open(log_path, "a"):
            pass
    except OSError as exc:
        logger.error("Cannot write log file: %s", _exception_text(exc))
        sys.exit(1)
    print(
        f"slurmwatch: logging job {job_id} to {log_path} (PID {os.getpid()})",
        file=sys.stderr,
    )

    code = asyncio.run(_headless_loop(job_ctx, config, log_path, fmt, append))
    if code:
        sys.exit(code)


def _exception_text(exc: BaseException) -> str:
    """Describe ``exc`` in a way that is never blank.

    Several exceptions worth reporting carry no message: ``str(TimeoutError())``
    and ``str(KeyError())`` are both ``""``, which turns "Cannot write log file:
    %s" into a line that names no reason at all. Fall back to the class name,
    which at least says what kind of failure it was.
    """
    return str(exc) or type(exc).__name__


# Grace period given to an in-flight log write to finish after a SIGINT/SIGTERM
# before we conclude the sink is wedged and hard-exit (B-C6). A module constant
# so tests can shorten it.
_HEADLESS_STUCK_WRITE_GRACE_SECONDS = 2.0

# How often a remote (off-node) headless run polls squeue to notice its job has
# ended — a remote collector never latches job_ended, so without this the log
# would grow forever after the job finishes (M3). squeue is an RPC, so throttle
# it; a module constant so tests can shorten it.
_HEADLESS_REMOTE_LIVENESS_SECONDS = 30.0


async def _headless_loop(
    job_ctx: JobContext,
    config: SlurmwatchConfig,
    log_path: str,
    fmt: str = "",
    append: bool = False,
) -> int:
    """Sample into ``log_path`` until the job ends or a signal stops us.

    Returns the process exit code: 0 when the job simply finished, 128+signum when a
    signal ended the run — the same convention the dashboard reports (143/129/130).
    """
    collector = TelemetryCollector(job_ctx, config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Which signal stopped us, so the exit code can say so (0 stays 0 for a job that
    # simply ended). The dashboard reports 143/129/130 for exactly this reason; a
    # --log run reporting 0 for "someone killed the logger" is the same ambiguity on
    # the path most likely to be wrapped by a script.
    stopped_by: list[int] = []

    def _signal_handler(signum: int) -> None:
        stopped_by.append(signum)
        shutdown_event.set()

    # SIGHUP too, and it is the one that matters most here: a `--log` run is the case
    # people start in a terminal and walk away from, so a closing tmux pane / IDE
    # terminal / dropped ssh session SIGHUPs it. Unhandled, the default action killed
    # the process outright — measured as rc -1, with no drain of the in-flight write
    # and no "monitoring stopped" line. Records are single atomic writes, so nothing
    # was corrupted, but the graceful path existed for SIGINT/SIGTERM only.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(signum, _signal_handler, signum)

    # A remote (off-node) collector never latches job_ended — the liveness monitor
    # that sets it needs the local cgroup to vanish, so it runs only for on-node
    # collectors. Poll squeue directly for a remote run and stop when the job leaves
    # the queue, or the log grows forever after the job ends (M3). The job is
    # known-running now, so the first check waits one interval.
    remote = job_ctx.remote
    liveness_id = job_ctx.raw_job_id or job_ctx.job_id
    last_liveness = loop.time()

    async def _remote_job_gone() -> bool:
        nonlocal last_liveness
        if not remote:
            return False
        now = loop.time()
        if now - last_liveness < _HEADLESS_REMOTE_LIVENESS_SECONDS:
            return False
        last_liveness = now
        try:
            # Off the event-loop thread — is_job_active shells out to squeue.
            active = await loop.run_in_executor(None, is_job_active, liveness_id)
        except Exception:
            return False  # transient squeue failure — assume alive, retry next interval
        return active is False  # None = unknown: keep going, never stop on uncertainty

    # fmt already folds in a validated, normalized SLURMWATCH_FORMAT (see the
    # caller); an explicit format always wins, the extension is only a fallback.
    use_json = _infer_use_json(fmt, log_path)
    # Set only when appending to a file whose header differs from this build's; the
    # writer then projects each row onto that header by name.
    conform_map: list[int | None] | None = None

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
        if forced_max_gpus is not None:
            # Same GPU width, so any remaining difference is in the FIXED columns —
            # i.e. the file predates a schema change and a positional append would
            # shift every field after the insertion point. Write through the file's
            # own layout instead of corrupting it quietly (SW-25).
            conform_map = _csv_append_layout(
                log_path, config.csv_dialect, forced_max_gpus, job_ctx.gpu_count_requested
            )

        # A raw fd, one os.write per record — NOT a buffered TextIOWrapper. Two
        # concurrent loggers on one path used to corrupt the file two different
        # ways: the default "w" made both truncate and then hold INDEPENDENT
        # offsets, so each wrote over the other's bytes mid-record; and even "a"
        # wasn't record-atomic, because a buffered write of a record larger than
        # the buffer becomes several write() syscalls that another appender can
        # interleave with. O_APPEND plus one write() per record is atomic on a
        # regular file, so a line is either whole or absent — never spliced. That
        # atomicity is why the lock below can ALLOW a second appender (with a warning)
        # and refuse only a truncating one: interleaved whole records are usable,
        # destroyed bytes are not. SW-16, narrowed by round 47's measurement.
        # ALWAYS O_APPEND, and truncate only after the lock is held. O_TRUNC at open
        # time happens BEFORE any lock, so a second writer that is about to be
        # refused would still have truncated the first one's file — and the first,
        # writing at its own now-stale offset, then left a hole of NUL bytes that
        # reads as one unparseable line. Locking first makes the refusal harmless;
        # O_APPEND in both modes is what keeps each record whole.
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            _claim_log_file(fd, log_path, append)
            if not append:
                # The "w" semantic, minus the race. A pipe / tty / /dev/stdout can't
                # be truncated and doesn't need to be.
                with contextlib.suppress(OSError):
                    os.ftruncate(fd, 0)
            # Sized to the job's actual GPU count from the first snapshot, then
            # fixed for the file's lifetime so every row lines up under the one
            # header (a >8-GPU node isn't clipped at 8, #38).
            csv_max_gpus: int | None = None
            # Skip the CSV header when appending to a non-empty file. A pipe or
            # /dev/stdout reports size 0, so it counts as fresh and `--log
            # /dev/stdout` still streams a header to the node switcher.
            header_needed = os.fstat(fd).st_size == 0

            def _write(snap: TelemetrySnapshot) -> None:
                nonlocal csv_max_gpus, header_needed
                if use_json:
                    _write_record(fd, (snap.to_json() + "\n").encode())
                    return
                if csv_max_gpus is None:
                    csv_max_gpus = (
                        forced_max_gpus
                        if forced_max_gpus is not None
                        else max(len(snap.gpus), job_ctx.gpu_count_requested)
                    )
                # newline="" is the csv idiom (the reader uses it too): the csv
                # module owns the line endings, not the platform.
                buf = io.StringIO(newline="")
                writer = csv.writer(buf, dialect=config.csv_dialect)
                if header_needed:
                    writer.writerow(TelemetrySnapshot.csv_header(csv_max_gpus))
                    header_needed = False
                row = snap.to_csv_row(csv_max_gpus)
                writer.writerow(row if conform_map is None else _conform_csv_row(row, conform_map))
                _write_record(fd, buf.getvalue().encode())

            while not shutdown_event.is_set():
                # Yield once per iteration, for the same reason the dashboard's poll
                # loop does: this loop's signal handling is `loop.add_signal_handler`,
                # which REPLACES the default disposition with a callback the loop must
                # run — so a branch that returned without awaiting would make the
                # logger immune to SIGTERM/SIGHUP/SIGINT as well as unresponsive
                # (verified: a starved loop survives SIGHUP outright). Every branch
                # below does await today; this makes that structural rather than
                # incidental, and this is the path that runs unattended for days.
                await asyncio.sleep(0)
                try:
                    snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No frame this second: the job may have ended (the collector
                    # stops enqueuing then). Exit cleanly instead of spinning
                    # forever writing nothing (#28).
                    if collector.job_ended or await _remote_job_gone():
                        print(_JOB_ENDED_NOTE, file=sys.stderr)
                        break
                    continue
                except OSError as exc:
                    # Reading telemetry is not writing the log. ssh and sstat fail
                    # with OSError SUBCLASSES — TimeoutError on a wedged hop,
                    # ConnectionResetError/BrokenPipeError when one drops — and the
                    # outer `except OSError` would report every one of them as
                    # "Cannot write log file", then exit 1, killing a days-long run
                    # over one bad cycle on a file that is perfectly writable. Worse,
                    # `str(TimeoutError())` is empty, so the message named no reason
                    # at all. A failed read is transient: say what actually failed and
                    # take the next cycle. (Under Python 3.10 this was reachable for a
                    # plain wait_for timeout too, because asyncio.TimeoutError is not
                    # the builtin there — the two only became one class in 3.11.)
                    logger.warning("Telemetry read failed: %s", _exception_text(exc))
                    # Paced, not tight: a source that fails instantly and forever
                    # would otherwise spin the loop hot and starve the signal
                    # handlers this loop depends on (the same hazard the sleep(0)
                    # above guards against).
                    await asyncio.sleep(max(config.poll_interval, 0.5))
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
                    await reap_cancelled(shutdown_fut)
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
                if collector.job_ended or await _remote_job_gone():
                    print(_JOB_ENDED_NOTE, file=sys.stderr)
                    break
        finally:
            # Closing releases the flock too, so a later run on the same path is
            # never refused by a lock this process left behind.
            os.close(fd)

    except OSError as exc:
        # Any open()/write failure — not just a missing parent dir: a directory
        # target (IsADirectoryError) or an unwritable path (PermissionError) are
        # sibling OSErrors, and used to escape a FileNotFoundError-only handler as
        # a raw traceback instead of this clean message (#52).
        logger.error("Cannot write log file: %s", _exception_text(exc))
        sys.exit(1)
    finally:
        await collector.stop()
        why = f" ({signal.Signals(stopped_by[0]).name})" if stopped_by else ""
        print(f"slurmwatch: monitoring stopped{why}", file=sys.stderr)
    # 128+signum, so a script can tell "the job ended" from "someone stopped the
    # logger" — the log itself looks identical either way.
    return 128 + stopped_by[0] if stopped_by else 0
