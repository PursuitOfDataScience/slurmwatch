# ruff: noqa: T201
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import signal
import socket
import sys
from typing import Any

from ._version import VERSION
from .collector import TelemetryCollector
from .config import SlurmwatchConfig
from .exceptions import (
    CgroupAccessError,
    CgroupNotFoundError,
    JobNotFoundError,
    JobNotRunningError,
    LoginNodeError,
    SlurmCommandError,
)
from .model import JobContext, TelemetrySnapshot
from .slurm import resolve_current_jobs, resolve_job_context

logger = logging.getLogger("slurmwatch")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)

HOSTNAME = socket.gethostname().split(".")[0]


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"interval must be positive, got {value}")
    return parsed


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

    job_id = args.job_id
    log_path: str | None = args.log
    headless = log_path is not None
    once = args.once

    if once and headless:
        logger.error("--once and --log are mutually exclusive")
        sys.exit(1)

    fmt = args.format or ("json" if args.json else "") or os.environ.get("SLURMWATCH_FORMAT", "")

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
        _run_interactive(job_id, config)


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
    app.run()
    if app.return_code:
        sys.exit(app.return_code)
    return None


def _resolve_or_die(job_id: str) -> JobContext:
    try:
        return resolve_job_context(job_id)
    except LoginNodeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except JobNotFoundError:
        logger.error("Job %s does not exist in the Slurm database.", job_id)
        sys.exit(1)
    except JobNotRunningError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except (CgroupNotFoundError, CgroupAccessError) as exc:
        logger.error(
            "Job %s: %s\n\nTo resolve this:\n"
            "  1. Make sure you are on the compute node running the job\n"
            "  2. Verify the job is in RUNNING state\n"
            "  3. Try: srun --jobid %s --overlap slurmwatch",
            job_id,
            exc,
            job_id,
        )
        sys.exit(1)
    except SlurmCommandError as exc:
        logger.error("Slurm command failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to resolve job context: %s", exc)
        sys.exit(1)


def _run_once(job_id: str, config: SlurmwatchConfig, fmt: str = "") -> None:
    job_ctx = _resolve_or_die(job_id)
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
            writer = csv.writer(sys.stdout, dialect=csv_dialect)
            writer.writerow(TelemetrySnapshot.csv_header())
            writer.writerow(snapshot.to_csv_row())
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for first snapshot")
        sys.exit(1)
    finally:
        await collector.stop()


def _run_interactive(job_id: str, config: SlurmwatchConfig) -> None:
    job_ctx = _resolve_or_die(job_id)
    collector = TelemetryCollector(job_ctx, config)
    try:
        from .tui import SlurmwatchApp

        app = SlurmwatchApp(job_ctx=job_ctx, collector=collector, config=config)
        app.run()
    except Exception as exc:
        logger.error("TUI error: %s", exc)
        sys.exit(1)
    finally:
        # stop_sync() sets the stop event and shuts NVML down synchronously;
        # the background task is torn down when the app's event loop closes.
        collector.stop_sync()
    if app.return_code:
        sys.exit(app.return_code)


def _run_headless(
    job_id: str,
    config: SlurmwatchConfig,
    log_path: str,
    fmt: str = "",
    append: bool = False,
) -> None:
    job_ctx = _resolve_or_die(job_id)

    config.poll_interval = config.headless_interval
    print(
        f"slurmwatch: logging job {job_id} to {log_path} (PID {os.getpid()})",
        file=sys.stderr,
    )

    asyncio.run(_headless_loop(job_ctx, config, log_path, fmt, append))


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

    if not fmt:
        fmt = os.environ.get("SLURMWATCH_FORMAT", "")
    # An explicit format always wins; the extension is only a fallback.
    use_json = fmt == "json" if fmt in ("json", "csv") else not log_path.endswith(".csv")

    try:
        await collector.start()

        mode = "a" if append else "w"
        with open(log_path, mode) as f:
            csv_writer: Any = None
            csv_headers = TelemetrySnapshot.csv_header()
            # Skip the CSV header when appending to a non-empty file.
            header_needed = f.tell() == 0

            while not shutdown_event.is_set():
                try:
                    snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if use_json:
                    f.write(snapshot.to_json() + "\n")
                else:
                    if csv_writer is None:
                        csv_writer = csv.writer(f, dialect=config.csv_dialect)
                        if header_needed:
                            csv_writer.writerow(csv_headers)
                    csv_writer.writerow(snapshot.to_csv_row())
                f.flush()

    except FileNotFoundError as exc:
        logger.error("Cannot write log file: %s", exc)
        sys.exit(1)
    finally:
        await collector.stop()
        print("slurmwatch: monitoring stopped", file=sys.stderr)
