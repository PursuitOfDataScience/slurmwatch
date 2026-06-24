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
        "--once",
        action="store_true",
        default=False,
        help="Take a single snapshot and print to stdout, then exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output JSON (with --once or combined with --log for .csv files)",
    )
    parser.add_argument(
        "--interval",
        metavar="SECONDS",
        type=float,
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
        help="Output format for --log (default: inferred from extension, otherwise JSON)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("slurmwatch").setLevel(logging.DEBUG)

    config = SlurmwatchConfig.from_env()

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

    if once and job_id is None:
        logger.error("--once requires a job_id argument")
        sys.exit(1)

    if headless and job_id is None:
        logger.error("--log requires a job_id argument")
        sys.exit(1)

    if job_id is None:
        if os.environ.get("SLURMWATCH_MOCK") == "1":
            job_id = "12345"
        else:
            job_id = _auto_discover_job_id()
            if job_id is None:
                return

    if headless:
        assert log_path is not None
        fmt = args.format or os.environ.get("SLURMWATCH_FORMAT", "")
        _run_headless(job_id, config, log_path, fmt)
    elif once:
        _run_once(job_id, config, args.json)
    else:
        _run_interactive(job_id, config)


def _auto_discover_job_id() -> str | None:
    username = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    logger.info("Auto-discovering running jobs for user %s...", username)

    try:
        jobs = resolve_current_jobs(username)
    except Exception as exc:
        logger.error("Failed to query Slurm jobs: %s", exc)
        return None

    if not jobs:
        user_message = (
            f"No running Slurm jobs found for user '{username}'. "
            "Launch a job first or provide a job_id argument."
        )
        print(user_message)
        return None

    if len(jobs) == 1:
        jid: str = str(jobs[0]["job_id"])
        logger.info("Attaching to running job %s", jid)
        return jid

    from .tui import SlurmwatchApp

    app = SlurmwatchApp(jobs=jobs)
    app.run()
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


def _run_once(job_id: str, config: SlurmwatchConfig, json_output: bool) -> None:
    job_ctx = _resolve_or_die(job_id)
    collector = TelemetryCollector(job_ctx, config)
    asyncio.run(_once_loop(collector, json_output))


async def _once_loop(collector: TelemetryCollector, json_output: bool) -> None:
    await collector.start()
    try:
        snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=10.0)
        if json_output:
            print(snapshot.to_json())
        else:
            header = ",".join(TelemetrySnapshot.csv_header())
            row = ",".join(snapshot.to_csv_row())
            print(header)
            print(row)
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

        app = SlurmwatchApp(job_ctx=job_ctx, collector=collector)
        app.run()
    except Exception as exc:
        logger.error("TUI error: %s", exc)
        sys.exit(1)
    finally:
        # stop_sync() sets the stop event and shuts NVML down synchronously;
        # the background task is torn down when the app's event loop closes.
        collector.stop_sync()


def _run_headless(job_id: str, config: SlurmwatchConfig, log_path: str, fmt: str = "") -> None:
    job_ctx = _resolve_or_die(job_id)

    config.poll_interval = config.headless_interval
    print(
        f"slurmwatch: logging job {job_id} to {log_path} (PID {os.getpid()})",
        file=sys.stderr,
    )

    asyncio.run(_headless_loop(job_ctx, config, log_path, fmt))


async def _headless_loop(
    job_ctx: JobContext,
    config: SlurmwatchConfig,
    log_path: str,
    fmt: str = "",
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
    use_json = not (log_path.endswith(".csv") or fmt == "csv")

    try:
        await collector.start()

        with open(log_path, "w") as f:
            csv_writer: Any = None
            csv_headers = TelemetrySnapshot.csv_header()

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
                        csv_writer.writerow(csv_headers)
                    csv_writer.writerow(snapshot.to_csv_row())
                f.flush()

    except FileNotFoundError as exc:
        logger.error("Cannot write log file: %s", exc)
        sys.exit(1)
    finally:
        await collector.stop()
        print("slurmwatch: monitoring stopped", file=sys.stderr)
