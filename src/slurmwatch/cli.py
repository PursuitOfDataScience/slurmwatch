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
    JobNotFoundError,
    JobNotRunningError,
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
        type=int,
        default=None,
        help="Slurm job ID to monitor (auto-discovers if omitted)",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        type=str,
        default=None,
        help="Run headless and write JSON-Lines telemetry to FILE",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("slurmwatch").setLevel(logging.DEBUG)

    config = SlurmwatchConfig()
    if args.interval is not None:
        config.poll_interval = args.interval
        config.headless_interval = args.interval

    job_id = args.job_id
    log_path: str | None = args.log
    headless = log_path is not None

    if headless and job_id is None:
        logger.error("--log requires a job_id argument")
        sys.exit(1)

    if job_id is None:
        job_id = _auto_discover_job_id(headless=headless)
        if job_id is None:
            return

    if headless:
        _run_headless(job_id, config, log_path)  # type: ignore[arg-type]
    else:
        _run_interactive(job_id, config)


def _auto_discover_job_id(headless: bool = False) -> int | None:
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
        jid: int = int(jobs[0]["job_id"])  # type: ignore[call-overload]
        logger.info("Attaching to running job %d", jid)
        return jid

    if headless:
        return _select_job_interactive(jobs)

    _run_tui_selector()
    return None


def _select_job_interactive(jobs: list[dict[str, object]]) -> int | None:
    print(f"Running jobs ({len(jobs)} found):\n")
    for i, job in enumerate(jobs, 1):
        print(
            f"  {i}. [{job['job_id']}] "
            f"{job.get('partition', '?')}  "
            f"{job.get('name', '?')}  "
            f"nodes={job.get('nodes', '?')}  "
            f"wall={job.get('wall_time', '?')}"
        )
    print()

    while True:
        try:
            choice = input("Select a job by number (or press Enter to quit): ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(jobs):
                return int(jobs[idx]["job_id"])  # type: ignore[call-overload,no-any-return]
            print(f"Please enter a number between 1 and {len(jobs)}.", file=sys.stderr)
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


def _run_interactive(job_id: int, config: SlurmwatchConfig) -> None:
    try:
        job_ctx = resolve_job_context(job_id)
    except JobNotFoundError:
        logger.error("Job %s does not exist in the Slurm database.", job_id)
        sys.exit(1)
    except JobNotRunningError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to resolve job context: %s", exc)
        sys.exit(1)

    try:
        from .tui import SlurmwatchApp

        collector = TelemetryCollector(job_ctx, config)
        app = SlurmwatchApp(job_ctx=job_ctx, collector=collector)
        app.run()
    except Exception as exc:
        logger.error("TUI error: %s", exc)
        sys.exit(1)


def _run_tui_selector() -> None:
    try:
        from .tui import SlurmwatchApp

        app = SlurmwatchApp()
        app.run()
    except Exception as exc:
        logger.error("TUI error: %s", exc)
        sys.exit(1)


def _run_headless(job_id: int, config: SlurmwatchConfig, log_path: str) -> None:
    try:
        job_ctx = resolve_job_context(job_id)
    except (JobNotFoundError, JobNotRunningError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to resolve job context: %s", exc)
        sys.exit(1)

    config.poll_interval = config.headless_interval
    print(
        f"slurmwatch: logging job {job_id} to {log_path} (PID {os.getpid()})",
        file=sys.stderr,
    )

    asyncio.run(_headless_loop(job_ctx, config, log_path))


async def _headless_loop(
    job_ctx: JobContext,
    config: SlurmwatchConfig,
    log_path: str,
) -> None:
    collector = TelemetryCollector(job_ctx, config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    try:
        await collector.start()

        with open(log_path, "w") as f:
            csv_writer: Any = None
            csv_headers = TelemetrySnapshot.csv_header().split(",")

            while not shutdown_event.is_set():
                try:
                    snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if log_path.endswith(".csv"):
                    if csv_writer is None:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(csv_headers)
                    csv_writer.writerow(snapshot.to_csv_row().split(","))
                else:
                    f.write(snapshot.to_json() + "\n")
                f.flush()

    except FileNotFoundError as exc:
        logger.error("Cannot write log file: %s", exc)
        sys.exit(1)
    finally:
        await collector.stop()
        print("slurmwatch: monitoring stopped", file=sys.stderr)
