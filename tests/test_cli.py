from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import errno
import io
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import slurmwatch.cli as cli
from slurmwatch import aio
from slurmwatch import config as config_mod
from slurmwatch import model as model_mod
from slurmwatch import pending as pending_mod
from slurmwatch import slurm as slurm_mod
from slurmwatch.cli import (
    _auto_discover_job_id,
    _build_parser,
    _console_logging_suspended,
    _env_disables_hop,
    _env_disables_ssh,
    _env_output_format,
    _headless_loop,
    _hop_connect_timeout,
    _hop_to_compute_node,
    _infer_use_json,
    _print_pending_summary,
    _resolve_or_die,
    _run_foreign_summary,
    _run_interactive,
    _ssh_to_compute_node,
    main,
)
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import (
    CgroupNotFoundError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.model import JobContext, TelemetrySnapshot
from slurmwatch.pending import PartitionResources, PendingJob
from slurmwatch.slurm import _job_owner_differs, resolve_job_context


class TestArgParser:
    def test_parse_job_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345"])
        assert args.job_id == "12345"

    def test_parse_array_job_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345_3"])
        assert args.job_id == "12345_3"

    def test_parse_het_job_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345+0"])
        assert args.job_id == "12345+0"

    def test_job_id_is_string(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345"])
        assert isinstance(args.job_id, str)

    def test_no_job_id(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.job_id is None

    def test_log_argument(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--log", "test.jsonl"])
        assert args.log == "test.jsonl"

    def test_interval_argument(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--interval", "2.0"])
        assert args.interval == 2.0

    @pytest.mark.parametrize("val", ["inf", "-inf", "nan", "1e999"])
    def test_interval_rejects_non_finite(self, val: str) -> None:
        # C2: inf/nan slip past the <= 0 check, then asyncio.sleep() misbehaves;
        # the flag path must reject them like the env path does.
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["12345", "--interval", val])

    def test_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--verbose"])
        assert args.verbose is True

    def test_demo_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--demo"])
        assert args.demo is True

    def test_ascii_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--ascii"])
        assert args.ascii is True

    def test_once_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--once"])
        assert args.once is True

    def test_format_argument(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["12345", "--log", "test.csv", "--format", "json"])
        args2 = parser.parse_args(["12345", "--log", "test.jsonl", "--format", "csv"])
        assert args.format == "json"
        assert args2.format == "csv"

    def test_version(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])


class TestMainMockMode:
    @staticmethod
    def _stub_tui(monkeypatch: pytest.MonkeyPatch) -> None:
        # Don't launch the real (blocking) TUI; just confirm routing/env setup.
        import slurmwatch.tui as tui

        monkeypatch.setattr(tui.SlurmwatchApp, "run", lambda self, *a, **k: None)

    @staticmethod
    def _fake_tty(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make stdin/stdout look like a terminal.

        Under pytest they are captured pipes, and slurmwatch deliberately refuses to
        launch a TUI without a terminal (it would draw for nobody and never be
        quittable). A test that exercises the interactive path has to say it has one.
        """
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    def test_main_demo_sets_mock_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        self._stub_tui(monkeypatch)
        main(["--demo"])
        assert os.environ.get("SLURMWATCH_MOCK") == "1"

    def test_main_demo_with_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        self._stub_tui(monkeypatch)
        main(["--demo", "12345"])
        assert os.environ.get("SLURMWATCH_MOCK") == "1"

    def test_tui_disables_mouse_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mouse capture off by default so terminal text selection/copy works.
        import slurmwatch.tui as tui

        monkeypatch.delenv("SLURMWATCH_MOUSE", raising=False)
        captured: dict[str, object] = {}
        monkeypatch.setattr(tui.SlurmwatchApp, "run", lambda self, *a, **k: captured.update(k))
        self._fake_tty(monkeypatch)
        main(["--demo", "12345"])
        assert captured.get("mouse") is False

    def test_tui_mouse_env_enables_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOUSE", "1")
        captured: dict[str, object] = {}
        monkeypatch.setattr(tui.SlurmwatchApp, "run", lambda self, *a, **k: captured.update(k))
        self._fake_tty(monkeypatch)
        main(["--demo", "12345"])
        assert captured.get("mouse") is True


class TestRemoteSummary:
    def test_off_node_prints_summary_not_tui(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # When the job's cgroups aren't local, `slurmwatch <id>` prints an
        # sstat-derived summary instead of launching the TUI.
        import slurmwatch.cli as cli
        from slurmwatch import slurm
        from slurmwatch.model import JobContext

        ctx = JobContext(
            job_id="51397890",
            username="u",
            partition="gpu",
            nodelist="midway3-0602",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=200 * 1024**3,
            gpu_count_requested=2,
            gpu_indices=[],
            job_start_time=1000.0,
            job_state="RUNNING",
            remote=True,
        )
        monkeypatch.setattr(cli, "resolve_job_context", lambda job_id: ctx)
        # This exercises the caller's *own* off-node job (→ sstat summary), so make
        # the current user match the job's owner; otherwise the foreign-job path
        # (no live telemetry across users) takes over.
        monkeypatch.setattr("getpass.getuser", lambda: "u")
        monkeypatch.setattr(
            slurm,
            "resolve_remote_usage",
            lambda job_id, node_count=1: slurm.RemoteUsage(
                rss_bytes=174 * 1024**3, cpu_seconds=7200, sampled=True
            ),
        )
        # Fail loudly if the TUI is launched on the remote path.
        import slurmwatch.tui as tui

        monkeypatch.setattr(
            tui.SlurmwatchApp,
            "run",
            lambda self, *a, **k: pytest.fail("TUI launched in remote mode"),
        )
        main(["51397890"])
        out = capsys.readouterr().out
        assert "Job 51397890" in out
        assert "Memory" in out and "GiB" in out
        assert "sstat" in out

    def _summary(self, rss_gib: float, limit_gib: float = 200.0) -> str:
        """The off-node summary for a given MaxRSS, straight through the printer."""
        from slurmwatch.cli import _print_remote_summary
        from slurmwatch.model import JobContext

        ctx = JobContext(
            job_id="7",
            username="u",
            partition="gpu",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=int(limit_gib * 1024**3),
            gpu_count_requested=0,
            gpu_indices=[],
            job_state="RUNNING",
            remote=True,
        )
        # Through the real producer: the OOM flags, the clamp and the "peak is the
        # current" shape all come from _collect_remote, so a hand-built snapshot
        # could assert a state the off-node path never actually produces.
        from slurmwatch import slurm
        from slurmwatch.collector import TelemetryCollector

        real = slurm.resolve_remote_usage
        slurm.resolve_remote_usage = lambda job_id, node_count=1: slurm.RemoteUsage(
            rss_bytes=int(rss_gib * 1024**3), cpu_seconds=7200, sampled=True
        )
        try:
            collector = TelemetryCollector(ctx)
            snap = collector._collect_snapshot_sync()
        finally:
            slurm.resolve_remote_usage = real

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_remote_summary(ctx, snap)
        # Whitespace-collapsed: these assertions are about the SENTENCE, and a
        # substring spanning a line break broke the moment the wrap was rebalanced
        # by one word. The wrap itself is eyeballed, not asserted.
        return " ".join(buf.getvalue().split())

    def test_the_off_node_peak_says_it_can_overstate(self) -> None:
        """The figure the guard fires on is a per-process RSS SUM, not a footprint.

        Measured live off-node: MaxRSS 61.34 GB against the same job's
        cache-INCLUSIVE cgroup peak of 54.27 GB and a 32.65 GB working set — 1.88x
        the truth, enough to fire the warning at 87.9% on a job at roughly half its
        limit. This summary is what a reader gets when they cannot have the
        dashboard, so it is the one place where "88% of --mem" must not stand alone.
        """
        out = self._summary(174.0)  # 87% of 200 GiB — above the warning threshold
        assert "can overstate" in out, out
        assert "before raising --mem" in out, out

    def test_a_peak_above_the_limit_is_called_impossible_not_100_percent(self) -> None:
        """A real footprint over the limit would already have been OOM-killed.

        The percentage is clamped to 100 for exactly this case, which hides the
        strongest available evidence that the reading double-counts shared pages.
        """
        out = self._summary(260.0)  # MaxRSS ABOVE a 200 GiB limit
        assert "ABOVE the limit" in out, out
        assert "shared page once per process" in out, out

    def test_a_job_well_under_its_limit_gets_no_caveat(self) -> None:
        """Caveat only where it changes the reader's next move, or it is noise."""
        out = self._summary(40.0)
        assert "can overstate" not in out
        assert "ABOVE the limit" not in out

    def test_the_source_note_names_both_directions_of_error(self) -> None:
        """It named only the under-report (detached workers, SW-23).

        Memory can also read HIGHER than reality for a different reason, and a note
        that lists one direction reads as a guarantee about the other.
        """
        out = self._summary(40.0)
        assert "far lower than reality" in out
        assert "also read HIGHER" in out, out


class TestConfigFromEnv:
    def test_config_from_env_float(self) -> None:
        os.environ["SLURMWATCH_POLL_INTERVAL"] = "2.5"
        try:
            config = SlurmwatchConfig.from_env()
            assert config.poll_interval == 2.5
        finally:
            del os.environ["SLURMWATCH_POLL_INTERVAL"]

    def test_config_from_env_bool(self) -> None:
        os.environ["SLURMWATCH_ASCII"] = "true"
        try:
            config = SlurmwatchConfig.from_env()
            assert config.ascii_mode is True
        finally:
            del os.environ["SLURMWATCH_ASCII"]

    def test_config_from_env_empty(self) -> None:
        config = SlurmwatchConfig.from_env()
        assert config.poll_interval == 0.5

    def test_config_from_env_history_seconds(self) -> None:
        os.environ["SLURMWATCH_HISTORY_SECONDS"] = "30"
        try:
            config = SlurmwatchConfig.from_env()
            assert config.history_seconds == 30
            assert isinstance(config.history_seconds, int)
        finally:
            del os.environ["SLURMWATCH_HISTORY_SECONDS"]

    def test_history_seconds_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #54: a huge finite value is a valid float, so from_env stores it, but it
        # must be capped so deque(maxlen=…) can't overflow C ssize_t on the first
        # UI update. The cap keeps history_seconds a sane, usable int.
        from collections import deque

        from slurmwatch.config import MAX_HISTORY_SECONDS

        monkeypatch.setenv("SLURMWATCH_HISTORY_SECONDS", "1e19")
        config = SlurmwatchConfig.from_env()
        assert config.history_seconds == MAX_HISTORY_SECONDS
        # The dashboard's maxlen (history_seconds / poll_interval) must now be a
        # valid deque size — no OverflowError.
        maxlen = int(round(config.history_seconds / max(config.poll_interval, 0.01)))
        deque(maxlen=maxlen)  # must not raise

    def test_poll_interval_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A huge finite SLURMWATCH_POLL_INTERVAL passes from_env but would freeze the
        # refresh for ~decades; it must be clamped to the ceiling (symmetry with the
        # floor and with history_seconds).
        from slurmwatch.config import MAX_INTERVAL

        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "1e9")
        config = SlurmwatchConfig.from_env()
        assert config.poll_interval == MAX_INTERVAL

    def test_config_from_env_gpu_idle(self) -> None:
        os.environ["SLURMWATCH_GPU_IDLE_PCT"] = "10.0"
        try:
            config = SlurmwatchConfig.from_env()
            assert config.gpu_idle_threshold == 10.0
        finally:
            del os.environ["SLURMWATCH_GPU_IDLE_PCT"]

    @pytest.mark.parametrize("val", ["inf", "-inf", "Infinity", "nan"])
    def test_config_rejects_non_finite_int(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        # Regression: int(float('inf')) raises an *uncaught* OverflowError
        # (not ValueError), crashing from_env instead of the clean message.
        monkeypatch.setenv("SLURMWATCH_HISTORY_SECONDS", val)
        with pytest.raises(ValueError, match="SLURMWATCH_HISTORY_SECONDS"):
            SlurmwatchConfig.from_env()

    @pytest.mark.parametrize("val", ["inf", "nan", "-inf"])
    def test_config_rejects_non_finite_float(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        # Regression: nan/inf parsed fine and survived the min-interval clamp
        # (max(nan, 0.05) == nan), later crashing the TUI / hanging the loop.
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", val)
        with pytest.raises(ValueError, match="SLURMWATCH_POLL_INTERVAL"):
            SlurmwatchConfig.from_env()

    def test_non_finite_env_exits_cleanly_via_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "inf")
        with pytest.raises(SystemExit) as exc_info:
            main(["12345", "--once"])
        assert exc_info.value.code == 2

    @pytest.mark.parametrize("val", ["2.0", "-0.5"])
    def test_config_rejects_out_of_range_cpu_underuse(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        # C3: a ratio outside [0, 1] would produce a nonsensical underuse verdict.
        monkeypatch.setenv("SLURMWATCH_CPU_UNDERUSE", val)
        with pytest.raises(ValueError, match="SLURMWATCH_CPU_UNDERUSE"):
            SlurmwatchConfig.from_env()

    @pytest.mark.parametrize("val", ["-5", "150"])
    def test_config_rejects_out_of_range_gpu_idle(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        # C3: a percent outside [0, 100] is meaningless as an idle threshold.
        monkeypatch.setenv("SLURMWATCH_GPU_IDLE_PCT", val)
        with pytest.raises(ValueError, match="SLURMWATCH_GPU_IDLE_PCT"):
            SlurmwatchConfig.from_env()

    def test_config_rejects_unknown_csv_dialect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # C3: a bad dialect used to surface as a raw csv.Error deep in the output
        # path; catch it at config time with a clear message.
        monkeypatch.setenv("SLURMWATCH_CSV_DIALECT", "definitely-not-a-dialect")
        with pytest.raises(ValueError, match="SLURMWATCH_CSV_DIALECT"):
            SlurmwatchConfig.from_env()

    def test_config_bool_error_message_is_bool_specific(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C4: a bad boolean must not be reported as "expected a finite number".
        monkeypatch.setenv("SLURMWATCH_ASCII", "maybe")
        with pytest.raises(ValueError, match="boolean") as exc_info:
            SlurmwatchConfig.from_env()
        assert "finite number" not in str(exc_info.value)


class TestEnvOutputFormat:
    """C4: SLURMWATCH_FORMAT is normalized case-insensitively and validated."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("json", "json"), ("JSON", "json"), ("Csv", "csv"), ("  json  ", "json")],
    )
    def test_normalizes_case_and_whitespace(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_FORMAT", raw)
        assert _env_output_format() == expected

    def test_unset_or_empty_is_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMWATCH_FORMAT", raising=False)
        assert _env_output_format() == ""
        monkeypatch.setenv("SLURMWATCH_FORMAT", "")
        assert _env_output_format() == ""

    def test_unknown_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_FORMAT", "yaml")
        with pytest.raises(ValueError, match="SLURMWATCH_FORMAT"):
            _env_output_format()

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_uppercase_format_env_emits_json_not_csv(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Regression: SLURMWATCH_FORMAT=JSON used to silently emit CSV.
        monkeypatch.setenv("SLURMWATCH_FORMAT", "JSON")
        main(["12345", "--once"])
        out = capsys.readouterr().out.strip().split("\n")[-1]
        record = json.loads(out)  # parses only if it's JSON, not a CSV row
        assert record["job_id"] == "12345"


class TestHeadlessFormatInference:
    @pytest.mark.parametrize(
        ("fmt", "path", "expect_json"),
        [
            ("json", "out.csv", True),  # explicit format always wins
            ("csv", "out.jsonl", False),
            ("", "out.csv", False),  # inferred from extension
            ("", "out.CSV", False),  # #53: case-insensitive — CSV, not JSON
            ("", "out.Csv", False),
            ("", "out.jsonl", True),  # default to JSON
            ("", "/dev/stdout", True),
        ],
    )
    def test_infer_use_json(self, fmt: str, path: str, expect_json: bool) -> None:
        assert _infer_use_json(fmt, path) is expect_json


def _snap_with_gpus(n: int) -> TelemetrySnapshot:
    from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics, TelemetrySnapshot

    return TelemetrySnapshot(
        timestamp=1.0,
        job_id="12345",
        step_id="0",
        hostname="cn1",
        elapsed_seconds=1,
        cpu=CpuMetrics(cores_allocated=8, usage_ns=0, usage_percent=0.0),
        memory=MemoryMetrics(
            current_bytes=0,
            limit_bytes=1,
            peak_bytes=0,
            usage_percent=0.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
        ),
        gpus=[
            GpuMetrics(
                index=i,
                uuid=f"G{i}",
                name="A100",
                utilization_percent=0.0,
                memory_used_bytes=0,
                memory_total_bytes=1,
                memory_utilization_percent=0.0,
                power_watts=0.0,
                temperature_celsius=0.0,
                throttling=False,
            )
            for i in range(n)
        ],
    )


class TestCsvAppendWidth:
    def test_header_width_helper(self, tmp_path: Path) -> None:
        from slurmwatch.cli import _csv_max_gpus_from_header
        from slurmwatch.model import TelemetrySnapshot

        p = tmp_path / "log.csv"
        p.write_text(",".join(TelemetrySnapshot.csv_header(2)) + "\n")
        assert _csv_max_gpus_from_header(str(p), "excel") == 2
        # A non-slurmwatch or missing file -> None (fall back to snapshot sizing).
        (tmp_path / "junk.csv").write_text("a,b,c\n1,2,3\n")
        assert _csv_max_gpus_from_header(str(tmp_path / "junk.csv"), "excel") is None
        assert _csv_max_gpus_from_header(str(tmp_path / "nope.csv"), "excel") is None

    def test_schema_drift_warns_instead_of_silently_misaligning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Reusing the GPU width keeps the per-device groups aligned, but the FIXED
        # columns come from this build: a log written before a column was added gets
        # wider rows than its own header, and everything after the insertion point
        # reads shifted. That has to be said out loud, not left to a stale header.
        from slurmwatch.cli import _csv_append_layout
        from slurmwatch.model import TelemetrySnapshot

        current = TelemetrySnapshot.csv_header(2)
        # An up-to-date file, job GPU count matching the file's width: no warning.
        good = tmp_path / "good.csv"
        good.write_text(",".join(current) + "\n")
        _csv_append_layout(str(good), "excel", 2, 2)
        assert capsys.readouterr().err == ""
        # A file from an older build (a fixed column absent): one clear warning that
        # names the new column and what to do about it.
        old = tmp_path / "old.csv"
        old.write_text(",".join(c for c in current if c != "cpu_peak_effective_cores") + "\n")
        _csv_append_layout(str(old), "excel", 2, 2)
        err = capsys.readouterr().err
        assert "different CSV schema" in err
        assert "cpu_peak_effective_cores" in err
        assert "FILE's column order" in err
        # No header at all (new/foreign file) -> nothing to compare, no noise.
        _csv_append_layout(str(tmp_path / "nope.csv"), "excel", 2, 2)
        assert capsys.readouterr().err == ""

    def test_schema_drift_warns_when_job_has_more_gpus_than_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Reusing the file's GPU-column width (forced, so rows stay aligned) is, by
        # itself, lossy when this job's real GPU count is larger — the schema-drift
        # check above can never catch it, since both sides are sized to the SAME
        # forced width. Only gpu_count would otherwise hint at the loss.
        from slurmwatch.cli import _csv_append_layout
        from slurmwatch.model import TelemetrySnapshot

        log = tmp_path / "log.csv"
        log.write_text(",".join(TelemetrySnapshot.csv_header(2)) + "\n")
        _csv_append_layout(str(log), "excel", 2, 4)  # 4-GPU job, 2-GPU-wide file
        err = capsys.readouterr().err
        assert "2 GPU column(s)" in err and "4" in err
        assert "--append" in err
        # The reverse (job GPU count <= the file's width) is lossless: no warning.
        _csv_append_layout(str(log), "excel", 2, 2)
        assert capsys.readouterr().err == ""

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_append_reuses_existing_header_width(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # #62 regression: a run whose job has 4 GPUs, appended to a file whose
        # header was written for 2 GPUs, must write 2-GPU-wide rows so every row
        # still lines up under the header — not 4-GPU-wide rows that overflow it.
        # That reuse is itself lossy (the extra 2 GPUs' columns never get written),
        # so it must also say so: the mock job context requests 4 GPUs, matching
        # the wider snapshot below.
        import slurmwatch.cli as climod
        from slurmwatch.model import TelemetrySnapshot

        log = tmp_path / "agg.csv"
        header = TelemetrySnapshot.csv_header(2)
        log.write_text(",".join(header) + "\n" + ",".join(_snap_with_gpus(2).to_csv_row(2)) + "\n")

        class _OneShot:
            def __init__(self, job_ctx: object, config: object) -> None:
                self.job_ended = False
                self._snap = _snap_with_gpus(4)  # a wider job than the file's header

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> TelemetrySnapshot:
                self.job_ended = True  # exit after one write
                return self._snap

        monkeypatch.setattr(climod, "TelemetryCollector", _OneShot)
        climod._run_headless("12345", SlurmwatchConfig(), str(log), append=True)

        rows = [ln for ln in log.read_text().splitlines() if ln]
        widths = {len(ln.split(",")) for ln in rows}
        assert widths == {len(header)}  # every row (incl. the 4-GPU append) matches the header
        err = capsys.readouterr().err
        assert "2 GPU column(s)" in err and "4" in err and "--append" in err
        assert len(rows) == 3  # header + seeded row + one appended row


class TestHeadlessLogErrors:
    @pytest.mark.usefixtures("mock_slurm_env")
    def test_log_to_a_directory_reports_clean_error_not_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # #52: a directory target raises IsADirectoryError (an OSError sibling of
        # FileNotFoundError). It must surface as the clean "Cannot write log file"
        # error + exit 1, not a raw traceback.
        with pytest.raises(SystemExit) as exc:
            main(["12345", "--log", str(tmp_path)])
        assert exc.value.code == 1


class TestRunOnce:
    @pytest.mark.usefixtures("mock_slurm_env")
    def test_run_once_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["12345", "--once", "--json"])
        out = capsys.readouterr().out
        record = json.loads(out.strip().split("\n")[-1])
        assert record["job_id"] == "12345"
        assert "cpu" in record and "memory" in record

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_run_once_csv(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["12345", "--once"])
        out = capsys.readouterr().out.strip().split("\n")
        assert out[0].startswith("timestamp")
        # header and data row have identical column counts
        assert len(out[0].split(",")) == len(out[1].split(","))
        assert "12345" in out[1]

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_run_once_csv_sizes_gpu_columns_to_device_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # #38: the mock job has 4 GPUs, so the CSV must carry gpu_0..gpu_3 columns
        # (not a fixed 8-then-clip) and report gpu_count=4, with node columns.
        main(["12345", "--once"])
        out = capsys.readouterr().out.strip().split("\n")
        header = out[0].split(",")
        row = out[1].split(",")
        assert "gpu_3_index" in header
        assert "gpu_4_index" not in header  # not padded to a fixed 8
        assert row[header.index("gpu_count")] == "4"
        assert "node_count" in header and "node_index" in header and "remote" in header

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_run_once_format_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Regression: --format used to be silently ignored with --once.
        main(["12345", "--once", "--format", "json"])
        out = capsys.readouterr().out
        record = json.loads(out.strip().split("\n")[-1])
        assert record["job_id"] == "12345"

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_run_once_json_broken_pipe_exits_quietly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # N6: `sw --once --json | head` — when the reader closes the pipe the write
        # raises BrokenPipeError; exit via _bounded_exit(0), not a raw traceback.
        import builtins

        def _broken_print(*_a: object, **_k: object) -> None:
            raise BrokenPipeError(32, "Broken pipe")

        codes: list[int] = []

        def _fake_exit(code: int) -> None:
            codes.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(builtins, "print", _broken_print)
        monkeypatch.setattr(cli, "_bounded_exit", _fake_exit)
        with pytest.raises(SystemExit):
            main(["12345", "--once", "--json"])
        assert codes == [0]

    def test_demo_once_needs_no_job_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Regression: --demo --once used to fail with 'requires a job_id'.
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        main(["--demo", "--once", "--json"])
        record = json.loads(capsys.readouterr().out.strip().split("\n")[-1])
        assert record["job_id"] == "12345"

    def test_once_and_log_are_exclusive(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["12345", "--once", "--log", "x.jsonl"])
        assert exc_info.value.code == 1

    def test_interval_must_be_positive(self) -> None:
        with pytest.raises(SystemExit):
            main(["12345", "--once", "--interval", "-3"])

    def test_bad_env_value_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: garbage env values used to crash with a raw traceback.
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "abc")
        with pytest.raises(SystemExit) as exc_info:
            main(["12345", "--once"])
        assert exc_info.value.code == 2


async def _wait_for_lines(path: Path, n: int, timeout: float = 5.0) -> None:
    """Wait until ``path`` has at least ``n`` non-empty lines.

    Deterministic replacement for a fixed sleep, which races the headless write
    loop under CPU load and made these tests flaky.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if path.exists():
            content = path.read_text().strip()
            if content and len(content.split("\n")) >= n:
                return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{path} did not reach {n} lines within {timeout}s")


class TestHeadlessLoop:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_headless_writes_jsonl(self, tmp_path: Path) -> None:
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "metrics.jsonl"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), ""))
        await _wait_for_lines(out, 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        lines = out.read_text().strip().split("\n")
        assert lines and json.loads(lines[0])["job_id"] == "12345"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_headless_writes_csv_via_format(self, tmp_path: Path) -> None:
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "metrics.log"  # no .csv extension; format forces csv
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "csv"))
        await _wait_for_lines(out, 2)  # header + at least one data row
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("timestamp")
        assert len(lines[0].split(",")) == len(lines[1].split(","))

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_format_json_overrides_csv_extension(self, tmp_path: Path) -> None:
        # Regression: --format json used to be silently ignored when the log
        # path ended in .csv.
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "metrics.csv"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "json"))
        await _wait_for_lines(out, 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        lines = out.read_text().strip().split("\n")
        assert json.loads(lines[0])["job_id"] == "12345"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_append_preserves_existing_lines(self, tmp_path: Path) -> None:
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "metrics.jsonl"
        out.write_text('{"existing": true}\n')
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "", append=True))
        await _wait_for_lines(out, 2)  # pre-existing line + at least one new row
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        lines = out.read_text().strip().split("\n")
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1])["job_id"] == "12345"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_headless_exits_when_job_ends(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Round 6: liveness is polled, so the tail of a log can post-date the exit
        # (a 35 s job's last row read elapsed 43 s with 0% CPU). The reader is told
        # where the honest final reading is instead of rediscovering it.
        # #28: when the collector reports the job ended, the headless logger must
        # write the frames it has and then exit cleanly (not spin forever).
        from slurmwatch.model import CpuMetrics, MemoryMetrics, TelemetrySnapshot

        snap = TelemetrySnapshot(
            timestamp=1.0,
            job_id="12345",
            step_id="0",
            hostname="cn1",
            elapsed_seconds=1,
            cpu=CpuMetrics(cores_allocated=1, usage_ns=0, usage_percent=0.0),
            memory=MemoryMetrics(
                current_bytes=0,
                limit_bytes=1,
                peak_bytes=0,
                usage_percent=0.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
            ),
            gpus=[],
        )

        class _EndingCollector:
            def __init__(self, *a: object, **k: object) -> None:
                self.job_ended = False
                self._served = False

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def next_snapshot(self) -> TelemetrySnapshot:
                if self._served:
                    self.job_ended = True
                    raise asyncio.TimeoutError  # no more frames; job has ended
                self._served = True
                return snap

        monkeypatch.setattr(cli, "TelemetryCollector", _EndingCollector)
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        out = tmp_path / "m.jsonl"
        # Must return on its own (job ended) without cancellation.
        await asyncio.wait_for(_headless_loop(ctx, cfg, str(out), "json"), timeout=5.0)
        lines = out.read_text().strip().split("\n")
        assert json.loads(lines[0])["job_id"] == "12345"
        # And it says where the honest final reading is: liveness is polled, so the
        # tail can straddle the exit (round 6's 35 s job whose last row read 43 s).
        err = capsys.readouterr().err
        assert "job ended" in err
        assert "peak columns" in err and "not the tail" in err, err

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_remote_headless_exits_when_squeue_says_job_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M3: a remote (off-node) collector never latches job_ended, so the headless
        # loop must poll squeue (is_job_active) and stop when the job leaves the
        # queue — else the log grows forever after the job ends.
        from slurmwatch.model import CpuMetrics, MemoryMetrics, TelemetrySnapshot

        snap = TelemetrySnapshot(
            timestamp=1.0,
            job_id="12345",
            step_id=None,
            hostname="cn1",
            elapsed_seconds=1,
            cpu=CpuMetrics(cores_allocated=1, usage_ns=0, usage_percent=0.0),
            memory=MemoryMetrics(
                current_bytes=0,
                limit_bytes=1,
                peak_bytes=0,
                usage_percent=0.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
            ),
            gpus=[],
            remote=True,
        )

        class _NeverEndsCollector:
            """A remote collector: emits frames forever, never sets job_ended."""

            def __init__(self, *a: object, **k: object) -> None:
                self.job_ended = False

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def next_snapshot(self) -> TelemetrySnapshot:
                return snap

        monkeypatch.setattr(cli, "TelemetryCollector", _NeverEndsCollector)
        monkeypatch.setattr(cli, "is_job_active", lambda _id: False)  # squeue: job gone
        monkeypatch.setattr(cli, "_HEADLESS_REMOTE_LIVENESS_SECONDS", 0.0)  # check at once
        ctx = resolve_job_context("12345")
        ctx.remote = True
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        out = tmp_path / "m.jsonl"
        # Must return on its own (squeue says gone) without cancellation.
        await asyncio.wait_for(_headless_loop(ctx, cfg, str(out), "json"), timeout=5.0)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_stuck_sink_does_not_block_shutdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-C6: a SIGINT/SIGTERM while a log write is wedged (a full pipe whose
        # reader stopped, a hung NFS mount) must still stop the process — the old
        # plain `await run_in_executor(_write)` never returned, so the loop could
        # never re-check the shutdown event. We now race the write against
        # shutdown and hard-exit if the sink stays stuck past a short grace.
        write_started = threading.Event()
        release = threading.Event()

        def _stuck_write(_fd: int, _payload: bytes) -> None:
            # The sink is a raw fd now (one write() per record, SW-16), so the wedge
            # goes on the record write rather than on a fake file object.
            write_started.set()
            release.wait(timeout=10.0)  # bounded so pytest can't hang

        monkeypatch.setattr(cli, "_write_record", _stuck_write)
        # Shorten the stuck-write grace so the test doesn't wait the real 2s.
        monkeypatch.setattr(cli, "_HEADLESS_STUCK_WRITE_GRACE_SECONDS", 0.1)

        exited: list[int] = []

        class _HardExitError(Exception):
            pass

        def _fake_bounded_exit(code: int) -> Any:
            exited.append(code)
            raise _HardExitError

        monkeypatch.setattr(cli, "_bounded_exit", _fake_bounded_exit)

        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        out = tmp_path / "m.jsonl"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "json"))
        try:
            # Wait until a write is genuinely in-flight and blocked.
            for _ in range(200):
                if write_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert write_started.is_set(), "the write never started"
            # Now interrupt: the real signal path the fix is about.
            os.kill(os.getpid(), signal.SIGINT)
            # The loop must terminate via the hard-exit path within a bounded time,
            # NOT hang on the wedged write.
            with pytest.raises(_HardExitError):
                await asyncio.wait_for(task, timeout=3.0)
            assert exited == [0]
        finally:
            release.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, _HardExitError):
                await task


class TestAutoDiscover:
    """B-T8: the advertised no-job-id default is never hit under SLURMWATCH_MOCK."""

    def test_no_jobs_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [])
        with pytest.raises(SystemExit) as exc:
            _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert exc.value.code == 1

    def test_single_job_auto_attaches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Headless (no picker possible): a lone job attaches directly.
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [{"job_id": "777"}])
        assert _auto_discover_job_id(SlurmwatchConfig(), interactive=False) == "777"

    def test_single_job_interactive_shows_picker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Interactive: a lone job must show the picker (consistent `sw`), NOT drop
        # straight into the dashboard — so it launches the app and returns None.
        import slurmwatch.tui as tui

        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [{"job_id": "777"}])
        launched: dict[str, Any] = {}

        class _FakeApp:
            def __init__(self, **kwargs: object) -> None:
                launched.update(kwargs)
                self.return_code = 0

            def run(self, **kwargs: object) -> None:
                launched["ran"] = True

        monkeypatch.setattr(tui, "SlurmwatchApp", _FakeApp)
        result = _auto_discover_job_id(SlurmwatchConfig(), interactive=True)
        assert result is None  # the picker app ran; no direct job_id returned
        assert launched.get("ran") is True
        assert [str(j["job_id"]) for j in launched["jobs"]] == ["777"]

    def test_multiple_jobs_non_interactive_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli,
            "resolve_current_jobs",
            lambda username=None: [{"job_id": "1"}, {"job_id": "2"}],
        )
        with pytest.raises(SystemExit) as exc:
            _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert exc.value.code == 1


class TestResolveOrDie:
    """B-T7: the CLI's primary failure messages (exit 1) per exception class."""

    @pytest.mark.parametrize(
        "exc",
        [
            JobNotFoundError("nope"),
            JobNotRunningError("pending"),
            CgroupNotFoundError("no cgroup"),
            SlurmCommandError("scontrol failed"),
            RuntimeError("unexpected"),
        ],
    )
    def test_each_error_exits_1(self, monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
        def _raise(job_id: str) -> JobContext:
            raise exc

        monkeypatch.setattr(cli, "resolve_job_context", _raise)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_or_die("12345")
        assert exc_info.value.code == 1


class TestSrunHop:
    """B-T2: the login-node srun hop — argv, env stripping, and NO_HOP marker."""

    def _ctx(self) -> JobContext:
        return JobContext(
            job_id="12345_3",
            username="u",
            partition="gpu",
            nodelist="cn007",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[],
            nodelist_resolved=["cn007"],
            raw_job_id="12348",
            remote=True,
        )

    def _force_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _TTY:
            def isatty(self) -> bool:
                return True

            def write(self, _data: str) -> int:  # for the terminal-reset write
                return len(_data)

            def flush(self) -> None:
                pass

        # cli looks these up on their modules at call time, so patching the real
        # modules (rather than re-exported names on cli) is what takes effect.
        monkeypatch.setattr("sys.stdin", _TTY())
        monkeypatch.setattr("sys.stdout", _TTY())
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        # Default: the job is still alive, so an abnormal session exit falls back to
        # the summary. Tests override this to exercise the "job ended" path. Without
        # this the hop would shell out to a real `squeue` on an abnormal exit.
        monkeypatch.setattr("slurmwatch.cli.is_job_active", lambda _id: True)

    @pytest.fixture(autouse=True)
    def _popen_delegates_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The --pty session runs under Popen now (SW-26 needs a handle to forward a
        signal to and to reap), while the GPU probe still uses subprocess.run. Rather
        than teach every test about both, delegate: this fake resolves
        `subprocess.run` at CALL time, so whatever each test patched it to is what the
        session sees, including its return code and its `calls` bookkeeping."""

        class _Popen:
            def __init__(self, cmd: list[str], env: dict[str, str] | None = None, **kw: Any):
                self._result = subprocess.run(cmd, env=env, **kw)

            def wait(self, timeout: float | None = None) -> int:
                return int(self._result.returncode)

            def send_signal(self, signum: int) -> None:  # pragma: no cover - unused here
                pass

        monkeypatch.setattr("subprocess.Popen", _Popen)

    def test_builds_command_and_sanitizes_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        calls: list[tuple[list[str], dict[str, str] | None]] = []

        # subprocess.run is called twice: a silent GPU probe (no --pty, runs
        # `true`) then the real --pty session. rc 0 for both = GPU available + a
        # clean quit. The probe passes stdout/stderr kwargs, so accept **kwargs.
        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            calls.append((cmd, env))

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.setenv("SLURM_NTASKS", "8")
        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)

        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN

        probe_cmd = next(c for c, _ in calls if "--pty" not in c)
        assert probe_cmd[-1] == "true"  # throwaway probe, not the TUI
        assert "--overlap" in probe_cmd and "-m" not in probe_cmd

        cmd, env = next((c, e) for c, e in calls if "--pty" in c)
        assert "--jobid=12348" in cmd  # numeric raw id, not the 12345_3 form
        assert "--overlap" in cmd
        # Bounded step creation so a GPU-saturated job can't hang the login node.
        assert "--immediate=10" in cmd
        assert "--nodelist=cn007" in cmd
        assert "-m" in cmd and "slurmwatch" in cmd
        assert "12345_3" in cmd  # the inner positional keeps the user's form
        assert "--gres=none" not in cmd  # probe passed -> request the GPU

        assert env is not None
        assert "SLURM_NTASKS" not in env  # surrounding allocation sizing dropped
        assert env["SLURM_CONF"] == "/etc/slurm/slurm.conf"  # but SLURM_CONF kept
        assert env["SLURMWATCH_NO_HOP"] == "1"  # child can't re-hop

    def test_gpu_probe_timeout_does_not_hang_the_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: the GPU probe (no --pty) must have a wall-clock timeout so a
        # wedged/slow slurmctld can't hang `sw <jobid>` forever. A TimeoutExpired
        # is treated as "no GPU" (probe False) and the hop still attaches.
        import subprocess as _sp

        self._force_tty(monkeypatch)

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            if "--pty" not in cmd:  # the GPU probe — simulate a hung controller
                raise _sp.TimeoutExpired(cmd, kwargs.get("timeout", 9))

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        # A failed probe now prefers the ssh transport (it can read the GPUs a step
        # cannot); pin it off so this test still exercises the srun attach path.
        monkeypatch.setenv("SLURMWATCH_NO_SSH", "1")
        args = _build_parser().parse_args(["12345_3"])
        # If the timeout weren't caught, this would raise instead of returning.
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN

    def test_unreachable_gpu_prefers_ssh_so_real_numbers_show(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: a blind step reports no GPU numbers; ssh reports real ones.

        On a multi-node training job an inner srun holds every GPU and this Slurm
        cannot share GRES between steps, so the monitor step is denied the devices
        outright. That is the COMMON shape, not an edge case, so when the probe
        fails the ssh transport is tried before settling for a blind step.
        """
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        monkeypatch.delenv("SLURMWATCH_ON_NODE", raising=False)
        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kw: Any) -> Any:
            calls.append(cmd)

            class _R:
                returncode = 1 if cmd[-1] == "true" else 0  # probe fails, rest OK

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN
        # ssh was used, and NO blind --gres=none step was attached.
        assert any(c[0].endswith("ssh") for c in calls), calls
        assert not any("--gres=none" in c for c in calls), calls

    def test_ssh_hop_marks_on_node_without_disabling_the_ssh_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recursion guard must not disable the node switcher's ssh transport.

        The two used to share SLURMWATCH_NO_SSH, so the on-node dashboard could not
        read ANY other node's GPUs — the switcher silently fell back to a blind
        step. ON_NODE stops self-hopping; NO_SSH stays the user's preference.
        """
        from slurmwatch.cli import _env_says_already_on_node, _ssh_to_compute_node

        self._force_tty(monkeypatch)
        monkeypatch.setenv("SLURMWATCH_ON_NODE", "1")
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert _env_says_already_on_node() is True
        # A SUCCEEDING ssh, so the only thing that can return False is the guard —
        # otherwise this passes for the wrong reason (ssh merely failing to run)
        # and the guard could be deleted unnoticed.
        ran: list[list[str]] = []

        def _fake_run(cmd: list[str], **kw: Any) -> Any:
            ran.append(cmd)

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        # Won't ssh to itself...
        assert _ssh_to_compute_node(self._ctx(), args) is False
        assert ran == [], "guard must short-circuit BEFORE spawning ssh"
        # ...but the stream transport for OTHER nodes remains available.
        from slurmwatch.remote import _ssh_stream_allowed

        assert _ssh_stream_allowed() is True

    def test_no_hop_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        args = _build_parser().parse_args(["12345_3"])
        # DECLINED, not failed: the difference decides whether the caller may climb
        # to the far more invasive ssh rung on this user's behalf (SW-21).
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_DECLINED_POLICY

    def test_nonzero_exit_falls_back_to_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-P8: if even the GPU-less attach can't run, return False so the caller
        # shows the remote summary instead of a blank screen.
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            class _R:
                returncode = 1

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_FAILED

    def test_gpu_available_requests_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Probe succeeds -> a monitor step can get the job's GPU -> attach WITH it
        # so the dashboard shows live GPU util.
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        cmds: list[list[str]] = []

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            cmds.append(cmd)

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN
        session = next(c for c in cmds if "--pty" in c)
        assert "--gres=none" not in session

    def test_gpu_busy_uses_gres_none_silently(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Probe fails (GPU held by the job's own step) -> attach with --gres=none
        # so the dashboard STILL opens (CPU/mem live) — and, crucially, with NO
        # "busy"/"held" noise reaching the user (the removed feature).
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        cmds: list[list[str]] = []

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            cmds.append(cmd)

            class _R:  # probe (no --pty) busy; --pty session attaches
                returncode = 0 if "--pty" in cmd else 1

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN
        session = next(c for c in cmds if "--pty" in c)
        assert "--gres=none" in session and "--mem=0" in session
        err = capsys.readouterr().err.lower()
        assert "busy" not in err and "held by" not in err

    def test_gpu_busy_session_fails_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # If even the --gres=none attach can't run, return False so the caller
        # shows the remote summary — but only after trying the GPU-less step.
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        cmds: list[list[str]] = []

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            cmds.append(cmd)

            class _R:
                returncode = 1

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_FAILED
        assert any("--gres=none" in c for c in cmds)  # it did try the GPU-less attach
        assert "couldn't run the live dashboard" in capsys.readouterr().err

    def test_job_cancelled_exits_clean_no_stale_summary(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # scancel SIGTERM-kills the --pty step (rc 143) *because* the job ended. The
        # hop must report "job ended" and return True (so the caller does NOT dump a
        # stale RUNNING sstat summary on the just-killed dashboard).
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)

        # is_job_active would say COMPLETING (alive) right after a cancel; the fix
        # keys off the SIGTERM exit code instead, so this must NOT reach squeue.
        def _boom(_id: str) -> bool:
            raise AssertionError("must not call is_job_active for a signal-killed step")

        monkeypatch.setattr("slurmwatch.cli.is_job_active", _boom)

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            class _R:
                returncode = 0 if cmd[-1] == "true" else 143  # probe ok; session SIGTERM'd

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN  # clean -> no summary
        err = capsys.readouterr().err
        assert "cancelled or ended" in err
        assert "remote summary" not in err

    def test_hop_timeout_env_override_flows_to_srun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "25")
        cmds: list[list[str]] = []

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> Any:
            cmds.append(cmd)

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) == cli._HOP_RAN
        session = next(c for c in cmds if "--pty" in c)
        probe = next(c for c in cmds if "--pty" not in c)
        assert "--immediate=25" in session  # session honors the full timeout
        assert "--immediate=6" in probe  # probe capped at _GPU_PROBE_SECONDS


class TestHopConnectTimeout:
    """SLURMWATCH_HOP_TIMEOUT parsing: default, clamp, and bad input."""

    def test_default_and_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMWATCH_HOP_TIMEOUT", raising=False)
        assert _hop_connect_timeout() == 10  # unset -> default
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "30")
        assert _hop_connect_timeout() == 30
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "0")
        assert _hop_connect_timeout() == 2  # clamped to floor
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "9999")
        assert _hop_connect_timeout() == 120  # clamped to ceiling
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "not-a-number")
        assert _hop_connect_timeout() == 10  # bad input -> default

    def test_non_finite_values_fall_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # int(float("inf")) raises OverflowError; must not crash the hop.
        for bad in ("inf", "-inf", "1e400", "nan"):
            monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", bad)
            assert _hop_connect_timeout() == 10


class TestEnvDisablesHop:
    """B-P2: NO_HOP is a boolean, not a truthiness test."""

    def test_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        assert _env_disables_hop() is False  # unset -> hop allowed
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        assert _env_disables_hop() is True
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "0")
        assert _env_disables_hop() is False  # 0/false must NOT disable (the bug)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "false")
        assert _env_disables_hop() is False
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "yes")
        assert _env_disables_hop() is True


class TestConsoleLoggingSuspended:
    """B-C3: while the TUI owns the screen, logging is buffered, then replayed."""

    def test_buffers_then_restores(self) -> None:
        assert cli._handler in cli.logger.handlers
        with _console_logging_suspended():
            # The stderr handler is detached so records can't hit the screen.
            assert cli._handler not in cli.logger.handlers
            cli.logger.warning("collector hiccup")
        # Restored afterwards so post-TUI logging works again.
        assert cli._handler in cli.logger.handlers


class TestConfigEnvExtras:
    """B-P14: boolean spellings and OOM-threshold validation."""

    def test_ascii_accepts_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_ASCII", "on")
        assert SlurmwatchConfig.from_env().ascii_mode is True

    def test_ascii_rejects_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_ASCII", "maybe")
        with pytest.raises(ValueError, match="SLURMWATCH_ASCII"):
            SlurmwatchConfig.from_env()

    def test_inverted_oom_thresholds_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_OOM_WARN", "0.95")
        monkeypatch.setenv("SLURMWATCH_OOM_CRIT", "0.85")
        with pytest.raises(ValueError, match="OOM"):
            SlurmwatchConfig.from_env()

    def test_out_of_range_oom_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_OOM_CRIT", "1.5")
        with pytest.raises(ValueError, match="SLURMWATCH_OOM_CRIT"):
            SlurmwatchConfig.from_env()


def _plain_markup(text: str) -> str:
    """Strip Rich/Textual markup so a card's figures compare to the plain-text ones."""
    return re.sub(r"\[/?[^\]]*\]", "", text)


def _remote_snapshot(rss: int, limit: int, cpu_seconds: float, sampled: bool) -> TelemetrySnapshot:
    """An off-node snapshot built by the REAL producer, so its OOM flags are real."""
    from slurmwatch import slurm
    from slurmwatch.collector import TelemetryCollector

    ctx = JobContext(
        job_id="9",
        username="u",
        partition="p",
        nodelist="cn001",
        hostname="login-01",
        cpus_allocated=4,
        mem_limit_bytes=limit,
        gpu_count_requested=0,
        gpu_indices=[],
        job_start_time=1000.0,
        remote=True,
    )
    real = slurm.resolve_remote_usage
    slurm.resolve_remote_usage = lambda job_id, node_count=1: slurm.RemoteUsage(
        rss_bytes=rss, cpu_seconds=cpu_seconds, sampled=sampled
    )
    try:
        return TelemetryCollector(ctx)._collect_snapshot_sync()
    finally:
        slurm.resolve_remote_usage = real


def _held_pending_job() -> PendingJob:
    """A job whose reason is held-like, so capacity is not what it waits on (SW-29)."""
    return PendingJob(
        job_id="1",
        raw_job_id="1",
        name="j",
        username="u",
        partition="cur",
        qos="",
        account="",
        reason="Dependency",
        submit_time=None,
        start_time_estimate=None,
        priority=100,
        req_cpus=4,
        req_nodes=1,
        req_mem_bytes=8 * 1024**3,
        req_gpus=1,
        req_gpu_type="",
        time_limit_seconds=3600,
    )


class TestAsciiModeLeavesNoUnicodeInAnyTextSurface:
    """`--ascii` is a promise about BYTES, so assert on bytes, not on one glyph.

    The three plain-text summaries are what a reader falls back to when they cannot
    have the dashboard — which is exactly the terminal most likely to have asked for
    ascii. Two of them leaked: the foreign summary's `source:` line hardcoded an em
    dash, and `_print_remote_summary` never consulted ascii_mode AT ALL, so all six
    of its dashes reached a terminal that had explicitly asked for none. Per-glyph
    substring checks are what let that stand — this walks every branch and asserts
    the whole output is ASCII, so the next hardcoded dash fails here rather than in
    somebody's terminal.
    """

    ASCII = SlurmwatchConfig(ascii_mode=True)

    @staticmethod
    def _assert_pure(out: str, label: str) -> None:
        bad = [(i + 1, ln) for i, ln in enumerate(out.splitlines()) if not ln.isascii()]
        assert not bad, f"{label}: non-ascii under --ascii: {bad}"

    def _foreign_ctx(self, **over: object) -> JobContext:
        ctx = JobContext(
            job_id="7_3",
            username="other",
            partition="amd",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=8,
            mem_limit_bytes=32 * 1024**3,
            gpu_count_requested=2,
            gpu_indices=[],
            nodelist_resolved=["cn001"],
            raw_job_id="7",
            job_state="RUNNING",
            job_start_time=1000.0,
            time_limit_seconds=7200,
            array_job_id="7",
            array_task_id="3",
            remote=True,
        )
        for k, v in over.items():
            setattr(ctx, k, v)
        return ctx

    def test_the_foreign_summary_is_pure_ascii(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "resolve_array_task_counts", lambda _id: (4, 2))
        _run_foreign_summary(self._foreign_ctx(), self.ASCII)
        self._assert_pure(capsys.readouterr().out, "foreign summary")

    def test_the_off_node_summary_is_pure_ascii_in_every_branch(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch.cli import _print_remote_summary

        limit = 64 * 1024**3
        ctx = self._foreign_ctx(username="u", mem_limit_bytes=limit, gpu_count_requested=4)
        # every branch that prints prose: healthy, warning, above the limit, an
        # underused CPU, nothing sampled yet, and acct-gather off.
        for label, rss, cpu_s, sampled in [
            ("healthy", 8 * 1024**3, 7200.0, True),
            ("warning", 60 * 1024**3, 7200.0, True),
            ("over the limit", 70 * 1024**3, 7200.0, True),
            ("underused cpu", 8 * 1024**3, 1.0, True),
            ("not sampled", 0, 0.0, False),
        ]:
            snap = _remote_snapshot(rss, limit, cpu_s, sampled)
            _print_remote_summary(ctx, snap, self.ASCII)
            self._assert_pure(capsys.readouterr().out, f"off-node summary ({label})")
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: True)
        _print_remote_summary(ctx, _remote_snapshot(0, limit, 0.0, False), self.ASCII)
        self._assert_pure(capsys.readouterr().out, "off-node summary (acct gather off)")

    def test_the_pending_summary_is_pure_ascii_when_capacity_is_not_the_constraint(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The held/dependency branch — the one SW-29 added, and it hardcoded a dash."""
        # Patched, not ambient. The branch needs a non-empty partition list, and
        # resolving that for real needs Slurm on PATH: this test passed on the cluster
        # and silently skipped the branch anywhere else (CI included) until the pin
        # below caught it.
        parts = [
            PartitionResources(
                "cur",
                True,
                idle_nodes=8,
                cpus_idle=240,
                max_node_cpus=48,
                max_idle_node_cpus=48,
                is_current=True,
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        job = _held_pending_job()
        buf = io.StringIO()
        _print_pending_summary(job, stream=buf, ascii_mode=True)
        out = buf.getvalue()
        # Pin that the branch was REACHED. Without this the test degrades quietly
        # into checking a shorter report if the partitions ever resolve empty — an
        # assertion aimed at a line it cannot get to, which is the failure mode this
        # whole exercise keeps turning up.
        assert "capacity is not the constraint" in out, out
        self._assert_pure(out, "pending summary (held)")


class TestForeignJob:
    """Watching another user's job: no doomed srun hop, an honest read-only summary.

    Slurm restricts step creation and sstat to a job's owner, so the live
    dashboard and the sstat summary both fail for someone else's job. slurmwatch
    detects the ownership mismatch up front (`_job_owner_differs`) and shows a
    facts-only summary instead of leaking srun's "Access/permission denied".
    """

    def _ctx(self, *, owner: str = "yifchen") -> JobContext:
        return JobContext(
            job_id="52211701_20",
            username=owner,
            partition="amd",
            nodelist="midway3-0523",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[],
            nodelist_resolved=["midway3-0523"],
            raw_job_id="52211747",
            job_state="RUNNING",
            job_start_time=1000.0,
            time_limit_seconds=7200,
            remote=True,
        )

    def test_differs_true_when_caller_not_owner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("getpass.getuser", lambda: "youzhi")
        assert _job_owner_differs(self._ctx(owner="yifchen")) is True

    def test_differs_false_for_own_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("getpass.getuser", lambda: "youzhi")
        assert _job_owner_differs(self._ctx(owner="youzhi")) is False

    def test_differs_false_when_owner_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # scontrol parse gap: don't guess — fall through to the normal hop path.
        monkeypatch.setattr("getpass.getuser", lambda: "youzhi")
        assert _job_owner_differs(self._ctx(owner="")) is False

    def test_differs_false_when_caller_unresolvable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> str:
            raise OSError("no login name")

        monkeypatch.setattr("getpass.getuser", _boom)
        assert _job_owner_differs(self._ctx(owner="yifchen")) is False

    def test_differs_false_for_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # root/SlurmUser can attach to any job — don't call it unreachable.
        monkeypatch.setattr("getpass.getuser", lambda: "root")
        monkeypatch.setattr("os.getuid", lambda: 0)
        assert _job_owner_differs(self._ctx(owner="yifchen")) is False

    def test_uid_match_beats_stale_username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # N11: in a `su`/`sudo -E` shell $USER is stale, so getpass.getuser() can
        # differ from the owner even for your OWN job. A uid match must win, or the
        # live hop for your own job is silently skipped.
        ctx = self._ctx(owner="youzhi")
        ctx.uid = 4242
        monkeypatch.setattr("os.getuid", lambda: 4242)  # our real uid == the owner's
        monkeypatch.setattr("getpass.getuser", lambda: "somebodyelse")  # stale $USER
        assert _job_owner_differs(ctx) is False

    def test_uid_mismatch_wins_over_matching_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The uid is authoritative the other way too: different uids => foreign, even
        # if a stale $USER happens to match the owner name.
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr("getpass.getuser", lambda: "yifchen")
        assert _job_owner_differs(ctx) is True

    def test_interactive_skips_hop_for_foreign_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ctx = self._ctx(owner="yifchen")
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("getpass.getuser", lambda: "youzhi")

        def _no_hop(*_a: Any, **_k: Any) -> bool:
            raise AssertionError("must not attempt the srun hop for another user's job")

        def _no_remote(*_a: Any, **_k: Any) -> None:
            raise AssertionError("must not run the sstat remote summary for another user's job")

        monkeypatch.setattr(cli, "_hop_to_compute_node", _no_hop)
        monkeypatch.setattr(cli, "_run_remote_summary", _no_remote)

        args = _build_parser().parse_args(["52211701_20"])
        _run_interactive("52211701_20", SlurmwatchConfig(), args)

        out = capsys.readouterr().out
        assert "52211701_20" in out
        assert "yifchen" in out  # names the owner
        # The honest message, not the misleading "not yet sampled" timing line.
        assert "another user's job" in out
        assert "not yet sampled" not in out

    def test_foreign_summary_reports_facts(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_foreign_summary(self._ctx(owner="yifchen"), SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "RUNNING" in out and "midway3-0523" in out and "amd" in out
        assert "owner: yifchen" in out
        assert "scontrol/squeue" in out  # source line makes the read-only origin explicit

    def test_foreign_summary_says_what_the_job_asked_for(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The request is the ONLY resource fact a foreign job can offer.

        Nothing about someone else's job can be measured, so what it asked for is all
        there is — and it is the question the reader has when they run this on another
        user's job. The summary printed the GPU count and dropped the cores and the
        memory sitting in the same job_ctx, while the TUI's foreign card had shown all
        of them all along. Measured live against a real foreign array task on this
        cluster: 4 CPU / 30 GiB were resolved and never displayed.
        """
        ctx = self._ctx()
        ctx.cpus_allocated = 8
        ctx.mem_limit_bytes = 32 * 1024**3
        ctx.gpu_count_requested = 2
        _run_foreign_summary(ctx, SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "8 CPU" in out, out
        assert "32.0 GiB" in out, out
        assert "2x GPU" in out, out
        assert "1 node" in out, out

    def test_a_multi_node_foreign_request_says_per_node(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Per-node figures on a multi-node job, or "4 nodes 16 CPU" reads as the total.

        Same rule as the TUI card, now from one shared helper instead of two copies.
        """
        ctx = self._ctx()
        ctx.nodelist_resolved = ["cn001", "cn002", "cn003"]
        ctx.cpus_allocated = 16
        ctx.mem_limit_bytes = 64 * 1024**3
        _run_foreign_summary(ctx, SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "3 nodes" in out, out
        assert "16 CPU/node" in out, out
        assert "64.0 GiB/node" in out, out

    def test_the_foreign_card_and_summary_name_the_same_numbers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Parity: fixing one renderer has been half a fix here repeatedly."""
        from slurmwatch.tui import ForeignJobView

        ctx = self._ctx()
        ctx.cpus_allocated = 12
        ctx.mem_limit_bytes = 48 * 1024**3
        ctx.gpu_count_requested = 3
        _run_foreign_summary(ctx, SlurmwatchConfig())
        text = capsys.readouterr().out
        view = ForeignJobView()
        view.job_ctx = ctx
        view.config = SlurmwatchConfig()
        card = _plain_markup(view._alloc(ctx, "·"))
        for figure in ("12 CPU", "48.0 GiB", "3x GPU"):
            assert figure in text, f"{figure} missing from the summary"
            assert figure in card, f"{figure} missing from the card"

    def test_once_foreign_job_keeps_prose_off_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # M2: --once on another user's job must not emit an all-zero telemetry row —
        # still true, and what stdout carries instead is FACTS with the measured
        # fields empty, not prose. SW-27 supersedes the "stdout must be empty" half:
        # `--once`'s documented default format is csv, so bare --once emitted zero
        # bytes on the payload channel while --json got a full object from the same
        # event, and the default path called that same event a success.
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr(cli, "TelemetryCollector", self._no_collector)
        with pytest.raises(SystemExit) as exc:
            cli._run_once("52211701_20", SlurmwatchConfig(), fmt="")
        assert exc.value.code == 0, "the default path calls this success; so does this"
        cap = capsys.readouterr()
        rows = list(csv.DictReader(cap.out.splitlines()))
        assert len(rows) == 1, cap.out
        assert rows[0]["owner"] == "yifchen"
        assert rows[0]["telemetry_available"] == "False"
        assert rows[0]["telemetry_unavailable_reason"] == "foreign_owner"
        assert rows[0]["cpu_percent"] == "", "measured fields empty, never 0"
        # The prose belongs INSIDE the payload's own field (SW-8), which is different
        # from prose being the payload — stdout parsing as one CSV row is the check
        # that matters, and the DictReader above is it.
        assert "another user's job" in rows[0]["reason"]

    def test_once_foreign_job_answers_json_in_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SW-8: `--once --json | jq` used to receive prose and a zero exit. The
        facts Slurm hands out cross-user ARE expressible as JSON; what it can't
        measure is null (never 0, the M2 hazard) and `telemetry_available` says so."""
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr(cli, "TelemetryCollector", self._no_collector)
        with pytest.raises(SystemExit) as exc:
            cli._run_once("52211701_20", SlurmwatchConfig(), fmt="json")
        # SW-27: rc 0, matching the DEFAULT path on the identical event — the two used
        # to disagree about whether a colleague's running job is an error. The payload
        # is complete and says `telemetry_available: false`, which is machine-readable
        # where an exit code shared with "job does not exist" is not.
        assert exc.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["telemetry_available"] is False
        assert payload["telemetry_unavailable_reason"] == "foreign_owner"
        assert payload["owner"] == "yifchen"
        assert payload["state"] == "RUNNING"
        assert "another user's job" in payload["reason"]
        for measured in (
            "cpu_percent",
            "cpu_effective_cores",
            "mem_working_set_bytes",
            "mem_peak_bytes",
            "gpu_utilization_percent",
        ):
            assert payload[measured] is None, f"{measured} must be null, not 0"
        # Requested figures are facts from scontrol, so they are filled in.
        assert payload["cpus_allocated"] == 4
        assert payload["gpu_count_requested"] == 1

    def test_once_foreign_job_answers_csv_in_csv(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr(cli, "TelemetryCollector", self._no_collector)
        with pytest.raises(SystemExit):
            cli._run_once("52211701_20", SlurmwatchConfig(), fmt="csv")
        rows = list(csv.reader(capsys.readouterr().out.splitlines()))
        assert rows[0][:3] == ["timestamp", "job_id", "job_name"]
        record = dict(zip(rows[0], rows[1], strict=True))
        assert record["owner"] == "yifchen"
        assert record["telemetry_available"] == "False"
        # Unmeasurable fields are EMPTY, not "0" — a 0 would be read as a reading.
        assert record["cpu_percent"] == ""
        assert record["mem_working_set_bytes"] == ""

    @staticmethod
    def _no_collector(*_a: Any, **_k: Any) -> None:
        raise AssertionError("must not build a collector for another user's job")

    def test_headless_foreign_job_exits_without_writing_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # M2: --log on another user's job must not create a log of all-zero rows —
        # still true. SW-27 changes the other half: instead of an EMPTY file, write the
        # one row we can produce (facts, measured fields empty, reason token), because
        # a pipeline appending to that path otherwise got no schema and no explanation
        # and had to read English on stderr. rc stays non-zero: unlike --once, this
        # mode promised a recording and there is none.
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)

        def _no_collector(*_a: Any, **_k: Any) -> None:
            raise AssertionError("must not build a collector for another user's job")

        monkeypatch.setattr(cli, "TelemetryCollector", _no_collector)
        log = tmp_path / "foreign.jsonl"
        with pytest.raises(SystemExit) as exc:
            cli._run_headless("52211701_20", SlurmwatchConfig(), str(log), fmt="json")
        assert exc.value.code == 1
        assert log.exists(), "one facts row, not an empty file"
        rows = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
        assert len(rows) == 1, rows
        assert rows[0]["telemetry_available"] is False
        assert rows[0]["telemetry_unavailable_reason"] == "foreign_owner"
        assert rows[0]["owner"] == "yifchen"
        assert rows[0]["cpu_percent"] is None, "never a zero a consumer would average"
        assert "another user's job" in capsys.readouterr().err


class TestNonTerminalAndInterrupts:
    """The machine-facing edges of the interactive path."""

    def test_redirected_stdout_emits_one_snapshot_instead_of_hanging(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `sw $SLURM_JOB_ID >> mon.log` in a batch script used to block forever in a TUI
        # nobody could see or quit, writing 0 bytes to stdout while pumping ANSI redraw
        # traffic into stderr (~320 KB per 20 s). Under pytest stdout is already a pipe,
        # so this is the natural state — assert we degrade instead of launching the app.
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")

        def _boom(self: object, *a: object, **k: object) -> None:
            raise AssertionError("the TUI must not launch without a terminal")

        monkeypatch.setattr(tui.SlurmwatchApp, "run", _boom)
        main(["--demo", "12345"])
        out = capsys.readouterr()
        # One CSV header + one data row on stdout, and guidance on stderr.
        assert out.out.startswith("timestamp,job_id,")
        assert len([ln for ln in out.out.splitlines() if ln.strip()]) == 2
        assert "not a terminal" in out.err
        assert "--log" in out.err and "--once" in out.err

    def test_redirected_stdout_with_json_is_honored_without_a_contradictory_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A RUNNING job on a redirected stdout degrades to one snapshot (previous
        # test) and DOES honour --json there — so warning "has no effect ... ignoring"
        # up front, then using it a moment later, was a direct contradiction. The
        # module's own StreamHandler binds sys.stderr at import time (before capsys
        # swaps it), so the warning is checked via caplog, not capsys.
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        monkeypatch.setattr(
            tui.SlurmwatchApp, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
        )
        with caplog.at_level(logging.WARNING, logger="slurmwatch"):
            main(["--demo", "12345", "--json"])
        assert "ignoring" not in caplog.text
        record = json.loads(capsys.readouterr().out.strip().split("\n")[-1])
        assert record["job_id"] == "12345"  # --json really was honoured

    def test_json_warning_still_fires_for_the_real_interactive_tui(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The complement: on a genuine terminal (both stdin and stdout ttys) --json
        # really is a no-op (the live TUI never reads it), so the warning must stay.
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(tui.SlurmwatchApp, "run", lambda *a, **k: None)  # never really launch
        with caplog.at_level(logging.WARNING, logger="slurmwatch"):
            main(["--demo", "12345", "--json"])
        assert "--json has no effect without --once/--log; ignoring" in caplog.text

    def test_redirected_stdout_honours_slurmwatch_format_env_var(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # --once/--log read SLURMWATCH_FORMAT as a fallback; this degradation path
        # is the same "one snapshot, no flags needed" shape, so it should too.
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        monkeypatch.setenv("SLURMWATCH_FORMAT", "json")
        monkeypatch.setattr(
            tui.SlurmwatchApp, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
        )
        main(["--demo", "12345"])
        record = json.loads(capsys.readouterr().out.strip().split("\n")[-1])
        assert record["job_id"] == "12345"

    def test_redirected_stdout_ignores_a_bad_slurmwatch_format_rather_than_dying(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Unlike --once/--log (where a bad value is a fatal error), this path exists
        # specifically to degrade gracefully — dying on a stale env var here would
        # defeat the point, so an invalid value is silently ignored (falls to CSV).
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        monkeypatch.setenv("SLURMWATCH_FORMAT", "xml")
        monkeypatch.setattr(
            tui.SlurmwatchApp, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
        )
        main(["--demo", "12345"])
        out = capsys.readouterr().out
        assert out.startswith("timestamp,job_id,")  # CSV, not a crash

    def test_empty_job_id_does_not_silently_monitor_another_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "" is not None, so it skipped auto-discovery and ran `scontrol show job -d ""`
        # -- which means "all jobs". slurmwatch then monitored whatever that resolved to
        # (the caller's own allocation) while every record kept the empty id, so a --log
        # CSV had a blank job_id primary key on every row. Reached by the ordinary
        # `sw "$JOBID" --once` in a script where JOBID happens to be unset.
        seen: list[str | None] = []

        def _fake_once(job_id: str, config: object, fmt: str = "") -> None:
            seen.append(job_id)

        monkeypatch.setattr(cli, "_run_once", _fake_once)
        monkeypatch.setattr(cli, "_auto_discover_job_id", lambda *a, **k: "999")
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        main(["", "--once"])
        # Auto-discovery ran, so the empty string never reached the resolver.
        assert seen == ["999"]

    def test_whitespace_job_id_is_also_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(cli, "_run_once", lambda j, c, f="": seen.append(j))
        monkeypatch.setattr(cli, "_auto_discover_job_id", lambda *a, **k: "999")
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        main(["   ", "--once"])
        assert seen == ["999"]

    def test_keyboard_interrupt_exits_130_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ctrl-C during a slow scontrol or the first-snapshot wait escaped as a six-line
        # KeyboardInterrupt traceback; the hop/ssh paths already exited 130 cleanly.
        def _interrupt(job_id: str, config: object, fmt: str = "") -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_run_once", _interrupt)
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        with pytest.raises(SystemExit) as exc:
            main(["12345", "--once"])
        assert exc.value.code == 130


class _FakeStream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, _s: str) -> int:
        return 0

    def flush(self) -> None:
        pass


class TestSshToComputeNode:
    """The ssh-to-node fallback transport (rung 2 of the login->node ladder).

    Universally reaches the node where the srun step can't be created (nested-srun
    hang, GRES/step policy, `--gres` rejection) — pam_slurm_adm adopts the session
    into the job cgroup, which the /proc/self/cgroup discovery then finds.
    """

    @pytest.fixture(autouse=True)
    def _popen_delegates_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """This transport also runs its session under Popen now (SW-26: a signal must
        reach a handle to forward and reap). Delegate to whatever the test patched
        onto subprocess.run, resolved at call time."""

        class _Popen:
            def __init__(self, cmd: list[str], env: dict[str, str] | None = None, **kw: Any):
                self._result = subprocess.run(cmd, env=env, **kw)

            def wait(self, timeout: float | None = None) -> int:
                return int(self._result.returncode)

            def send_signal(self, signum: int) -> None:  # pragma: no cover
                pass

        monkeypatch.setattr("subprocess.Popen", _Popen)

    @staticmethod
    def _ctx() -> JobContext:
        return JobContext(
            job_id="123",
            username="u",
            partition="gpu",
            nodelist="cn01",
            hostname="login",
            cpus_allocated=4,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn01"],
            raw_job_id="123",
            remote=True,
        )

    @staticmethod
    def _args() -> Any:
        import argparse

        return argparse.Namespace(ascii=False, interval=None)

    def _tty(self, monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
        monkeypatch.setattr("sys.stdin", _FakeStream(on))
        monkeypatch.setattr("sys.stdout", _FakeStream(on))

    def test_bails_without_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=False)

        def _no_run(*a: Any, **k: Any) -> None:
            raise AssertionError("ssh must not run without a tty")

        monkeypatch.setattr("subprocess.run", _no_run)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False

    def test_env_disables_ssh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=True)
        monkeypatch.setenv("SLURMWATCH_NO_SSH", "1")
        assert _env_disables_ssh() is True

        def _no_run(*a: Any, **k: Any) -> None:
            raise AssertionError("ssh must not run when disabled")

        monkeypatch.setattr("subprocess.run", _no_run)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False

    def test_env_disables_ssh_false_value_keeps_it_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_NO_SSH", "0")
        assert _env_disables_ssh() is False

    def test_builds_command_and_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")
        captured: dict[str, list[str]] = {}

        class _R:
            returncode = 0

        def _run(cmd: list[str], *a: Any, **k: Any) -> _R:
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr("subprocess.run", _run)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is True
        cmd = captured["cmd"]
        assert cmd[0] == "/usr/bin/ssh"
        assert "-t" in cmd and "cn01" in cmd
        assert "BatchMode=yes" in cmd
        remote = cmd[-1]
        # ON_NODE (not NO_SSH) is the recursion guard: it stops the child hopping
        # to itself while leaving the ssh STREAM transport available, which is how
        # the node switcher reads ANOTHER node's GPUs on a multi-node job.
        assert "SLURMWATCH_NO_HOP=1" in remote and "SLURMWATCH_ON_NODE=1" in remote
        assert "SLURMWATCH_NO_SSH=1" not in remote
        assert "-m slurmwatch 123" in remote

    def test_command_carries_path_and_slurm_conf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # F1/F2: a non-login ssh shell doesn't source module PATH, so the rung must
        # carry PATH + SLURM_CONF via `env VAR=val` (csh-safe), not inline VAR=val cmd.
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")
        monkeypatch.setenv("PATH", "/opt/slurm/bin:/usr/bin")
        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
        captured: dict[str, list[str]] = {}

        class _R:
            returncode = 0

        def _run(cmd: list[str], *a: Any, **k: Any) -> _R:
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr("subprocess.run", _run)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is True
        remote = captured["cmd"][-1]
        assert remote.startswith("env ")  # external env, parsed by every shell
        assert "PATH=/opt/slurm/bin:/usr/bin" in remote
        assert "SLURM_CONF=/etc/slurm/slurm.conf" in remote
        assert not remote.startswith("SLURMWATCH_NO_HOP=")  # not the inline form

    def test_signal_killed_remote_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # F4: a signal-killed remote TUI (137/143) must reset the terminal and report
        # cancellation cleanly, not fall through to a stale summary on a torn screen.
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")

        class _R:
            returncode = 143  # SIGTERM from scancel/timeout

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        assert _ssh_to_compute_node(self._ctx(), self._args()) is True
        assert "cancelled or ended" in capsys.readouterr().err

    def test_returns_false_on_ssh_transport_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")

        class _R:
            returncode = 255  # ssh-level failure (unreachable / not permitted)

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False

    def test_a_refused_ssh_says_so_instead_of_leaking_ssh_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The hop runs on a PTY, so ssh's own stderr lands on the user's terminal.

        Measured on a Booth cluster that refuses login->compute ssh: between
        slurmwatch's two lines the reader got a bare
        `youzhi@mcn57: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).`
        — unattributed, and alarming for a path that recovered on its own. There is no
        second stream to redirect on a PTY, so ssh's diagnostics are silenced and the
        cause is stated in our own words.
        """
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")
        captured: dict[str, list[str]] = {}

        class _R:
            returncode = 255

        def _run(cmd: list[str], *a: Any, **k: Any) -> _R:
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr("subprocess.run", _run)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False
        assert "LogLevel=QUIET" in captured["cmd"]
        # ...and the node is still the last argument before the remote command.
        assert captured["cmd"][-2] == "cn01"
        err = capsys.readouterr().err
        assert "ssh to cn01 is not permitted here" in err
        assert "remote summary" in err

    def test_verbose_keeps_sshs_own_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reader debugging the transport is the one who needs ssh's words."""
        import logging

        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")
        captured: dict[str, list[str]] = {}

        class _R:
            returncode = 0

        def _run(cmd: list[str], *a: Any, **k: Any) -> _R:
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr("subprocess.run", _run)
        old = cli.logger.level
        cli.logger.setLevel(logging.DEBUG)
        try:
            assert _ssh_to_compute_node(self._ctx(), self._args()) is True
        finally:
            cli.logger.setLevel(old)
        assert "LogLevel=QUIET" not in captured["cmd"]

    def test_remote_exit_with_live_job_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")

        class _R:
            returncode = 1

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.setattr(cli, "is_job_active", lambda jid: True)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False

    def _ladder(self, monkeypatch: pytest.MonkeyPatch, hop_outcome: str) -> dict[str, bool]:
        ctx = self._ctx()
        monkeypatch.setattr(cli, "resolve_job_context", lambda job_id: ctx)
        monkeypatch.setattr("getpass.getuser", lambda: "u")
        monkeypatch.setattr(cli, "_hop_to_compute_node", lambda *a: hop_outcome)
        seen: dict[str, bool] = {}

        def _ssh(*a: Any, **k: Any) -> bool:
            seen["ssh"] = True
            return True

        monkeypatch.setattr(cli, "_ssh_to_compute_node", _ssh)
        monkeypatch.setattr(
            cli, "_run_remote_summary", lambda *a, **k: seen.__setitem__("summary", True)
        )
        main(["123"])
        return seen

    def test_ladder_prefers_ssh_over_summary_when_the_hop_TRIED(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The hop attempted an attach and couldn't: ssh is the right next rung, and
        # it runs BEFORE the sstat summary.
        seen = self._ladder(monkeypatch, cli._HOP_FAILED)
        assert seen.get("ssh") is True
        assert "summary" not in seen

    def test_a_declined_hop_does_not_buy_an_interactive_login(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SW-21: some sites prohibit interactive logins to compute nodes because
        each leaks threads into the job's .extern stepd that are never freed — past a
        few hundred the stepd livelocks and the allocation must be abandoned. A user
        who set SLURMWATCH_NO_HOP opted out of a cheap STEP; taking the invasive
        login on their behalf is the opposite of what they asked for."""
        seen = self._ladder(monkeypatch, cli._HOP_DECLINED_POLICY)
        assert "ssh" not in seen, "climbed to a login the user did not ask for"
        assert seen.get("summary") is True

    def test_no_srun_at_all_still_allows_the_login_rung(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing was declined — the transport simply isn't installed, so ssh is the
        only way to reach the node and remains worth trying."""
        seen = self._ladder(monkeypatch, cli._HOP_NO_SRUN)
        assert seen.get("ssh") is True


class TestPrintPendingSummary:
    """The plain-text 'why / when / where' report (--once/--log and non-tty paths) —
    cli.py's twin of tui.py's PendingView, sharing pending.py's fit_blocker/
    available_node_count/requeue_could_help, but rendered independently."""

    def test_tip_omitted_when_current_partition_can_hold_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mirrors test_pending.py's PendingView equivalent: a job can sit PENDING on
        # Reason=Priority while its own partition has ample room, so the "no partition
        # has capacity" tip — which would contradict the free-node/idle-core columns
        # printed above it — must not appear.
        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="",
            account="",
            reason="Priority",  # queued behind others, NOT short of resources
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=4,
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        parts = [
            PartitionResources(
                "cur",
                True,
                idle_nodes=8,
                cpus_idle=240,
                max_node_cpus=48,
                max_idle_node_cpus=48,
                is_current=True,
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        assert "no partition currently has free capacity" not in out

    def test_tip_still_shown_when_no_partition_fits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The complement: when the job genuinely does not fit its own partition, the
        # explanatory tip must still appear.
        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="",
            account="",
            reason="Resources",
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=999,  # more cores than any node in the partition has
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        parts = [
            PartitionResources(
                "cur", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        # SW-28: this fixture asks for 999 CPUs against a 48-CPU node, which is
        # PERMANENT — the old wording promised it "will start once resources free up".
        assert "can ever hold this request" in out, out
        assert "largest node: 48 CPU" in out
        assert "will not start as submitted" in out
        assert "once resources free up" not in out

    def test_the_transient_tip_survives_for_a_request_that_could_fit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The complement of the above: a job that WOULD fit on this hardware and is
        only waiting for cores must still be told it will start."""
        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="",
            account="",
            reason="Resources",
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=16,  # fits a 48-CPU node; simply none free right now
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        parts = [
            PartitionResources(
                "cur", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        assert "no partition currently has free capacity" in out, out
        assert "can ever hold" not in out

    def test_header_says_free_nodes_for_gpu_job_with_gpu_detail(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mirrors the PendingView test: post-4e91d55 a GPU job's node count includes
        # mixed nodes with enough free GPUs left, so the header must say "free
        # nodes", not claim every counted node is fully idle.
        job = pending_mod._mock_pending_job("777")
        job.req_gpus = 1
        job.reason = "Priority"  # "Resources" 's own explanation contains "free
        # nodes", which would pass the assertion below for the wrong reason.
        parts = [
            PartitionResources(
                "gpu-shared",
                True,
                idle_nodes=0,
                mix_nodes=3,
                cpus_idle=8,
                has_gpus=True,
                gpu_types=["a100"],
                free_gpus_per_node=[2, 0, 1],
                gpu_detail=True,
                is_current=True,
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        assert "free nodes" in out
        assert "empty nodes" not in out

    def test_header_still_says_empty_nodes_for_gpu_job_without_gpu_detail(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        job = pending_mod._mock_pending_job("777")
        job.req_gpus = 1
        job.reason = "Priority"
        parts = [
            PartitionResources(
                "gpu-shared",
                True,
                idle_nodes=2,
                cpus_idle=8,
                has_gpus=True,
                gpu_types=["a100"],
                is_current=True,
            )
        ]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        assert "empty nodes" in out
        assert "free nodes" not in out

    def test_where_table_truncates_with_a_more_partitions_notice(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mirrors the TUI's PendingView: the cap exists so a pathological unfiltered
        # list can't flood the terminal, but truncation must say so, not cut silently.
        job = pending_mod._mock_pending_job("777")
        job.reason = "Priority"
        # 1 current (fits, always kept) + 29 down partitions (none fit) — the fill
        # loop keeps current + 23 of the down ones to reach the cap of 24, dropping 6.
        parts = [
            PartitionResources("cur", True, idle_nodes=8, cpus_idle=64, is_current=True),
        ] + [PartitionResources(f"down{i}", False) for i in range(29)]
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        _print_pending_summary(job)
        out = capsys.readouterr().out
        assert "and 6 more partition(s)" in out


class TestRedirectedPendingRun:
    """A queued job on a redirected stdout must not put prose in the data stream.

    `--once` already got this right and says why in its own comment: "keep stdout
    clean … rather than polluting the stream with prose that a downstream jq/CSV
    reader would choke on". The plain redirected run did the opposite, and since
    `sw JOBID --json > out.json` emits real JSON the moment the job starts, a
    script got a parse error indistinguishable from a real failure whenever the
    job happened to still be queued (#91).
    """

    @staticmethod
    def _args(**over: object) -> Any:
        import argparse

        base: dict[str, object] = {"once": False, "log": "", "json": False, "format": ""}
        base.update(over)
        return argparse.Namespace(**base)

    @staticmethod
    def _stub_streams(monkeypatch: pytest.MonkeyPatch, *, stdin: bool, stdout: bool) -> None:
        monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout, raising=False)

    def test_redirected_run_writes_the_report_to_stderr_and_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        job = pending_mod._mock_pending_job("777")
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        self._stub_streams(monkeypatch, stdin=False, stdout=False)
        with pytest.raises(SystemExit) as exc:
            cli._run_pending(job, SlurmwatchConfig(), self._args())
        assert exc.value.code == 1  # the same status --once uses for "no snapshot"
        cap = capsys.readouterr()
        # The report is on stderr and the DATA stream carries data, not prose: one
        # CSV row (the default format for the machine paths) with the reason token.
        rows = list(csv.DictReader(cap.out.splitlines()))
        assert len(rows) == 1 and rows[0]["telemetry_unavailable_reason"] == "job_pending"
        assert "Why" not in cap.out and "Where" not in cap.out
        assert "is PENDING" in cap.err and "no snapshot to emit" in cap.err
        assert "Why" in cap.err  # the full why/when/where report, just redirected

    def test_json_on_a_queued_job_never_emits_half_a_document(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        job = pending_mod._mock_pending_job("777")
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        self._stub_streams(monkeypatch, stdin=False, stdout=False)
        with pytest.raises(SystemExit):
            cli._run_pending(job, SlurmwatchConfig(), self._args(json=True, format="json"))
        # A page of prose is what must never land here. Since SW-27's fifth outcome
        # this path answers in the requested format instead of leaving the file empty:
        # one complete document, parseable, with the token that says why there is no
        # telemetry — which is strictly more useful to `jq` than nothing.
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["telemetry_unavailable_reason"] == "job_pending"
        assert payload["telemetry_available"] is False
        assert "Why" not in out and "Where" not in out, "no prose on the data stream"

    def test_a_terminal_stdout_still_gets_the_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `echo | sw JOBID`: no TUI is possible (stdin is a pipe) but the screen is
        # still where a human's report belongs, and there is no data stream to
        # protect. This is also the post-TUI-failure fallback's case.
        job = pending_mod._mock_pending_job("777")
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        self._stub_streams(monkeypatch, stdin=False, stdout=True)
        cli._run_pending(job, SlurmwatchConfig(), self._args())  # returns, no exit
        cap = capsys.readouterr()
        assert "PENDING" in cap.out and "Why" in cap.out
        assert cap.err == ""

    def test_every_machine_path_agrees_on_a_queued_job(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The three shapes a script can use. They disagreed on where the prose went
        # and on the exit status, so which one you picked decided whether "queued"
        # was detectable at all.
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 777 is in state 'PENDING'.")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", pending_mod._mock_pending_job)
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        self._stub_streams(monkeypatch, stdin=False, stdout=False)
        log = tmp_path / "out.csv"
        runs: dict[str, Callable[[], None]] = {
            "--once": lambda: cli._run_once("777", SlurmwatchConfig()),
            "--log": lambda: cli._run_headless("777", SlurmwatchConfig(), str(log)),
            "redirected": lambda: cli._run_interactive("777", SlurmwatchConfig(), self._args()),
        }
        for label, run in runs.items():
            with pytest.raises(SystemExit) as exc:
                run()
            cap = capsys.readouterr()
            assert exc.value.code == 1, label
            assert "PENDING" in cap.err, label
            # Prose stays OFF stdout on every path — that was the disagreement this
            # test was written for, and it still holds. What changed (SW-27, fifth
            # outcome) is that "clean stdout" no longer means "empty stdout" for
            # --once: a queued job is the commonest no-telemetry outcome, and a poller
            # had to parse English on stderr to detect it.
            assert "Why" not in cap.out, label
            if label in ("--once", "redirected"):
                # Both are machine-oriented paths — the second degrades into the
                # first — so both answer a queued job in the requested format. They
                # disagreed until a stale comment claiming they shared a rule was
                # audited after the rule changed.
                rows = list(csv.DictReader(cap.out.splitlines()))
                assert len(rows) == 1, (label, cap.out)
                assert rows[0]["telemetry_unavailable_reason"] == "job_pending"
                assert rows[0]["state"] == "PENDING"
                assert rows[0]["cpu_percent"] == "", "requested, never measured"
            else:
                assert cap.out == "", f"{label} writes to the log file, not stdout"
        # The log holds the one row it CAN produce, in its own format, rather than
        # nothing at all — same as the foreign-job branch.
        rows = list(csv.DictReader(log.read_text().splitlines()))
        assert len(rows) == 1 and rows[0]["telemetry_unavailable_reason"] == "job_pending"


class TestHelpDocumentsTransports:
    """The ssh transport must be discoverable, and so must its opt-out.

    ssh used to be a rare last resort; it is now the PRIMARY path for a multi-node
    GPU job (a step cannot be granted GPUs an inner srun already holds). Each login
    leaks threads into the job's .extern stepd, so a user who does not want that
    needs to be able to find SLURMWATCH_NO_SSH without reading the source.
    """

    def _help(self) -> str:
        return _build_parser().format_help()

    def test_names_both_transports(self) -> None:
        text = self._help()
        assert "ssh" in text
        assert "srun" in text

    def test_names_the_opt_outs(self) -> None:
        text = self._help()
        assert "SLURMWATCH_NO_SSH" in text
        assert "SLURMWATCH_NO_HOP" in text

    def test_says_the_ssh_cost_is_bounded(self) -> None:
        """ "One per node per session" is the fact that makes the cost judgeable."""
        assert "One ssh login per node per session" in self._help()

    def test_says_what_opting_out_gives_up(self) -> None:
        """Opting out is not free: the GPU numbers go away. Say so."""
        text = self._help()
        assert "unavailable" in text


class TestNoJobIdWithoutATerminal:
    """SW-10: `slurmwatch > log` / `slurmwatch | tee` with no job id built the
    Textual app anyway — entered the alternate screen, drew the job picker into the
    pipe and waited forever for a keypress that cannot arrive. Measured: rc=137
    after a 20 s SIGKILL cap, 73 KB of escape sequences. The guard existed five
    times over in this file and simply was not on this path.
    """

    def _no_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)

        class _NeverApp:
            def __init__(self, **kwargs: object) -> None:
                raise AssertionError("a TUI must never be built on a non-terminal stdout")

        import slurmwatch.tui as tui

        monkeypatch.setattr(tui, "SlurmwatchApp", _NeverApp)

    def test_a_lone_job_is_attached_instead_of_shown_a_picker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_tty(monkeypatch)
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [{"job_id": "777"}])
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            cli, "_run_interactive", lambda job_id, config, args: seen.update(job_id=job_id)
        )
        main([])
        assert seen == {"job_id": "777"}

    def test_several_jobs_say_why_there_is_no_picker(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._no_tty(monkeypatch)
        monkeypatch.setattr(
            cli,
            "resolve_current_jobs",
            lambda username=None: [{"job_id": "1"}, {"job_id": "2"}],
        )
        with caplog.at_level("ERROR", logger="slurmwatch"), pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1
        assert "not a terminal" in caplog.text
        assert "1, 2" in caplog.text, "name the ids, so the caller can pass one"

    def test_a_real_terminal_still_gets_the_picker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tty path is unchanged: a picker even for one job (a deliberate `sw`
        behaviour, not an accident) — only the non-tty case may skip it."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [{"job_id": "777"}])
        launched: dict[str, object] = {}

        class _FakeApp:
            def __init__(self, **kwargs: object) -> None:
                launched.update(kwargs)
                self.return_code = 0

            def run(self, **kwargs: object) -> None:
                launched["ran"] = True

        import slurmwatch.tui as tui

        monkeypatch.setattr(tui, "SlurmwatchApp", _FakeApp)
        assert _auto_discover_job_id(SlurmwatchConfig(), interactive=True) is None
        assert launched.get("ran") is True


class TestLogBannerFollowsTheWriteCheck:
    """Round-3 nit: the `logging job … to <path>` line printed BEFORE anything was
    written, so an unwritable path announced success and was then followed by its
    own failure."""

    def test_an_unwritable_path_is_not_announced_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        ctx = TestForeignJob()._ctx(owner="youzhi")
        ctx.uid = 4242
        ctx.remote = False
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        unwritable = tmp_path / "nope-dir" / "run.jsonl"  # parent doesn't exist
        with caplog.at_level("ERROR", logger="slurmwatch"), pytest.raises(SystemExit) as exc:
            cli._run_headless("52211701_20", SlurmwatchConfig(), str(unwritable), "json")
        assert exc.value.code == 1
        assert "Cannot write log file" in caplog.text
        cap = capsys.readouterr()
        assert "logging job" not in cap.err, "announced a log it could not write"

    def test_a_writable_path_is_still_announced(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """The check must not swallow the banner on the normal path."""
        ctx = TestForeignJob()._ctx(owner="youzhi")
        ctx.uid = 4242
        ctx.remote = False
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr("asyncio.run", lambda coro: coro.close())
        target = tmp_path / "run.jsonl"
        cli._run_headless("52211701_20", SlurmwatchConfig(), str(target), "json")
        assert "logging job" in capsys.readouterr().err


class TestSamplingFloor:
    """SW-13: `--interval 0.001` — one misplaced character from `0.1` — was accepted
    in silence and ran at the old 0.05s floor, ~19 samples a second, indefinitely."""

    def _cfg(self, interval: float) -> SlurmwatchConfig:
        # requested_interval too, because main() sets it for every typed --interval
        # (and for SLURMWATCH_POLL_INTERVAL). Without it this helper described a
        # state the CLI cannot produce — a below-floor interval nobody asked for —
        # which is now the DEFAULT's case and deliberately silent.
        cfg = SlurmwatchConfig(poll_interval=interval, headless_interval=interval)
        cfg.requested_interval = interval
        return cfg

    def test_the_default_being_floored_is_not_announced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A notice must name something the reader asked for.

        The default poll interval (0.5) is below the sstat floor, so every off-node
        run opened with "slurmwatch: --interval 0.5 raised to 1s" — quoting a flag
        that was never typed. The clamp still happens; only the notice is gated.
        """
        cfg = SlurmwatchConfig()
        assert cfg.requested_interval is None, "nothing was asked for"
        cli._apply_sampling_floor(cfg, remote=True)
        assert cfg.poll_interval == 1.0, "the floor still applies"
        assert capsys.readouterr().err == ""

    def test_an_interval_asked_for_through_the_environment_is_announced(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SLURMWATCH_POLL_INTERVAL is a request as much as --interval is."""
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "0.2")
        cfg = SlurmwatchConfig.from_env()
        assert cfg.requested_interval == 0.2
        cli._apply_sampling_floor(cfg, remote=True)
        err = capsys.readouterr().err
        assert "0.2 raised to 1s" in err, err

    def test_on_node_floor(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = self._cfg(0.001)
        cli._apply_sampling_floor(cfg, remote=False)
        assert cfg.poll_interval == 0.1 and cfg.headless_interval == 0.1
        err = capsys.readouterr().err
        assert "raised to 0.1s" in err and "cgroup" in err, err

    def test_off_node_floor_is_higher(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Off the node every sample is a subprocess plus a slurmdbd query."""
        cfg = self._cfg(0.1)
        cli._apply_sampling_floor(cfg, remote=True)
        assert cfg.poll_interval == 1.0
        assert "sstat" in capsys.readouterr().err

    def test_a_reasonable_interval_is_left_alone_and_unannounced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = self._cfg(2.0)
        cli._apply_sampling_floor(cfg, remote=True)
        assert cfg.poll_interval == 2.0 and cfg.headless_interval == 2.0
        assert capsys.readouterr().err == ""

    def _as_the_cli_builds_it(self, interval: float) -> SlurmwatchConfig:
        """The config the way `main()` actually produces it: override, then clamp.

        The tests above construct SlurmwatchConfig(poll_interval=0.001) DIRECTLY, so
        they never went through the `config.clamp()` that main() applies right after
        the override — which is why they passed while the real path was silent. The
        clamp had already raised 0.001 to 0.1, so `_apply_sampling_floor` could not
        tell anything had happened.
        """
        cfg = SlurmwatchConfig()
        cfg.requested_interval = interval
        cfg.poll_interval = interval
        cfg.headless_interval = interval
        cfg.clamp()
        return cfg

    def test_the_on_node_raise_is_still_announced_after_the_clamp(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = self._as_the_cli_builds_it(0.001)
        assert cfg.poll_interval == 0.1, "the clamp already moved it"
        cli._apply_sampling_floor(cfg, remote=False)
        err = capsys.readouterr().err
        assert "raised to 0.1s" in err, "silence is the one outcome SW-13 forbids"
        assert "0.001" in err, "quote what was TYPED"

    def test_the_off_node_notice_quotes_the_typed_value_not_the_clamped_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = self._as_the_cli_builds_it(0.001)
        cli._apply_sampling_floor(cfg, remote=True)
        err = capsys.readouterr().err
        assert "--interval 0.001 raised to 1s" in err, err
        assert "0.1 raised" not in err, "it used to quote its own clamped value back"

    def test_an_interval_above_the_floor_is_still_silent_through_the_real_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = self._as_the_cli_builds_it(2.0)
        cli._apply_sampling_floor(cfg, remote=True)
        assert cfg.poll_interval == 2.0
        assert capsys.readouterr().err == ""

    def test_main_records_what_was_typed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through main() itself, because the helper above sets requested_interval by
        hand — so nothing would notice if the CLI stopped recording it."""
        seen: list[SlurmwatchConfig] = []
        monkeypatch.setattr(cli, "_run_once", lambda jid, config, fmt="": seen.append(config))
        monkeypatch.setattr(cli, "_job_id_without_step", lambda j: j)
        with contextlib.suppress(SystemExit):
            main(["12345", "--once", "--interval", "0.001"])
        assert seen, "main did not reach the once path"
        assert seen[0].requested_interval == 0.001, seen[0].requested_interval
        assert seen[0].poll_interval == 0.1, "and the clamp still applies"

    def test_once_applies_the_floor_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--once was the one measuring path that never called it, so the flag was
        honoured at a value the tool elsewhere calls pathological.

        Uses the REMOTE floor (1.0) with an interval the clamp leaves alone, so the
        assertion can only pass if this function applied it — an on-node 0.001 is
        already 0.1 by the time it gets here, which no amount of skipping changes.
        """
        seen: list[float] = []

        class _Collector:
            def __init__(self, ctx: object, config: SlurmwatchConfig) -> None:
                seen.append(config.poll_interval)

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

            async def next_snapshot(self) -> Any:
                raise asyncio.CancelledError

        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="cn1",
            hostname="cn1",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        ctx.remote = True
        ctx.hostname = "login"
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")  # stay on the local sstat path
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda j: (ctx, None))
        monkeypatch.setattr(cli, "_job_owner_differs", lambda c: False)
        monkeypatch.setattr(cli, "TelemetryCollector", _Collector)
        cfg = self._as_the_cli_builds_it(0.5)  # above the clamp, below the sstat floor
        assert cfg.poll_interval == 0.5, "the clamp leaves this one alone"
        with contextlib.suppress(BaseException):
            cli._run_once("1", cfg)
        assert seen == [1.0], f"the collector was handed {seen}, not the sstat floor"

    def test_the_once_hop_forwards_the_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both interactive hops forward it; this one dropped it, and the child is the
        process that takes the measurement — so the flag had no effect at all."""
        captured: list[list[str]] = []

        class _R:
            returncode = 0
            stdout = "{}\n"  # the hop passes the child's text through verbatim

        def _fake_run(cmd: list[str], **kw: Any) -> Any:
            captured.append(cmd)
            return _R()

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.delenv("SLURMWATCH_ON_NODE", raising=False)
        ctx = JobContext(
            job_id="7",
            username="u",
            partition="p",
            nodelist="cn1",
            hostname="login",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn1"],
            raw_job_id="7",
            remote=True,
        )
        cfg = self._as_the_cli_builds_it(5.0)
        cli._once_on_node(ctx, cfg, "json")
        inner = [c for c in captured if "--once" in c]
        assert inner, captured
        assert "--interval" in inner[-1] and "5" in inner[-1], inner[-1]

    def test_the_default_config_is_never_raised_on_the_node(self) -> None:
        cfg = SlurmwatchConfig()
        before = (cfg.poll_interval, cfg.headless_interval)
        cli._apply_sampling_floor(cfg, remote=False)
        assert (cfg.poll_interval, cfg.headless_interval) == before

    def test_the_config_floor_itself_moved_up(self) -> None:
        cfg = SlurmwatchConfig(poll_interval=0.001, headless_interval=0.001)
        cfg.clamp()
        assert cfg.poll_interval >= 0.1, "clamp() is the last line of defence"


class TestAnArrayRangeIsRewrittenNotRejected:
    """`squeue` prints `54222358_[1-9%3]` for a pending array, and sw refused it.

    Measured on a live queue: that id — bracketed range, `%N` throttle and all — is
    what the JOBID column holds for every unstarted array, so it is exactly what gets
    pasted. sw answered "'54222358_[1-9%3]' is not a job id" and then advised finding
    the id with `squeue -o '%i %j'`, which prints the same string: a closed loop. The
    range names no single task and an unstarted array has no per-task telemetry
    anyway, so the array's own job — its pending reason, request and queue position —
    is what the reader was after. Fourth instance of the SW-7 / RD-2 / SW-14 family.
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, job_id: str) -> str | None:
        seen: dict[str, str] = {}
        monkeypatch.setattr(cli, "_run_once", lambda jid, cfg, fmt="": seen.update(job_id=jid))
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        main([job_id, "--once"])
        return seen.get("job_id")

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("54222358_[1-9%3]", "54222358"),  # squeue's throttled form
            ("54471281_[0-15%12]", "54471281"),
            ("53737565_[0-25]", "53737565"),  # no throttle
            ("54113454_[9-15,20,22]", "54113454"),  # a discontinuous range
            ("54113454_[7]", "54113454"),  # one task left, still bracketed
        ],
    )
    def test_a_bracketed_range_attaches_to_its_array_job(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        given: str,
        expected: str,
    ) -> None:
        assert self._run(monkeypatch, given) == expected
        err = capsys.readouterr().err
        assert f"monitoring array job {expected}" in err, err
        # Say what to pass for ONE task, or the reader has to guess the form.
        assert f"{expected}_<task>" in err, err

    def test_the_notice_never_touches_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--once --json | jq` must still get only JSON."""
        self._run(monkeypatch, "54222358_[1-9%3]")
        assert capsys.readouterr().out == ""

    def test_a_single_array_task_is_untouched_and_silent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A started task HAS telemetry — rewriting it to the array would lose it."""
        assert self._run(monkeypatch, "54222358_7") == "54222358_7"
        assert capsys.readouterr().err == ""

    def test_a_bracket_that_is_not_a_range_is_still_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The rewrite must not turn any bracketed string into a job id."""
        from slurmwatch.cli import _job_id_without_array_range

        for bogus in ("54222358_[bogus]", "54222358_[]", "job_[1-9]", "54222358_[1-9"):
            assert _job_id_without_array_range(bogus) == bogus, bogus
        assert capsys.readouterr().err == ""

    def test_the_refusal_lists_the_form_it_now_accepts(self) -> None:
        """The enumeration in the error has to match what the parser takes."""
        from slurmwatch.cli import _JOB_ID_FORM

        assert _JOB_ID_FORM.match("54222358_[1-9%3]"), "an id squeue prints must pass"
        assert not _JOB_ID_FORM.match("54222358_[bogus]")


class TestStepIdIsRewrittenNotRejected:
    """SW-14: `slurmwatch 48819348.0` answered "Job 48819348.0 does not exist in the
    Slurm database" — false, since `sacct -j 48819348.0` prints it. `<job>.<step>` is
    the form BOTH sacct and `squeue -s` print, so it is what a user pastes.
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, job_id: str) -> tuple[str | None, str]:
        seen: dict[str, str] = {}
        monkeypatch.setattr(cli, "_run_once", lambda jid, cfg, fmt="": seen.update(job_id=jid))
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        main([job_id, "--once"])
        return seen.get("job_id"), ""

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("48819348.0", "48819348"),
            ("48819348.batch", "48819348"),
            ("48819348.extern", "48819348"),
            ("48819348_3.0", "48819348_3"),  # an array task's step keeps its task
            ("48819348+1.0", "48819348+1"),  # ...and a het component keeps its index
        ],
    )
    def test_a_step_id_attaches_to_its_job(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        given: str,
        expected: str,
    ) -> None:
        got, _ = self._run(monkeypatch, given)
        assert got == expected
        err = capsys.readouterr().err
        assert f"monitoring job {expected}" in err, err
        assert "across all its steps" in err

    def test_the_notice_never_touches_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--once --json | jq` must still get only JSON."""
        self._run(monkeypatch, "48819348.0")
        assert capsys.readouterr().out == ""

    def test_a_plain_job_id_is_untouched_and_silent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        got, _ = self._run(monkeypatch, "48819348")
        assert got == "48819348"
        assert capsys.readouterr().err == ""

    def test_an_array_task_is_not_mistaken_for_a_step(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        got, _ = self._run(monkeypatch, "48818945_2")
        assert got == "48818945_2"
        assert capsys.readouterr().err == ""

    def test_an_id_that_is_not_job_shaped_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Only rewrite what really is `<job>.<step>`; a garbage id must still reach
        the resolver and get the honest "does not exist" answer."""
        got, _ = self._run(monkeypatch, "notanumber.0")
        assert got == "notanumber.0"
        assert capsys.readouterr().err == ""


class TestPythonDashMEntryPoint:
    """`__main__.py` had ZERO coverage — and it is what the hop RUNS.

    Every relocation launches `[sys.executable, "-m", "slurmwatch", <job>, ...]`
    (by absolute interpreter path, so it resolves over the shared filesystem rather
    than depending on the compute node's PATH). If that entry point were broken, the
    console script would still work perfectly and every hop would fail — which is a
    login-node-only symptom, i.e. exactly the kind that shows up on someone else's
    cluster.
    """

    def test_module_exposes_the_same_main(self) -> None:
        import importlib

        from slurmwatch.cli import main as cli_main

        entry = importlib.import_module("slurmwatch.__main__")
        assert entry.main is cli_main

    def test_running_it_as_a_module_works(self) -> None:
        """End-to-end through the real interpreter, the way the hop invokes it."""
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-m", "slurmwatch", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0, out.stderr
        assert "slurmwatch" in (out.stdout + out.stderr).lower()


class TestHelpRendersAndSaysWhatTheToolAccepts:
    """`--help` was not rendered by any test, and it is the one invocation that must
    never fail.

    Proved live while writing this: adding the array-range form to the job_id help
    put a literal `%` in it, argparse interpolates `help % params`, and `--help`
    died with `ValueError: unsupported format character ']'`. Every documented form
    and knob is asserted here too, so the docs cannot quietly lose one — a knob
    nobody can discover is a knob that does not exist on a cluster that needs it.
    """

    def test_help_renders_at_all(self) -> None:
        """format_help() expands EVERY action's help string, which is the trap."""
        text = _build_parser().format_help()
        assert "job_id" in text and "--once" in text

    def test_no_help_string_breaks_percent_interpolation(self) -> None:
        """A stray % in any help= would raise here, as one did."""
        parser = _build_parser()
        for action in parser._actions:
            if action.help:
                parser._get_formatter()._expand_help(action)

    @pytest.mark.parametrize(
        "form",
        [
            "12345_3",  # array task
            "12345_[1-9%3]",  # a pending array's range, as squeue prints it (round 62)
            "12345.0",  # a step, as sacct / squeue -s print it (SW-14)
            "123+1",  # het component
        ],
    )
    def test_every_accepted_id_form_is_documented(self, form: str) -> None:
        """The parser takes these; a reader pasting one from squeue should see it here."""
        assert form in _build_parser().format_help(), form

    @pytest.mark.parametrize(
        "var", ["SLURMWATCH_NO_HOP", "SLURMWATCH_NO_SSH", "SLURMWATCH_HOP_TIMEOUT"]
    )
    def test_the_transport_knobs_are_discoverable(self, var: str) -> None:
        """The three a user on a DIFFERENT cluster reaches for: don't relocate, don't
        ssh, wait longer for a slow step. HOP_TIMEOUT was read, validated and clamped
        by the code while appearing in no help text at all."""
        assert var in _build_parser().format_help(), var


class TestAnUnsampledOffNodeReadingIsNotAMeasurement:
    """Off-node, `--once` published all-zero CPU and memory as if measured.

    Slurm samples accounting roughly every 30s, so a young job (or a site where sstat
    has nothing) has no sample yet and every metric reads 0. The plain-text summary has
    always said "usage not yet sampled by Slurm — try again shortly"; the machine
    payload emitted a full snapshot with `usage_ns: 0`, `limit_bytes: 0` and
    `source: "sstat"`. A right-sizing consumer reads that as "this job uses nothing"
    and acts on it by shrinking --mem and --cpus-per-task to the floor. Measured by
    running the real binary against a fake Slurm with no sstat at all: the human path
    said "not yet sampled" and the JSON on the same input said zero.
    """

    @staticmethod
    def _remote_ctx() -> JobContext:
        return JobContext(
            job_id="9",
            username="u",
            partition="p",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=64 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            job_start_time=1000.0,
            remote=True,
        )

    def _snapshot(self, monkeypatch: pytest.MonkeyPatch, sampled: bool) -> Any:
        from slurmwatch import slurm
        from slurmwatch.collector import TelemetryCollector

        monkeypatch.setattr(
            slurm,
            "resolve_remote_usage",
            lambda job_id, node_count=1: slurm.RemoteUsage(
                rss_bytes=0 if not sampled else 8 * 1024**3,
                cpu_seconds=0.0 if not sampled else 100.0,
                sampled=sampled,
            ),
        )
        return TelemetryCollector(self._remote_ctx())._collect_snapshot_sync()

    def test_the_snapshot_says_whether_it_measured_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._snapshot(monkeypatch, sampled=False).usage_sampled is False
        assert self._snapshot(monkeypatch, sampled=True).usage_sampled is True

    def test_an_on_node_snapshot_is_always_a_measurement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cgroup is read every sample, so there is no unsampled state to flag."""
        ctx = self._remote_ctx()
        ctx.remote = False
        from slurmwatch.collector import TelemetryCollector

        assert TelemetryCollector(ctx)._collect_snapshot_sync().usage_sampled is True

    def test_the_flag_reaches_json_and_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        snap = self._snapshot(monkeypatch, sampled=False)
        assert json.loads(snap.to_json())["usage_sampled"] is False
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        assert cells["usage_sampled"] == "0"

    def test_an_older_payload_without_the_flag_reads_as_sampled(self) -> None:
        """There is nothing to re-derive it from, and calling every old row unsampled
        would be its own lie — so absent means sampled, deliberately."""
        snap = _snap_with_gpus(0)
        payload = json.loads(snap.to_json())
        del payload["usage_sampled"]
        assert TelemetrySnapshot.from_dict(payload).usage_sampled is True

    @pytest.mark.asyncio
    async def test_once_emits_the_no_telemetry_shape_instead_of_zeros(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The whole point: the payload channel must not carry a zero measurement."""
        snap = self._snapshot(monkeypatch, sampled=False)

        class _C:
            job_ctx = self._remote_ctx()

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> Any:
                return snap

        with pytest.raises(SystemExit) as exc:
            await cli._once_loop(_C(), json_output=True)  # type: ignore[arg-type]
        assert exc.value.code == 1, "a script must not record this as a reading"
        payload = json.loads(capsys.readouterr().out)
        assert payload["telemetry_unavailable_reason"] == "usage_not_sampled"
        assert payload["telemetry_available"] is False
        # Nulls, not zeros — the distinction the whole class exists for.
        assert payload["cpu_percent"] is None
        assert payload["mem_working_set_bytes"] is None
        assert "30s" in payload["reason"], payload["reason"]

    @pytest.mark.asyncio
    async def test_a_sampled_reading_still_emits_the_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The complement: this must not swallow every off-node reading."""
        snap = self._snapshot(monkeypatch, sampled=True)

        class _C:
            job_ctx = self._remote_ctx()

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> Any:
                return snap

        await cli._once_loop(_C(), json_output=True)  # type: ignore[arg-type]
        payload = json.loads(capsys.readouterr().out)
        assert payload["memory"]["current_bytes"] == 8 * 1024**3
        assert payload["usage_sampled"] is True


class TestTheHeadlessLoopAlwaysYields:
    """The `--log` loop's signals are loop-based, so it must never stop yielding.

    `loop.add_signal_handler` replaces the default disposition with a callback the
    event loop has to run. A loop that stops yielding therefore becomes immune to
    SIGTERM/SIGHUP/SIGINT — not merely unresponsive — and this is the path that runs
    unattended for days. The dashboard's equivalent loop learned this the hard way
    (round 73); asserting it here keeps the guarantee structural instead of depending
    on every future branch remembering to await.
    """

    def test_the_loop_body_opens_with_an_unconditional_sleep(self) -> None:
        import inspect

        src = inspect.getsource(cli._headless_loop)
        body = src.split("while not shutdown_event.is_set():", 1)[1]
        lines = (ln.strip() for ln in body.splitlines())
        first = next(ln for ln in lines if ln and not ln.startswith("#"))
        assert first == "await asyncio.sleep(0)", first

    def test_the_signal_style_is_the_one_that_needs_the_yield(self) -> None:
        """If this ever moves to signal.signal, the yield above stops being load-bearing
        for killability (though it still is for responsiveness) — so pin which style is
        in use, and let a change here be a deliberate decision."""
        import inspect

        src = inspect.getsource(cli._headless_loop)
        assert "add_signal_handler" in src
        assert "signal.signal(" not in src


class TestASignalledLogRunSaysSoInItsExitCode:
    """A `--log` run stopped from outside used to be indistinguishable from one that
    finished because the job ended: both exited 0, and the log looks the same either
    way. Measured against a live job before the fix — SIGINT 0, SIGTERM 0, and SIGHUP
    **rc -1**, killed by the default action with no drain at all. The dashboard has
    reported 143/129/130 for exactly this reason since SW-26; this is its headless
    sibling, on the path most likely to be wrapped by a script.
    """

    @staticmethod
    def _collector(ended_after: int = 10**6) -> type:
        class _C:
            def __init__(self, *a: object, **k: object) -> None:
                self.job_ended = False
                self._n = 0

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> TelemetrySnapshot:
                self._n += 1
                if self._n > ended_after:
                    self.job_ended = True
                    raise asyncio.TimeoutError
                await asyncio.sleep(0.01)
                return _snap_with_gpus(0)

        return _C

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    @pytest.mark.parametrize(
        ("signame", "expected"),
        [("SIGINT", 130), ("SIGTERM", 143), ("SIGHUP", 129)],
    )
    async def test_each_handled_signal_reports_128_plus_itself(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        signame: str,
        expected: int,
    ) -> None:
        monkeypatch.setattr(cli, "TelemetryCollector", self._collector())
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        out = tmp_path / "s.jsonl"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "json"))
        await _wait_for_lines(out, 2)
        os.kill(os.getpid(), getattr(signal, signame))
        code = await asyncio.wait_for(task, timeout=10.0)
        assert code == expected
        # SIGHUP is the one that used to kill the process outright, so the graceful
        # tail matters: the run must still say it stopped, and name the reason.
        err = capsys.readouterr().err
        assert f"monitoring stopped ({signame})" in err, err
        for line in out.read_text().splitlines():
            if line.strip():
                json.loads(line)

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_code_reaches_the_PROCESS_not_just_the_caller(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The loop returning 143 is useless if the wiring drops it.

        Asserting on `_headless_loop`'s return value proves the loop worked out the
        code; what a script sees is `_run_headless` turning it into an exit. Deleting
        that one line left every test above passing.
        """
        out = tmp_path / "w.jsonl"
        monkeypatch.setattr("slurmwatch.cli.asyncio.run", lambda coro: (coro.close(), 143)[1])
        with pytest.raises(SystemExit) as exc:
            cli._run_headless("12345", SlurmwatchConfig(), str(out), "json")
        assert exc.value.code == 143

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_a_zero_from_the_loop_is_not_turned_into_an_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """...and the ordinary case must still fall through without raising."""
        out = tmp_path / "z.jsonl"
        monkeypatch.setattr("slurmwatch.cli.asyncio.run", lambda coro: (coro.close(), 0)[1])
        cli._run_headless("12345", SlurmwatchConfig(), str(out), "json")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_job_that_simply_ended_still_reports_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The complement: 128+n must not swallow the ordinary case.

        Without this the fix could report a signal for every exit and no test here
        would notice — a `--log` run in a batch script would then look failed on the
        happy path.
        """
        monkeypatch.setattr(cli, "TelemetryCollector", self._collector(ended_after=2))
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        out = tmp_path / "e.jsonl"
        code = await asyncio.wait_for(_headless_loop(ctx, cfg, str(out), "json"), timeout=10.0)
        assert code == 0


class TestConcurrentLogWriters:
    """SW-16: two `--log` writers on one path corrupted it two ways — the default
    "w" made both truncate and hold INDEPENDENT offsets (each overwriting the
    other's bytes mid-record), and even `--append` wasn't record-atomic, because a
    buffered write bigger than the buffer is several write() syscalls another
    appender can interleave with. A file that parses at the start and raises
    JSONDecodeError partway through defeats the point of --log.
    """

    def test_a_second_writer_is_refused_not_allowed_to_corrupt(self, tmp_path: Path) -> None:
        target = tmp_path / "shared.jsonl"
        first = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(first, str(target))
            second = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                with pytest.raises(SystemExit) as exc:
                    cli._claim_log_file(second, str(target))
                assert exc.value.code == 1
            finally:
                os.close(second)
        finally:
            os.close(first)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_refused_writer_does_not_truncate_the_first_ones_file(
        self, tmp_path: Path
    ) -> None:
        """O_TRUNC at open time happens BEFORE any lock, so the writer about to be
        refused still wiped the file — and the first writer, still at its own
        offset, then left a hole of NUL bytes that reads as one corrupt line. Found
        by running the report's own two-writer reproduction after the lock was
        added, so the lock alone was not the whole fix."""
        ctx = resolve_job_context("12345")
        target = tmp_path / "shared.jsonl"
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        first = asyncio.create_task(_headless_loop(ctx, cfg, str(target), ""))
        await _wait_for_lines(target, 3)
        before = target.read_text()

        # BOUNDED. _headless_loop only ever returns by raising, so awaiting a second
        # writer that is NOT refused never completes: the test hangs instead of
        # failing, and a lock regression would sit in CI until the job timeout rather
        # than reporting in seconds. Measured for real — a sweep that removed the lock
        # left a pytest process wedged for 13 hours.
        #
        # The conversion is not decoration: SystemExit is a BaseException, and asyncio
        # re-raises those OUT of the event loop instead of handing them to whoever
        # awaits the task, so `pytest.raises(SystemExit)` around a plain
        # `wait_for(_headless_loop(...))` never sees it and the run dies in the
        # runner. Turning it into a value inside the coroutine is what lets a Task
        # deliver it — and lets this assert the exit CODE, which it never checked.
        async def _second_writer() -> int:
            try:
                await _headless_loop(ctx, cfg, str(target), "")
            except SystemExit as exc:
                return int(exc.code or 0)
            raise AssertionError("the second truncating writer was not refused")

        assert await asyncio.wait_for(_second_writer(), timeout=10.0) == 1
        assert target.read_text().startswith(before), "the refused writer truncated it"
        first.cancel()
        # Bounded for the same reason the second writer is: a join that cannot
        # complete turns a regression into a hang, and this one DID hang a CI job
        # for 120s — the collector's teardown joined its own cancelled tasks without
        # a bound, so an executor thread that had already started held the whole
        # unwind open (fixed in collector.stop, tested there).
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(first, timeout=10.0)
        assert "\x00" not in target.read_text()
        for ln in target.read_text().splitlines():
            if ln.strip():
                json.loads(ln)

    def test_a_second_appender_proceeds_but_says_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Narrowed on the reporter's own round-47 measurement: two writers with
        --append gave "31 data rows from two writers, one header, 0 rows with a wrong
        field count", interleaved rather than clobbering. Refusing that forbids a
        useful pattern (one aggregate log, told apart by job_id) for no correctness
        gain — but the forgotten-logger case is common, so it is announced."""
        target = tmp_path / "shared.jsonl"
        first = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            cli._claim_log_file(first, str(target), append=True)
            second = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                with caplog.at_level("WARNING"):
                    cli._claim_log_file(second, str(target), append=True)  # must NOT exit
            finally:
                os.close(second)
        finally:
            os.close(first)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "appending alongside it" in msg, msg
        assert "job_id" in msg, "say how to tell the streams apart"

    def test_a_second_truncating_writer_is_still_refused(self, tmp_path: Path) -> None:
        """The case that cannot be made right: a truncating open destroys the other
        writer's data, and nothing recovers that."""
        target = tmp_path / "shared.jsonl"
        first = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(first, str(target), append=True)
            second = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                with pytest.raises(SystemExit) as exc:
                    cli._claim_log_file(second, str(target), append=False)
                assert exc.value.code == 1
            finally:
                os.close(second)
        finally:
            os.close(first)

    def test_the_refusal_names_the_way_out(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        target = tmp_path / "shared.jsonl"
        first = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(first, str(target))
            second = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                with caplog.at_level("ERROR"), pytest.raises(SystemExit):
                    cli._claim_log_file(second, str(target))
            finally:
                os.close(second)
        finally:
            os.close(first)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "TRUNCATE" in msg and "--append" in msg, msg

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_the_headless_loop_passes_append_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The narrowed rule is only narrowed if the flag reaches the claim: with
        `append` hardcoded False, a second appender is refused again."""
        seen: list[bool] = []
        monkeypatch.setattr(
            cli, "_claim_log_file", lambda fd, path, append=False: seen.append(append)
        )

        class _Ending:
            def __init__(self, ctx: object, config: object) -> None:
                self.job_ended = True

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

            async def next_snapshot(self) -> Any:
                # asyncio.TimeoutError, not the builtin: that is what wait_for
                # raises, and under Python 3.10 the two are DIFFERENT classes (they
                # were only unified in 3.11). Raising the builtin here sent the loop
                # down its OSError path on 3.10 — see the class below.
                raise asyncio.TimeoutError

        monkeypatch.setattr(cli, "TelemetryCollector", _Ending)
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        for append in (True, False):
            await asyncio.wait_for(
                _headless_loop(ctx, cfg, str(tmp_path / "m.jsonl"), "json", append=append),
                timeout=5.0,
            )
        assert seen == [True, False], seen

    def test_both_lock_mechanisms_are_taken(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`flock` does NOT exclude across nodes on this cluster's GPFS /home —
        measured between two nodes of one allocation: a lock held on midway3-0200 was
        granted AGAIN on beagle3-0009. A POSIX record lock (`lockf`) is refused there,
        correctly. But record locks are per-PROCESS, so lockf alone stops excluding two
        opens in one process, which flock does catch. Neither is a superset; take both.
        """
        calls: list[str] = []
        monkeypatch.setattr("fcntl.lockf", lambda fd, op: calls.append("lockf"))
        monkeypatch.setattr("fcntl.flock", lambda fd, op: calls.append("flock"))
        target = tmp_path / "shared.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(fd, str(target))
        finally:
            os.close(fd)
        assert calls == ["lockf", "flock"], calls

    @pytest.mark.parametrize("busy_mechanism", ["fcntl.lockf", "fcntl.flock"])
    def test_a_busy_answer_from_either_mechanism_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy_mechanism: str
    ) -> None:
        def _busy(fd: int, op: int) -> None:
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

        monkeypatch.setattr("fcntl.lockf", lambda fd, op: None)
        monkeypatch.setattr("fcntl.flock", lambda fd, op: None)
        monkeypatch.setattr(busy_mechanism, _busy)
        target = tmp_path / "shared.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            with pytest.raises(SystemExit) as exc:
                cli._claim_log_file(fd, str(target))
            assert exc.value.code == 1
        finally:
            os.close(fd)

    def test_an_unsupported_mechanism_falls_through_to_the_other(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A filesystem whose F_SETLK is unsupported must not cost us today's flock
        protection — nor stop the run, which is the pre-existing contract."""
        taken: list[str] = []

        def _unsupported(fd: int, op: int) -> None:
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr("fcntl.lockf", _unsupported)
        monkeypatch.setattr("fcntl.flock", lambda fd, op: taken.append("flock"))
        target = tmp_path / "shared.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(fd, str(target))  # must not exit
        finally:
            os.close(fd)
        assert taken == ["flock"]

    def test_neither_mechanism_supported_still_logs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _unsupported(fd: int, op: int) -> None:
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr("fcntl.lockf", _unsupported)
        monkeypatch.setattr("fcntl.flock", _unsupported)
        target = tmp_path / "shared.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(fd, str(target))  # a lockless FS is not a refusal
        finally:
            os.close(fd)

    def test_the_lock_is_released_with_the_fd(self, tmp_path: Path) -> None:
        """A crashed or finished run must not leave the path unusable."""
        target = tmp_path / "shared.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        cli._claim_log_file(fd, str(target))
        os.close(fd)
        again = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(again, str(target))  # must not exit
        finally:
            os.close(again)

    def test_only_a_regular_file_is_claimed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--log /dev/stdout` is how the node switcher streams a remote node's
        snapshots back. A lock on a pipe/tty succeeds but protects nothing, and when
        /dev/stdout resolves to the PARENT's file it would refuse the switcher
        outright — so the guard's whole job is to not make the call. Asserting on the
        absence of the syscall, since that is the entire observable behaviour."""
        locked: list[int] = []
        monkeypatch.setattr("fcntl.flock", lambda fd, op: locked.append(fd))

        read_fd, write_fd = os.pipe()
        try:
            cli._claim_log_file(write_fd, "/dev/stdout")
            assert locked == [], "a pipe must not be locked"
        finally:
            os.close(read_fd)
            os.close(write_fd)

        target = tmp_path / "real.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            cli._claim_log_file(fd, str(target))
            assert locked == [fd], "a regular file must be locked"
        finally:
            os.close(fd)

    def test_a_record_is_one_write_syscall(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The atomicity guarantee: O_APPEND plus ONE write() per record means a
        line is either whole or absent. Two syscalls per record would let another
        appender splice itself into the middle."""
        calls: list[bytes] = []
        real = os.write

        def _counting(fd: int, payload: bytes) -> int:
            calls.append(payload)
            return real(fd, payload)

        target = tmp_path / "one.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            monkeypatch.setattr(os, "write", _counting)
            cli._write_record(fd, b'{"a": 1}\n')
        finally:
            monkeypatch.undo()
            os.close(fd)
        assert len(calls) == 1, calls
        assert target.read_text() == '{"a": 1}\n'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_two_appenders_produce_only_whole_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end-to-end shape the report measured. The lock normally refuses the
        second writer outright, so bypass it here — that is the real case of a
        filesystem that cannot lock (some NFS mounts), where record atomicity is the
        only thing standing between two appenders and a corrupt file."""
        monkeypatch.setattr(cli, "_claim_log_file", lambda fd, path, append=False: None)
        ctx = resolve_job_context("12345")
        target = tmp_path / "shared.jsonl"
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        async def _run() -> None:
            task = asyncio.create_task(_headless_loop(ctx, cfg, str(target), "", append=True))
            await _wait_for_lines(target, 8)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await asyncio.gather(_run(), _run())
        lines = [ln for ln in target.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 8, lines
        for ln in lines:
            json.loads(ln)  # a spliced record raises here — the reported symptom
            assert ln.count('"timestamp"') == 1, "two records share a line"


class TestEnvKnobsMatchTheFlags:
    """SW-17: four knobs were validated less strictly from the environment than from
    the command line — and env vars are what a site module file or a `.bashrc`
    carried between clusters sets, so a value that was right on one cluster arrives
    silently on the next."""

    @pytest.mark.parametrize(
        ("var", "value"),
        [
            ("SLURMWATCH_POLL_INTERVAL", "-5"),
            ("SLURMWATCH_POLL_INTERVAL", "0"),
            ("SLURMWATCH_HEADLESS_INTERVAL", "-1"),
            ("SLURMWATCH_HISTORY_SECONDS", "-100"),
            ("SLURMWATCH_HISTORY_SECONDS", "0"),
            ("SLURMWATCH_MOUSE", "7"),
            ("SLURMWATCH_MOUSE", "maybe"),
        ],
    )
    def test_a_value_the_cli_would_reject_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, var: str, value: str
    ) -> None:
        monkeypatch.setenv(var, value)
        with pytest.raises(ValueError, match=var):
            SlurmwatchConfig.from_env()

    def test_the_message_names_the_variable_the_value_and_the_expectation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The house style for env errors, which eight other knobs already met."""
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "-5")
        with pytest.raises(ValueError) as exc:
            SlurmwatchConfig.from_env()
        text = str(exc.value)
        assert "SLURMWATCH_POLL_INTERVAL" in text and "-5" in text and "positive" in text

    def test_validation_runs_before_the_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clamp() raises a negative interval to the floor, so validating after it
        could never see what the user actually set — the reason -5 was accepted."""
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "-5")
        with pytest.raises(ValueError):
            SlurmwatchConfig.from_env()

    @pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False), ("true", True)])
    def test_a_good_mouse_value_still_works(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_MOUSE", value)
        cfg = SlurmwatchConfig.from_env()
        assert cfg.mouse is expected
        assert cli._mouse_enabled(cfg) is expected

    def test_a_legitimate_extreme_is_still_clamped_not_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The floor/ceiling behaviour is a safety net for big-but-sane values; only
        nonsense (negative, zero) is an error."""
        monkeypatch.setenv("SLURMWATCH_HISTORY_SECONDS", "999999999")
        monkeypatch.setenv("SLURMWATCH_POLL_INTERVAL", "0.001")
        cfg = SlurmwatchConfig.from_env()
        assert cfg.history_seconds == 86_400
        assert cfg.poll_interval == 0.1  # the SW-13 floor

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_an_unusable_format_is_reported_not_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SLURMWATCH_FORMAT is load-bearing (=csv really produces CSV), so an
        unusable value must not be dropped in SILENCE — which is what this
        degradation branch did. Still non-fatal here: a stale variable from a site
        module file shouldn't kill `sw $JOBID | tee`, so it emits the default and
        says so on stderr, leaving stdout a clean data stream."""
        monkeypatch.setenv("SLURMWATCH_FORMAT", "xml")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        args = argparse.Namespace(once=False, log=None, format="", json=False)
        with caplog.at_level("WARNING", logger="slurmwatch"):
            cli._run_interactive("12345", SlurmwatchConfig(), args)
        # One shape for the whole family: Ignoring VAR='v' (reason); using X.
        assert "Ignoring SLURMWATCH_FORMAT='xml'" in caplog.text, caplog.text
        assert "using csv" in caplog.text
        out = capsys.readouterr().out
        assert out.startswith("timestamp,"), out[:80]

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_a_usable_format_is_honoured_silently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_FORMAT", "json")
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        args = argparse.Namespace(once=False, log=None, format="", json=False)
        with caplog.at_level("WARNING", logger="slurmwatch"):
            cli._run_interactive("12345", SlurmwatchConfig(), args)
        assert "SLURMWATCH_FORMAT" not in caplog.text
        json.loads(capsys.readouterr().out.strip().splitlines()[0])


class TestToleratedEnvValuesSayWhatTheyDid:
    """Round 17 states SW-17's complaint precisely: the problem was never "always
    reject" — `SLURMWATCH_HOP_TIMEOUT`'s tolerance is documented and defensible —
    it was falling back SILENTLY. These knobs are set far from where they take
    effect (a site module file, a .bashrc carried between clusters), so the person
    who set the value is not the person reading the output."""

    def test_a_bad_hop_timeout_reports_the_default_it_used(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "garbage")
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert cli._hop_connect_timeout() == 10  # still tolerant
        assert "SLURMWATCH_HOP_TIMEOUT" in caplog.text
        assert "not a number" in caplog.text and "10s default" in caplog.text

    def test_an_out_of_range_hop_timeout_reports_the_clamp(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "-1")
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert cli._hop_connect_timeout() == 2
        assert "outside the 2-120s range" in caplog.text and "2s" in caplog.text

    def test_a_usable_hop_timeout_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "30")
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert cli._hop_connect_timeout() == 30
        assert caplog.text == ""

    @pytest.mark.parametrize(
        ("var", "reader", "expected"),
        [
            ("SLURMWATCH_NO_HOP", "_env_disables_hop", "hop disabled"),
            ("SLURMWATCH_NO_SSH", "_env_disables_ssh", "ssh disabled"),
        ],
    )
    def test_an_unparseable_transport_toggle_says_which_way_it_read_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        var: str,
        reader: str,
        expected: str,
    ) -> None:
        """`NO_HOP=flase` disables the hop — the safe reading, and the one a user
        would never guess from silence."""
        monkeypatch.setenv(var, "flase")
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert getattr(cli, reader)() is True
        assert var in caplog.text and expected in caplog.text

    def test_a_stale_variable_is_reported_once_not_per_frame(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """These readers run per frame on some paths; one stale setting must not
        become a stream of identical lines."""
        monkeypatch.setenv("SLURMWATCH_HOP_TIMEOUT", "garbage")
        with caplog.at_level("WARNING", logger="slurmwatch"):
            for _ in range(5):
                cli._hop_connect_timeout()
        assert caplog.text.count("SLURMWATCH_HOP_TIMEOUT") == 1

    def test_a_good_boolean_is_still_honoured_both_ways(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="slurmwatch"):
            monkeypatch.setenv("SLURMWATCH_NO_HOP", "0")
            assert cli._env_disables_hop() is False
            monkeypatch.setenv("SLURMWATCH_NO_HOP", "true")
            assert cli._env_disables_hop() is True
        assert caplog.text == ""


class TestDegradedSummaryCarriesTheAdvisory:
    """SW-18: the CPU-underuse advisory lived only in the TUI, so the degraded
    plain-text summary printed `~1.0 of 8 cores` and said nothing about it — and the
    readers of that view are exactly the ones who CANNOT get the live dashboard (a
    cluster that forbids step creation, or a redirect), which is when "you asked for
    8x what you use" is most worth saying."""

    def _summary(
        self,
        capsys: pytest.CaptureFixture[str],
        *,
        cores: int,
        busy: float,
        limit: int = 400 * 1024**2,
        used: int = 8 * 1024**2,
    ) -> str:
        from slurmwatch.model import CpuMetrics, MemoryMetrics

        ctx = TestForeignJob()._ctx(owner="youzhi")
        ctx.cpus_allocated = cores
        snap = TelemetrySnapshot(
            timestamp=0.0,
            job_id=ctx.job_id,
            step_id=None,
            hostname="midway2-0300",
            elapsed_seconds=53,
            cpu=CpuMetrics(
                cores_allocated=cores,
                usage_ns=53_000_000_000,
                usage_percent=busy / cores * 100,
                effective_cores=busy,
            ),
            memory=MemoryMetrics(
                current_bytes=used,
                limit_bytes=limit,
                peak_bytes=used,
                usage_percent=used / limit * 100,
                oom_guard_warning=False,
                oom_guard_critical=False,
                working_set_bytes=used,
                source="sstat",
                cache_measured=False,
            ),
            gpus=[],
        )
        cli._print_remote_summary(ctx, snap, SlurmwatchConfig())
        return capsys.readouterr().out

    def test_an_underused_job_is_told_so(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._summary(capsys, cores=8, busy=1.0)
        assert "only ~1.0 of 8 cores" in out, out
        assert "--cpus-per-task" in out and "schedule faster" in out

    def test_a_well_used_job_gets_no_advice(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._summary(capsys, cores=8, busy=7.0)
        assert "Advice" not in out, out

    def test_a_single_core_job_is_never_called_underused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert "Advice" not in self._summary(capsys, cores=1, busy=0.0)

    def test_the_wording_matches_the_dashboards(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Both surfaces end with the same shared sentence, so they cannot drift."""
        from slurmwatch.model import CPU_UNDERUSE_ADVICE

        assert CPU_UNDERUSE_ADVICE in self._summary(capsys, cores=8, busy=1.0)

    def test_the_memory_line_uses_the_limits_own_unit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SW-4's second sighting: this renderer had its own :.1f GiB, so a
        --mem=400M job read `peak 0.0 GiB / 0.4 GiB` here long after the gauge was
        fixed."""
        out = self._summary(capsys, cores=8, busy=1.0)
        assert "peak 8.0 / 400 MiB" in out, out  # one unit, on the limit, as the gauge does
        assert "GiB" not in out.split("CPU")[0]

    def test_a_tens_of_gib_job_still_reads_in_gib(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._summary(capsys, cores=8, busy=1.0, limit=64 * 1024**3, used=12 * 1024**3)
        assert "peak 12 / 64 GiB" in out, out


class TestSilentTransportDowngradeIsAnnounced:
    """Round 25 measured the TUI never attempting the hop while a hand-run
    `srun --overlap` worked four times out of four — and nothing said why. A valid
    `SLURMWATCH_NO_HOP` (from an earlier experiment, a site module file, a .bashrc
    carried between clusters — the SW-17 theme) is honoured correctly, but silently,
    so the dashboard serves sstat-quality data while a working transport sits
    unused."""

    def _ctx(self) -> JobContext:
        return JobContext(
            job_id="12345",
            username="u",
            partition="gpu",
            nodelist="cn007",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=1024,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn007"],
            raw_job_id="12345",
            remote=True,
        )

    def test_the_opt_out_says_what_it_costs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        monkeypatch.delenv("SLURMWATCH_ON_NODE", raising=False)
        args = _build_parser().parse_args(["12345"])
        with caplog.at_level("WARNING", logger="slurmwatch"):
            outcome = cli._hop_to_compute_node(self._ctx(), args)
        assert outcome == cli._HOP_DECLINED_POLICY
        assert "SLURMWATCH_NO_HOP" in caplog.text
        assert "sstat" in caplog.text, "name the cost, not just the setting"

    def test_the_relaunched_child_is_not_nagged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The hop sets NO_HOP on its own child; warning there would put a line on
        every on-node dashboard, about a decision slurmwatch made itself."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        monkeypatch.setenv("SLURMWATCH_ON_NODE", "1")
        args = _build_parser().parse_args(["12345"])
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert cli._hop_to_compute_node(self._ctx(), args) == cli._HOP_DECLINED_POLICY
        assert caplog.text == ""


class TestCgroupAdviceIsCopyPasteable:
    """SW-22: the advice printed `srun --jobid X --overlap slurmwatch`, which fails
    both ways — `slurmwatch` is not on the compute node's PATH (execve(): No such
    file or directory), and with no job id the inner process auto-discovers, finds
    nothing and exits 1. `_hop_to_compute_node` already avoids both, by absolute
    interpreter path and an explicit id; only this string didn't."""

    def _advice(self, monkeypatch: pytest.MonkeyPatch, job_id: str) -> str:
        from slurmwatch.exceptions import CgroupNotFoundError

        records: list[str] = []
        monkeypatch.setattr(
            cli.logger, "error", lambda msg, *a: records.append(str(msg) % a if a else str(msg))
        )
        with pytest.raises(SystemExit):
            cli._die_on_resolve_error(CgroupNotFoundError("no cgroup"), job_id)
        return "\n".join(records)

    def test_it_names_this_interpreter_not_a_bare_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advice = self._advice(monkeypatch, "12345")
        assert f"{sys.executable} -m slurmwatch" in advice, advice
        assert "--overlap slurmwatch" not in advice, "bare name depends on the node's PATH"

    def test_it_passes_the_job_id_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        advice = self._advice(monkeypatch, "12345")
        assert advice.rstrip().endswith("12345"), advice

    def test_an_array_task_keeps_its_task_for_slurmwatch_but_not_for_srun(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`srun --jobid=` rejects "12345_3"; slurmwatch itself wants exactly that."""
        advice = self._advice(monkeypatch, "12345_3")
        assert "--jobid=12345 " in advice, advice
        assert advice.rstrip().endswith("12345_3"), advice

    def test_a_het_component_is_reduced_for_srun_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        advice = self._advice(monkeypatch, "12345+1")
        assert "--jobid=12345 " in advice, advice
        assert advice.rstrip().endswith("12345+1"), advice


class TestOnceReadsTheNodeNotSstat:
    """SW-23: `--once`/`--log` never hopped, so the machine-readable path — the one
    right-sizing decisions are made from — reported ~0.1 of 8 cores for a job
    saturating all 8. Measured here on a real R PSOCK job: on-node 5.90 of 6 cores,
    sstat 0.00 of 6. The tty gate that (correctly) guards the interactive TUI does
    not apply: capturing a subprocess's stdout needs no terminal."""

    def _ctx(self) -> JobContext:
        return JobContext(
            job_id="12345_3",
            username="u",
            partition="gpu",
            nodelist="cn007",
            hostname="login-01",
            cpus_allocated=8,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn007"],
            raw_job_id="12348",
            remote=True,
        )

    def test_it_runs_this_interpreter_on_the_jobs_node(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: dict[str, Any] = {}

        class _R:
            returncode = 0
            stdout = '{"job_id": "12345_3", "remote": false}\n'
            stderr = ""

        def _run(cmd: list[str], **kw: Any) -> Any:
            seen["cmd"] = cmd
            seen["env"] = kw.get("env", {})
            return _R()

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setattr("subprocess.run", _run)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.delenv("SLURMWATCH_ON_NODE", raising=False)
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is True

        cmd = seen["cmd"]
        assert "--overlap" in cmd and "--jobid=12348" in cmd, cmd
        assert sys.executable in cmd and "-m" in cmd and "slurmwatch" in cmd
        assert "12345_3" in cmd, "the inner process needs the id the user asked about"
        assert "--gres=none" in cmd, "must not contend for the job's GPUs to read a counter"
        # No second hop from the child.
        assert seen["env"]["SLURMWATCH_ON_NODE"] == "1"
        assert seen["env"]["SLURMWATCH_NO_HOP"] == "1"
        # The child's snapshot is passed through untouched.
        assert capsys.readouterr().out == _R.stdout

    def test_run_once_actually_takes_the_on_node_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wiring, not just the helper: `--once` on a remote job must consult it
        BEFORE falling back to the sstat collector."""
        ctx = self._ctx()
        ctx.uid = 4242
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)  # our own job
        monkeypatch.setattr(cli, "_once_on_node", lambda *a: True)

        def _no_collector(*_a: Any, **_k: Any) -> None:
            raise AssertionError("fell back to sstat with the on-node reading available")

        monkeypatch.setattr(cli, "TelemetryCollector", _no_collector)
        cli._run_once("12345_3", SlurmwatchConfig(), fmt="json")

    def test_run_once_falls_back_when_the_node_is_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = self._ctx()
        ctx.uid = 4242
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)
        monkeypatch.setattr(cli, "_once_on_node", lambda *a: False)
        built: list[bool] = []

        def _collector(*_a: Any, **_k: Any) -> Any:
            built.append(True)
            raise RuntimeError("stop here")

        monkeypatch.setattr(cli, "TelemetryCollector", _collector)
        with pytest.raises(RuntimeError):
            cli._run_once("12345_3", SlurmwatchConfig(), fmt="json")
        assert built == [True], "sstat must remain the floor"

    def test_no_terminal_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: this path captures stdout, so the tty gate must not
        apply to it the way it does to the TUI."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")

        class _R:
            returncode = 0
            stdout = "{}\n"
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is True

    @pytest.mark.parametrize(("rc", "out"), [(1, "{}\n"), (0, ""), (0, "   \n")])
    def test_a_failed_hop_falls_back_to_sstat(
        self, monkeypatch: pytest.MonkeyPatch, rc: int, out: str
    ) -> None:
        """Today's behaviour is the floor: a step that can't be created, or a child
        that produced nothing, must leave the caller to its sstat reading."""

        class _R:
            returncode = rc
            stdout = out
            stderr = ""

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is False

    def test_the_opt_out_and_the_child_marker_are_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is False
        monkeypatch.delenv("SLURMWATCH_NO_HOP")
        monkeypatch.setenv("SLURMWATCH_ON_NODE", "1")
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is False

    def test_no_srun_means_no_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is False

    def test_the_caveat_names_what_is_actually_at_risk(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The old wording pointed at working-set fidelity and GPU while sitting
        directly under a CPU figure that can be 100x low."""
        from slurmwatch.model import CpuMetrics, MemoryMetrics

        snap = TelemetrySnapshot(
            timestamp=0.0,
            job_id="12345",
            step_id=None,
            hostname="cn007",
            elapsed_seconds=10,
            cpu=CpuMetrics(cores_allocated=8, usage_ns=1, usage_percent=1.0, effective_cores=0.1),
            memory=MemoryMetrics(
                current_bytes=1024,
                limit_bytes=4 * 1024**3,
                peak_bytes=1024,
                usage_percent=1.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
                source="sstat",
                cache_measured=False,
            ),
            gpus=[],
            remote=True,
        )
        cli._print_remote_summary(self._ctx(), snap, SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "tracked process tree" in out, out
        assert "detached workers" in out and "cpu and memory" in out
        # And no advisory on a figure that can be 100x low (SW-23 x SW-18).
        assert "Advice" not in out


class TestTheMonitorStepIsAnnounced:
    """Creating a step is not free: while one lives, a plain `srun` in the same
    allocation is refused ("Requested nodes are busy") — measured on this cluster,
    and neither `--overlap` nor `--exact -c1` changes it. SW-23's hop is still worth
    it (the alternative reads 0.00 of 6 cores on an R multisession job), but `--once`
    is the SCRIPTED path, so a loop creates one step per call and the caller has to
    be told it is happening."""

    def _ctx(self) -> JobContext:
        return JobContext(
            job_id="12345",
            username="u",
            partition="gpu",
            nodelist="cn007",
            hostname="login-01",
            cpus_allocated=8,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn007"],
            raw_job_id="12345",
            remote=True,
        )

    def test_it_says_a_step_is_being_created_and_how_to_decline(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _R:
            returncode = 0
            stdout = "{}\n"
            stderr = ""

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.setattr(cli, "_MONITOR_STEP_NOTED", False)
        with caplog.at_level("WARNING", logger="slurmwatch"):
            cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json")
        assert "monitor step on cn007" in caplog.text, caplog.text
        assert "SLURMWATCH_NO_HOP=1" in caplog.text, "name the opt-out"
        assert "srun" in caplog.text and "refused" in caplog.text, "name the cost"

    def test_it_is_said_once_per_process(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _R:
            returncode = 0
            stdout = "{}\n"
            stderr = ""

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)
        monkeypatch.setattr(cli, "_MONITOR_STEP_NOTED", False)
        with caplog.at_level("WARNING", logger="slurmwatch"):
            for _ in range(3):
                cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json")
        assert caplog.text.count("monitor step on") == 1

    def test_declining_the_hop_creates_no_step_and_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _no_run(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("created a step despite the opt-out")

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")
        monkeypatch.setattr("subprocess.run", _no_run)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        monkeypatch.setattr(cli, "_MONITOR_STEP_NOTED", False)
        with caplog.at_level("WARNING", logger="slurmwatch"):
            assert cli._once_on_node(self._ctx(), SlurmwatchConfig(), "json") is False
        assert caplog.text == ""


class TestNotYetSampledVersusNeverSampled:
    """`JobAcctGatherType=none` (Slurm's default when unset) means sstat reports
    nothing for a running job, ever — so "try again shortly" is a false promise."""

    @staticmethod
    def _ctx() -> JobContext:
        return JobContext(
            job_id="12345",
            username="u",
            partition="p",
            nodelist="cn007",
            hostname="login1",
            cpus_allocated=8,
            mem_limit_bytes=4 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            remote=True,
        )

    @staticmethod
    def _unsampled_snapshot() -> TelemetrySnapshot:
        from slurmwatch.model import CpuMetrics, MemoryMetrics

        return TelemetrySnapshot(
            timestamp=0.0,
            job_id="12345",
            step_id=None,
            hostname="cn007",
            elapsed_seconds=10,
            cpu=CpuMetrics(cores_allocated=8, usage_ns=0, usage_percent=0.0, effective_cores=0.0),
            memory=MemoryMetrics(
                current_bytes=0,
                limit_bytes=4 * 1024**3,
                peak_bytes=0,
                usage_percent=0.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
                source="sstat",
                cache_measured=False,
            ),
            gpus=[],
            remote=True,
        )

    def test_says_why_when_the_cluster_gathers_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: True)
        cli._print_remote_summary(self._ctx(), self._unsampled_snapshot(), SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "JobAcctGatherType=none" in out, out
        assert "try again shortly" not in out, "there is nothing to wait for"
        assert "ON the" in out and "compute node" in out, "point somewhere that works"

    def test_keeps_the_timing_line_where_a_sample_really_is_coming(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: False)
        cli._print_remote_summary(self._ctx(), self._unsampled_snapshot(), SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "not yet sampled by Slurm" in out
        assert "JobAcctGatherType" not in out

    def test_a_sampled_reading_says_neither(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: True)
        snap = self._unsampled_snapshot()
        snap.memory.current_bytes = 1024
        cli._print_remote_summary(self._ctx(), snap, SlurmwatchConfig())
        out = capsys.readouterr().out
        assert "JobAcctGatherType" not in out and "not yet sampled" not in out


class TestAppendNeverMisattributesAColumn:
    """SW-25 / round 39: the schema-drift path warned, then appended positionally
    anyway — header 25 columns, rows 27, so every named field after the insertion
    point read a value belonging to a different field. `csv.DictReader` reported it
    as harmless overflow into `None` while `mem_percent` quietly returned the wrong
    number, for exactly the half of the file written after the append."""

    @staticmethod
    def _older_log(
        tmp_path: Path, drop: list[str], rows: int = 2
    ) -> tuple[Path, list[str], list[str]]:
        from slurmwatch.model import TelemetrySnapshot

        current = TelemetrySnapshot.csv_header(0)
        old_header = [c for c in current if c not in drop]
        assert len(old_header) == len(current) - len(drop), "drop names must exist"
        log = tmp_path / "old.csv"
        body = ",".join(old_header) + "\n"
        log.write_text(body)
        return log, old_header, current

    def test_rows_are_written_in_the_files_column_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from slurmwatch.cli import _conform_csv_row, _csv_append_layout

        drop = ["cpu_peak_effective_cores", "mem_working_set_percent"]
        log, old_header, current = self._older_log(tmp_path, drop)
        cmap = _csv_append_layout(str(log), "excel", 0, 0)
        capsys.readouterr()
        assert cmap is not None, "a differing header must produce a mapping"
        row = [f"v{i}" for i in range(len(current))]
        conformed = _conform_csv_row(row, cmap)
        assert len(conformed) == len(old_header), "one width for the whole file"
        truth = dict(zip(current, row, strict=True))
        assert dict(zip(old_header, conformed, strict=True)) == {c: truth[c] for c in old_header}, (
            "every value under its own heading"
        )

    def test_a_dictreader_sees_no_overflow_and_no_shift(self, tmp_path: Path) -> None:
        """The reporter's own check, which is what a consumer actually runs."""
        import csv as csv_mod

        from slurmwatch.cli import _conform_csv_row, _csv_append_layout

        drop = ["cpu_peak_effective_cores", "mem_working_set_percent"]
        log, old_header, current = self._older_log(tmp_path, drop)
        row = [f"v{i}" for i in range(len(current))]
        truth = dict(zip(current, row, strict=True))
        old_row = [truth[c] for c in old_header]
        with log.open("a", newline="") as f:
            csv_mod.writer(f).writerow(old_row)
        cmap = _csv_append_layout(str(log), "excel", 0, 0)
        assert cmap is not None
        with log.open("a", newline="") as f:
            csv_mod.writer(f).writerow(_conform_csv_row(row, cmap))
        with log.open() as f:
            assert {len(r) for r in csv_mod.reader(f)} == {len(old_header)}
        with log.open() as f:
            parsed = list(csv_mod.DictReader(f))
        assert len(parsed) == 2
        assert not any(None in r for r in parsed), "no overflow columns"
        assert parsed[0] == parsed[1], "the appended row matches the file's own layout"
        assert parsed[1]["mem_percent"] == truth["mem_percent"]

    def test_a_matching_header_needs_no_mapping(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from slurmwatch.cli import _csv_append_layout
        from slurmwatch.model import TelemetrySnapshot

        log = tmp_path / "cur.csv"
        log.write_text(",".join(TelemetrySnapshot.csv_header(0)) + "\n")
        assert _csv_append_layout(str(log), "excel", 0, 0) is None
        assert capsys.readouterr().err == ""

    def test_a_column_the_file_has_and_we_dropped_is_blank_not_shifted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The other direction of drift: the file is WIDER than this build writes."""
        from slurmwatch.cli import _conform_csv_row, _csv_append_layout
        from slurmwatch.model import TelemetrySnapshot

        current = TelemetrySnapshot.csv_header(0)
        wider = [*current[:5], "retired_column", *current[5:]]
        log = tmp_path / "wide.csv"
        log.write_text(",".join(wider) + "\n")
        cmap = _csv_append_layout(str(log), "excel", 0, 0)
        err = capsys.readouterr().err
        assert "retired_column" in err and "blank" in err
        assert cmap is not None
        row = [f"v{i}" for i in range(len(current))]
        conformed = _conform_csv_row(row, cmap)
        assert len(conformed) == len(wider)
        assert conformed[5] == "", "the retired column is blank"
        assert (
            dict(zip(wider, conformed, strict=True))["gpu_count"]
            == dict(zip(current, row, strict=True))["gpu_count"]
        ), "nothing after it shifted"

    def test_a_short_row_is_padded_not_truncated(self) -> None:
        """Defensive: a mapping index past the row's end blanks rather than raising,
        so one odd frame can never abort a long-running log."""
        from slurmwatch.cli import _conform_csv_row

        assert _conform_csv_row(["a"], [0, 5, None]) == ["a", "", ""]

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_headless_writer_actually_applies_the_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End to end through --log --append, which is where the damage happened: the
        helpers being right is no use if the writer still appends positionally."""
        import csv as csv_mod

        import slurmwatch.cli as climod
        from slurmwatch.model import TelemetrySnapshot

        snap = _snap_with_gpus(2)
        current = TelemetrySnapshot.csv_header(2)
        drop = ["cpu_peak_effective_cores", "mem_working_set_percent"]
        old_header = [c for c in current if c not in drop]
        assert len(old_header) == len(current) - 2
        truth = dict(zip(current, snap.to_csv_row(2), strict=True))
        log = tmp_path / "older.csv"
        log.write_text(",".join(old_header) + "\n" + ",".join(truth[c] for c in old_header) + "\n")

        class _OneShot:
            def __init__(self, job_ctx: object, config: object) -> None:
                self.job_ended = False

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> TelemetrySnapshot:
                self.job_ended = True
                return snap

        monkeypatch.setattr(climod, "TelemetryCollector", _OneShot)
        climod._run_headless("12345", SlurmwatchConfig(), str(log), append=True)

        with log.open() as f:
            assert {len(r) for r in csv_mod.reader(f)} == {len(old_header)}, "one width"
        with log.open() as f:
            parsed = list(csv_mod.DictReader(f))
        assert len(parsed) == 2
        assert not any(None in r for r in parsed), "no overflow into None"
        assert parsed[1]["mem_percent"] == truth["mem_percent"], "not shifted"
        assert parsed[0] == parsed[1]
        assert "FILE's column order" in capsys.readouterr().err


class TestANonIdIsNotBlamedOnTheDatabase:
    """Every resolve failure said "Job X does not exist in the Slurm database" — a
    claim about the database, when for a non-numeric argument the id was never valid
    in the first place. Same wrong-diagnosis shape as SW-7/SW-19/SW-22, and the likely
    intent (a job NAME) has a completely different remedy."""

    @staticmethod
    def _message(job_id: str, caplog: pytest.LogCaptureFixture) -> str:
        with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc:
            cli._die_on_resolve_error(JobNotFoundError(f"Job {job_id} not found"), job_id)
        assert exc.value.code == 1
        return " ".join(r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("bad", ["not-a-job", "my-training-run", "/home/x", "12345abc", ""])
    def test_a_non_id_says_so_and_says_how_to_find_the_real_one(
        self, bad: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        msg = self._message(bad, caplog)
        assert "is not a job id" in msg, msg
        assert "squeue --me" in msg and "sacct --name=" in msg, "name the recovery"
        assert "does not exist in the Slurm database" not in msg

    @pytest.mark.parametrize("good", ["12345", "12345_3", "123+1", " 12345 "])
    def test_a_well_formed_but_absent_id_still_blames_the_database(
        self, good: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """That IS the honest diagnosis there: the form is right, the job is gone
        (purged, or a typo'd digit)."""
        msg = self._message(good, caplog)
        assert "does not exist in the Slurm database" in msg, msg
        assert "is not a job id" not in msg


def _hop_args() -> argparse.Namespace:
    """The real parsed argv, exactly as TestSrunHop builds it."""
    return _build_parser().parse_args(["12345_3"])


class TestTheSshTransportRestoresTheTerminalToo:
    """SW-26 is a property of the OUTER process, and slurmwatch has two transports.
    This cluster prefers ssh, so a fix confined to the srun hop would have left the
    defect live wherever that rung is taken — same three unguarded exits, its own
    inline copy of the reset string."""

    ALT = "\033[?1049l"

    @staticmethod
    def _tty(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
        # The gates this transport checks before running: a tty on both ends and an
        # ssh on PATH. stdout is then swapped for a capturing tty.
        monkeypatch.setattr("sys.stdin", _FakeStream(True))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")
        buf = io.StringIO()
        monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys, "stdout", buf)
        return buf

    def test_a_keyboard_interrupt_restores_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = self._tty(monkeypatch)

        def _boom(*a: Any, **k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_run_pty_child", _boom)
        with pytest.raises(SystemExit) as exc:
            cli._ssh_to_compute_node(TestSshToComputeNode._ctx(), TestSshToComputeNode._args())
        assert exc.value.code == 130
        assert self.ALT in buf.getvalue()

    def test_a_clean_session_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for rc in (0, 130):
            buf = self._tty(monkeypatch)
            monkeypatch.setattr(
                cli,
                "_run_pty_child",
                lambda *a, _rc=rc, **k: subprocess.CompletedProcess([], _rc),
            )
            assert (
                cli._ssh_to_compute_node(TestSshToComputeNode._ctx(), TestSshToComputeNode._args())
                is True
            )
            assert self.ALT in buf.getvalue(), f"rc={rc}"

    def test_a_signal_killed_remote_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """scancel's SIGTERM reaches the remote TUI (rc 143); the local terminal still
        needs the reset before anything is printed over it."""
        buf = self._tty(monkeypatch)
        monkeypatch.setattr(
            cli, "_run_pty_child", lambda *a, **k: subprocess.CompletedProcess([], 143)
        )
        assert (
            cli._ssh_to_compute_node(TestSshToComputeNode._ctx(), TestSshToComputeNode._args())
            is True
        )
        assert self.ALT in buf.getvalue()

    def test_the_session_runs_under_the_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A signal must have something to forward to and reap."""
        self._tty(monkeypatch)
        seen: list[Any] = []

        def _record(cmd: list[str], env: dict[str, str], guard: Any = None) -> Any:
            seen.append(guard)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(cli, "_run_pty_child", _record)
        cli._ssh_to_compute_node(TestSshToComputeNode._ctx(), TestSshToComputeNode._args())
        assert seen and isinstance(seen[0], cli._TerminalGuard)


class TestNoWindowWhereASignalOrphansTheChild:
    """Self-audit of the SW-26 fix. `Popen(...)` returning and registering the child
    with the guard are two statements; a signal delivered between them ran the handler
    with nothing to forward to, so the parent restored the terminal and exited while
    the freshly-spawned session kept the tty and the job step. Small window, but it is
    precisely the harm the guard exists to prevent."""

    def test_the_spawn_happens_with_the_signals_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trace: list[str] = []

        def _mask(how: int, mask: set[int] | None = None) -> set[int]:
            if how == signal.SIG_BLOCK:
                assert mask == {signal.SIGTERM, signal.SIGHUP}, mask
                trace.append("block")
            else:
                trace.append("unblock")
            return set()

        class _Proc:
            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _popen(*a: Any, **k: Any) -> Any:
            trace.append("spawn")
            return _Proc()

        monkeypatch.setattr(signal, "pthread_sigmask", _mask)
        monkeypatch.setattr(subprocess, "Popen", _popen)
        guard = cli._TerminalGuard()
        guard.spawn(["srun"], {})
        assert trace == ["block", "spawn", "unblock"], trace
        assert guard.child is not None, "and it is registered before unblocking"

    def test_the_mask_is_restored_even_if_the_spawn_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OSError from Popen (no srun on PATH, fork failure) must not leave the
        process permanently deaf to SIGTERM."""
        trace: list[str] = []

        def _mask(how: int, mask: set[int] | None = None) -> set[int]:
            trace.append("block" if how == signal.SIG_BLOCK else "unblock")
            return set()

        monkeypatch.setattr(signal, "pthread_sigmask", _mask)

        def _boom(*a: Any, **k: Any) -> Any:
            raise OSError("no such file")

        monkeypatch.setattr(subprocess, "Popen", _boom)
        guard = cli._TerminalGuard()
        with pytest.raises(OSError):
            guard.spawn(["srun"], {})
        assert trace == ["block", "unblock"], trace
        assert guard.child is None

    def test_the_runner_registers_the_child_on_the_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard is only useful if the session actually goes through it: spawning
        with a bare Popen would leave `guard.child` None for the whole session, so a
        signal mid-run would have nothing to forward to or reap."""
        guard = cli._TerminalGuard()
        seen: dict[str, Any] = {}

        class _Proc:
            def wait(self, timeout: float | None = None) -> int:
                seen["during_wait"] = guard.child
                return 0

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc())
        cli._run_pty_child(["srun"], {}, guard)
        assert seen["during_wait"] is not None, "nothing to forward a signal to"
        assert guard.child is None, "and cleared once the session is over"

    def test_a_platform_without_sigmask_still_spawns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blocking is an optimisation of correctness, not a precondition: if the
        interpreter has no pthread_sigmask, still run the session."""
        monkeypatch.delattr(signal, "pthread_sigmask", raising=False)

        class _Proc:
            pass

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc())
        guard = cli._TerminalGuard()
        assert guard.spawn(["srun"], {}) is guard.child


class TestASignalToTheHopRestoresTheTerminal:
    """SW-26: the inner --pty TUI restores the screen on its way out (measured clean
    on-node for SIGINT/SIGTERM/q), but a signal a user sends goes to the OUTER process
    in their shell — which exited through `sys.exit(130)` or the `returncode in
    (0, 130)` shortcut, both ABOVE the reset string the function already defined.
    The terminal was left in the alternate screen with ECHO and ICANON cleared."""

    ALT_SCREEN_OFF = "\033[?1049l"

    @staticmethod
    def _tty_stdout(monkeypatch: pytest.MonkeyPatch, *, full: bool = False) -> io.StringIO:
        """A capturing stdout that claims to be a tty. With ``full``, also satisfy the
        hop's other gates (tty stdin, an srun on PATH) so it reaches the session."""
        if full:
            TestSrunHop()._force_tty(monkeypatch)
        buf = io.StringIO()
        monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys, "stdout", buf)
        return buf

    def test_the_reset_reaches_a_tty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = self._tty_stdout(monkeypatch)
        cli._restore_terminal()
        written = buf.getvalue()
        assert self.ALT_SCREEN_OFF in written, repr(written)
        assert "\033[?25h" in written, "show the cursor too"

    def test_it_falls_back_to_stderr_when_stdout_is_redirected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--log`/`>` redirect stdout; the terminal is still stderr's."""
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(out, "isatty", lambda: False, raising=False)
        monkeypatch.setattr(err, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        cli._restore_terminal()
        assert self.ALT_SCREEN_OFF in err.getvalue()
        assert out.getvalue() == "", "never into a redirected stream"

    def test_neither_a_tty_writes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, err = io.StringIO(), io.StringIO()
        for stream in (out, err):
            monkeypatch.setattr(stream, "isatty", lambda: False, raising=False)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        cli._restore_terminal()
        assert out.getvalue() == "" and err.getvalue() == ""

    def test_a_keyboard_interrupt_out_of_the_session_restores_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reporter's own test: patch the child to raise KeyboardInterrupt, give
        stdout a tty, and assert the alt-screen exit was written before we left."""
        buf = self._tty_stdout(monkeypatch, full=True)

        def _interrupted(*a: Any, **k: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_run_pty_child", _interrupted)
        with pytest.raises(SystemExit) as exc:
            cli._hop_to_compute_node(TestSrunHop()._ctx(), _hop_args())
        assert exc.value.code == 130
        assert self.ALT_SCREEN_OFF in buf.getvalue(), repr(buf.getvalue())

    def test_a_clean_quit_restores_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rc 130 took a shortcut that ASSUMED the inner TUI had tidied up — which is
        the very assumption the reset exists because it does not always hold."""
        for rc in (0, 130):
            buf = self._tty_stdout(monkeypatch, full=True)
            monkeypatch.setattr(
                cli, "_run_pty_child", lambda *a, _rc=rc, **k: subprocess.CompletedProcess([], _rc)
            )
            assert cli._hop_to_compute_node(TestSrunHop()._ctx(), _hop_args()) == cli._HOP_RAN
            assert self.ALT_SCREEN_OFF in buf.getvalue(), f"rc={rc}"

    def test_a_forwarded_signal_reaps_the_child_and_restores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SIGTERM/SIGHUP had no handler in the outer process at all, so it died
        without restoring the terminal AND without reaping the child — leaving the
        step holding the allocation's CPUs."""
        import slurmwatch.cli as _cli  # noqa: F401  (kept for symmetry with below)

        sent: list[int] = []
        waited: list[float | None] = []
        installed: dict[int, Any] = {}

        class _Child:
            def send_signal(self, signum: int) -> None:
                sent.append(signum)

            def wait(self, timeout: float | None = None) -> int:
                waited.append(timeout)
                if timeout is None:
                    raise AssertionError("should be interrupted by the handler")
                return 143

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Child())

        def _fake_signal(signum: int, handler: Any) -> Any:
            # setdefault, not assignment: the real code RESTORES the previous handler
            # in its finally, which would otherwise overwrite the forwarder we came
            # to test with the SIG_DFL this fake hands back.
            installed.setdefault(signum, handler)
            return signal.SIG_DFL

        monkeypatch.setattr(signal, "signal", _fake_signal)
        buf = self._tty_stdout(monkeypatch)
        exits: list[int] = []
        monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

        # The guard installs the handlers for the WHOLE hop (the probe window
        # included), and a child is attached only while one is running.
        with cli._TerminalGuard() as guard:
            assert signal.SIGTERM in installed and signal.SIGHUP in installed
            with contextlib.suppress(AssertionError):
                cli._run_pty_child(["srun"], {}, guard)
            guard.child = _Child()  # type: ignore[assignment]  # live child when it lands
            installed[signal.SIGTERM](signal.SIGTERM, None)
        assert sent == [signal.SIGTERM], "forward it, do not just die"
        assert waited and waited[-1] is not None, "reap with a bounded grace"
        assert self.ALT_SCREEN_OFF in buf.getvalue()
        assert exits == [128 + signal.SIGTERM], exits


class TestEveryNoTelemetryOutcomeIsMachineReadable:
    """SW-27's second half: four outcomes shared one exit code and were separable only
    by matching English on stderr — "which will break the first time the wording is
    improved", and the wording has improved twice in this exercise. Two of the four are
    normal conditions (a colleague's job runs fine; mine finished) and two are errors."""

    @staticmethod
    def _emit(monkeypatch: pytest.MonkeyPatch, fmt: str, exc: Exception, job_id: str) -> str:
        monkeypatch.setattr(cli, "_MACHINE_FORMAT", fmt)
        with pytest.raises(SystemExit):
            cli._die_on_resolve_error(exc, job_id)
        return fmt

    def test_the_three_resolution_outcomes_carry_distinct_tokens(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cases = [
            (JobNotFoundError("nope"), "12345", "job_unknown"),
            (JobNotFoundError("nope"), "not-a-job", "bad_job_id"),
            (JobNotRunningError("Job 1 has finished (State: COMPLETED)"), "1", "job_finished"),
        ]
        for exc, job_id, token in cases:
            self._emit(monkeypatch, "json", exc, job_id)
            payload = json.loads(capsys.readouterr().out)
            assert payload["telemetry_unavailable_reason"] == token, (job_id, payload)
            assert payload["telemetry_available"] is False
            assert payload["job_id"] == job_id

    def test_a_finished_job_still_reports_the_state_it_finished_in(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The terminal state is the one fact worth having, and it was thrown away.

        Measured with the real binary against a cancelled job: the row read
        `"state": null, "owner": null, "partition": null` while the prose beside it
        said "Job 562683 is in state 'CANCELLED'". A poller following a job through
        its lifecycle got `state: "RUNNING"`, then `null` — so telling COMPLETED from
        CANCELLED from TIMEOUT meant parsing the English it was given a token to
        avoid. The resolver had already parsed all of it.
        """
        exc = JobNotRunningError(
            "Job 562683 is in state 'CANCELLED'. Only running jobs can be monitored.",
            {"state": "CANCELLED", "job_name": "sw-cpu-probe", "partition": "test"},
        )
        self._emit(monkeypatch, "json", exc, "562683")
        payload = json.loads(capsys.readouterr().out)
        assert payload["telemetry_unavailable_reason"] == "job_finished"
        assert payload["state"] == "CANCELLED"
        assert payload["job_name"] == "sw-cpu-probe"
        assert payload["partition"] == "test"
        # Still no measurements: knowing the state does not make them readable.
        assert payload["telemetry_available"] is False
        assert payload["cpu_percent"] is None

    def test_an_outcome_with_nothing_known_stays_all_null(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The complement: no facts in hand must not become invented ones."""
        self._emit(monkeypatch, "json", JobNotRunningError("Job 1 has finished"), "1")
        payload = json.loads(capsys.readouterr().out)
        assert payload["state"] is None and payload["partition"] is None

    def test_a_raiser_cannot_invent_a_column(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ONE schema is the whole point of SW-27, so only known keys are merged."""
        exc = JobNotRunningError("finished", {"state": "TIMEOUT", "made_up_field": "x"})
        self._emit(monkeypatch, "json", exc, "1")
        payload = json.loads(capsys.readouterr().out)
        assert payload["state"] == "TIMEOUT"
        assert "made_up_field" not in payload

    def test_the_state_reaches_the_csv_row_too(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both machine formats, or the fix holds on whichever one wasn't checked."""
        exc = JobNotRunningError("finished", {"state": "TIMEOUT"})
        self._emit(monkeypatch, "csv", exc, "1")
        rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        assert rows[0]["state"] == "TIMEOUT"

    def test_a_slurm_failure_is_machine_readable_too(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The fifth outcome, and it wrote NOTHING to stdout.

        `--once --json` on a host with no Slurm exited 1 with an empty payload channel,
        while a finished job on the same command wrote a full object — so a consumer
        could not tell "no Slurm here" from a crash. Measured by running the real
        binary with Slurm off PATH. Three distinct tokens, because the three causes
        want opposite actions: install/module-load, report a slurmwatch bug, or wait.
        """
        cases = [
            (SlurmCommandError("no slurm", kind="unavailable"), "slurm_unavailable"),
            (SlurmCommandError("bad field", kind="unsupported"), "slurm_query_unsupported"),
            (SlurmCommandError("busy"), "slurm_error"),
        ]
        for exc, token in cases:
            self._emit(monkeypatch, "json", exc, "12345")
            payload = json.loads(capsys.readouterr().out)
            assert payload["telemetry_unavailable_reason"] == token, (token, payload)
            assert payload["telemetry_available"] is False
            assert payload["reason"] == str(exc)

    def test_the_slurm_token_comes_from_the_kind_not_the_wording(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Rewording an error must not silently reclassify it.

        The first version of this keyed on "retrying will not help" appearing in the
        message — the same brittleness SW-27 exists to remove, reintroduced one layer
        down.
        """
        exc = SlurmCommandError("anything at all, worded however", kind="unavailable")
        self._emit(monkeypatch, "json", exc, "12345")
        payload = json.loads(capsys.readouterr().out)
        assert payload["telemetry_unavailable_reason"] == "slurm_unavailable"

    def test_the_schema_matches_the_foreign_job_payload(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One shape for every no-telemetry outcome, so a poller can switch on the
        token instead of on which keys happen to be present. Derived from
        _foreign_facts rather than duplicated, so the two cannot drift."""
        ctx = JobContext(
            job_id="9",
            username="someone",
            partition="p",
            nodelist="cn1",
            hostname="login",
            cpus_allocated=2,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            remote=True,
        )
        foreign_keys = set(cli._foreign_facts(ctx))
        self._emit(monkeypatch, "json", JobNotFoundError("nope"), "12345")
        unknown_keys = set(json.loads(capsys.readouterr().out))
        assert unknown_keys == foreign_keys, unknown_keys ^ foreign_keys

    def test_nothing_is_written_when_no_machine_format_was_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The interactive path prints its own message; a stray JSON object on stdout
        there would be noise."""
        self._emit(monkeypatch, "", JobNotFoundError("nope"), "12345")
        assert capsys.readouterr().out == ""

    def test_csv_emits_a_header_and_one_row(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._emit(monkeypatch, "csv", JobNotFoundError("nope"), "12345")
        rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
        assert len(rows) == 1
        assert rows[0]["telemetry_unavailable_reason"] == "job_unknown"
        assert rows[0]["job_id"] == "12345"

    def test_main_records_the_format_for_each_machine_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The tokens only reach stdout if main() recorded the format before
        resolution — that is where these three outcomes are decided."""
        seen: list[str] = []
        monkeypatch.setattr(cli, "_run_once", lambda *a, **k: seen.append(cli._MACHINE_FORMAT))
        monkeypatch.setattr(cli, "_job_id_without_step", lambda j: j)
        for argv, expected in (
            (["1", "--once"], "csv"),
            (["1", "--once", "--json"], "json"),
        ):
            monkeypatch.setattr(cli, "_MACHINE_FORMAT", "")
            with contextlib.suppress(SystemExit):
                main(argv)
            assert seen and seen[-1] == expected, (argv, seen)


class TestTheFactsCsvIsNotAFormulaVector:
    """The telemetry CSV has run `job_name` through `_csv_text` since a job named
    `=cmd|"/bin/sh"!A1` was shown to be a live DDE cell, not a label — but the FACTS
    writers used a raw csv.writer, so the same field was guarded on one CSV surface and
    executable on the other. Worse here than on the telemetry path: a foreign job's
    name was chosen by somebody else, so the hostile case is the ordinary one."""

    HOSTILE = '=cmd|"/bin/sh"!A1'

    @staticmethod
    def _ctx(name: str) -> JobContext:
        return JobContext(
            job_id="9",
            username="someone-else",
            partition="p",
            nodelist="cn1",
            hostname="login",
            cpus_allocated=2,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            job_name=name,
            remote=True,
        )

    def test_once_csv_neutralises_a_hostile_job_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli._emit_facts_payload(
            cli._foreign_facts(self._ctx(self.HOSTILE)), "csv", SlurmwatchConfig()
        )
        rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
        assert rows[0]["job_name"] == "'" + self.HOSTILE, rows[0]["job_name"]

    def test_the_log_row_neutralises_it_too(self, tmp_path: Path) -> None:
        """Same field, the other writer — fixing one is half a fix."""
        log = tmp_path / "f.csv"
        cli._write_facts_row(
            cli._foreign_facts(self._ctx(self.HOSTILE)), SlurmwatchConfig(), str(log), "csv"
        )
        rows = list(csv.DictReader(log.read_text().splitlines()))
        assert rows[0]["job_name"] == "'" + self.HOSTILE

    def test_the_no_telemetry_row_neutralises_it_too(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The job id is echoed back from argv, so it is user text as well."""
        monkeypatch.setattr(cli, "_MACHINE_FORMAT", "csv")
        with pytest.raises(SystemExit):
            cli._die_on_resolve_error(JobNotFoundError("nope"), "=HYPERLINK(1)")
        rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
        assert rows[0]["job_id"] == "'=HYPERLINK(1)", rows[0]["job_id"]

    def test_json_is_left_alone(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The quote is a spreadsheet convention; JSON consumers must get the real
        string, exactly as `_csv_text`'s docstring says."""
        cli._emit_facts_payload(
            cli._foreign_facts(self._ctx(self.HOSTILE)), "json", SlurmwatchConfig()
        )
        assert json.loads(capsys.readouterr().out)["job_name"] == self.HOSTILE

    def test_an_ordinary_name_is_untouched(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli._emit_facts_payload(
            cli._foreign_facts(self._ctx("train-llama-8b")), "csv", SlurmwatchConfig()
        )
        rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
        assert rows[0]["job_name"] == "train-llama-8b"

    def test_main_records_the_dialect_for_the_resolution_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dialect only reaches those paths if main() records it — they have no
        config in hand, so without this the row silently reverts to "excel"."""
        seen: list[str] = []
        monkeypatch.setattr(cli, "_run_once", lambda *a, **k: seen.append(cli._MACHINE_CSV_DIALECT))
        monkeypatch.setattr(cli, "_job_id_without_step", lambda j: j)
        monkeypatch.setattr(cli, "_MACHINE_CSV_DIALECT", "sentinel-never-set")
        monkeypatch.setenv("SLURMWATCH_CSV_DIALECT", "excel-tab")
        with contextlib.suppress(SystemExit):
            main(["1", "--once"])
        assert seen == ["excel-tab"], seen

    def test_the_no_telemetry_row_honours_the_configured_dialect(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """It hardcoded "excel", so this one row disagreed with the telemetry rows a
        consumer had set SLURMWATCH_CSV_DIALECT for."""
        monkeypatch.setattr(cli, "_MACHINE_FORMAT", "csv")
        monkeypatch.setattr(cli, "_MACHINE_CSV_DIALECT", "excel-tab")
        with pytest.raises(SystemExit):
            cli._die_on_resolve_error(JobNotFoundError("nope"), "12345")
        out = capsys.readouterr().out
        assert "\t" in out, out
        assert ",job_id," not in out


class TestAFailedReadIsNotALogWriteFailure:
    """A telemetry read that fails must not be blamed on the log file (SW-75).

    `--log` wrapped the whole sampling loop in one `except OSError`, so anything
    the READ raised was reported as "Cannot write log file" and exited 1. ssh and
    sstat fail with OSError subclasses — TimeoutError on a wedged hop,
    ConnectionResetError when one drops — so a days-long run could die over a
    single bad cycle, naming a file that was perfectly writable. And because
    `str(TimeoutError())` is empty, the line named no reason whatsoever.

    A builtin TimeoutError takes a DIFFERENT ROUTE per version and the tests here
    are split accordingly: asyncio.TimeoutError is not the builtin on 3.10 (they
    became one class in 3.11), so there it lands in the read handler, while on 3.11+
    the timeout branch claims it first — correctly, since it *is* the timeout. What
    must hold on every version is the outcome: the run survives and never blames the
    file. The read handler's own message is asserted with a non-timeout OSError, which
    is unambiguous everywhere.
    """

    def _collector(self, first: BaseException) -> type:
        class _Flaky:
            def __init__(self, ctx: object, config: object) -> None:
                self.job_ended = False
                self.calls = 0

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

            async def next_snapshot(self) -> Any:
                self.calls += 1
                if self.calls == 1:
                    raise first
                # Second cycle: the job is gone, which is how the loop ends.
                self.job_ended = True
                raise asyncio.TimeoutError

        return _Flaky

    async def _run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> int:
        monkeypatch.setattr(cli, "TelemetryCollector", self._collector(exc))
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        return await asyncio.wait_for(
            _headless_loop(ctx, cfg, str(tmp_path / "m.jsonl"), "json"), timeout=10.0
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    @pytest.mark.parametrize(
        "exc",
        [TimeoutError(), ConnectionResetError(104, "Connection reset by peer"), OSError()],
        ids=["timeout", "conn-reset", "bare-oserror"],
    )
    async def test_the_run_survives_it_and_never_blames_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        exc: BaseException,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="slurmwatch"):
            # No SystemExit: the loop returns 0 for "the job ended".
            code = await self._run(tmp_path, monkeypatch, exc)
        assert code == 0
        assert "Cannot write log file" not in caplog.text, caplog.text

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    @pytest.mark.parametrize(
        "exc",
        [ConnectionResetError(104, "Connection reset by peer"), OSError("no route to host")],
        ids=["conn-reset", "bare-oserror"],
    )
    async def test_it_says_the_read_is_what_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        exc: BaseException,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="slurmwatch"):
            await self._run(tmp_path, monkeypatch, exc)
        assert "Telemetry read failed" in caplog.text, caplog.text

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_blank_exception_still_names_its_kind(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`str(ConnectionResetError())` is "", so the message has to fall back to the
        class name — "failed: " with nothing after it tells the reader nothing."""
        with caplog.at_level(logging.WARNING, logger="slurmwatch"):
            await self._run(tmp_path, monkeypatch, ConnectionResetError())
        assert "Telemetry read failed: ConnectionResetError" in caplog.text, caplog.text

    def test_the_helper_never_returns_a_blank(self) -> None:
        assert cli._exception_text(TimeoutError()) == "TimeoutError"
        assert cli._exception_text(OSError()) == "OSError"
        assert cli._exception_text(OSError("no space left")) == "no space left"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_source_that_always_fails_is_paced_not_spun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read that fails instantly and forever must not spin the loop hot: this
        loop's signal handling is `loop.add_signal_handler`, so a hot branch starves
        the callback that stops it (verified: a starved loop survives SIGHUP)."""
        slept: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float, *a: Any, **kw: Any) -> Any:
            slept.append(delay)
            return await real_sleep(0, *a, **kw)  # keep the test instant

        class _AlwaysFails:
            def __init__(self, ctx: object, config: object) -> None:
                self.job_ended = False
                self.calls = 0

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

            async def next_snapshot(self) -> Any:
                self.calls += 1
                if self.calls > 3:
                    self.job_ended = True
                    raise asyncio.TimeoutError
                raise ConnectionResetError(104, "Connection reset by peer")

        monkeypatch.setattr(cli, "TelemetryCollector", _AlwaysFails)
        monkeypatch.setattr(asyncio, "sleep", _recording_sleep)
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.02, headless_interval=0.02)
        await asyncio.wait_for(
            _headless_loop(ctx, cfg, str(tmp_path / "m.jsonl"), "json"), timeout=10.0
        )
        # Not just the sleep(0) yield at the top of each iteration: a real delay,
        # floored well above zero so a fast-failing source cannot busy-loop.
        assert any(d >= 0.5 for d in slept), slept


class TestCancellingTheLoggerIsHonoured:
    """`--log` could swallow its own cancellation and keep running (SW-77).

    Each sample races the write against the shutdown event, then reaps the loser:

        shutdown_fut.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_fut

    That `suppress` cannot tell `shutdown_fut`'s cancellation from the enclosing
    task's. A cancel landing inside the window — a Ctrl-C, a SIGTERM, a caller
    cancelling the logger — was caught and discarded, and the loop carried on: not
    slow, but unkillable, and whoever awaited the task waited forever on an event
    loop with nothing left to schedule. Once per sample is a narrow window, which is
    why it read as one hung CI job in twenty and never reproduced locally.
    """

    @staticmethod
    def _uncooperative(release: asyncio.Event) -> asyncio.Task[None]:
        async def _body() -> None:
            while not release.is_set():
                try:
                    await asyncio.wait_for(release.wait(), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    continue

        return asyncio.create_task(_body())

    @pytest.mark.asyncio
    async def test_the_reap_does_not_eat_the_callers_cancellation(self) -> None:
        release = asyncio.Event()
        target = self._uncooperative(release)
        await asyncio.sleep(0.05)  # let it reach its suspension point
        outer = asyncio.create_task(aio.reap_cancelled(target))
        await asyncio.sleep(0.05)
        try:
            outer.cancel()
            # asyncio.wait, never `await outer` / wait_for: if the reap DOES swallow
            # the cancel, awaiting it waits on the uncooperative future forever and
            # this test becomes the hang it is meant to report. wait abandons.
            await asyncio.wait({outer}, timeout=2.0)
            assert outer.done(), "the reap ignored the cancellation and kept waiting"
            assert outer.cancelled(), "the cancellation was swallowed"
        finally:
            release.set()
            target.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(target, timeout=2.0)

    @pytest.mark.asyncio
    async def test_it_still_reaps_the_future_it_cancelled(self) -> None:
        """The point of the reap is that nothing is left dangling — and a future's
        OWN cancellation must not escape to the caller."""
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await aio.reap_cancelled(fut)  # must not raise
        assert fut.cancelled()


class TestCsvLineEndingsSuitTheirDestination:
    """SW-31: no dialect produced shell-usable output.

    The default, "excel", is RFC 4180 and ends every row with CRLF — correct for a
    file a spreadsheet opens, and a trap in a pipeline, because the trailing \\r ends
    up INSIDE the last field: `awk -F, '{print $NF}'` yields "100\\r", and a test
    against it fails for a reason nothing on screen explains. The only LF dialect the
    stdlib ships, "unix", is QUOTE_ALL, so every field arrives quoted and numbers
    read as strings. Neither is usable, and the knob that chose between them was
    undocumented.

    So slurmwatch registers its own (LF, minimal quoting) and picks by DESTINATION,
    which is the distinction --log already draws when it decides what to claim: a
    real file keeps CRLF, a pipe/tty/dev-stdout gets LF. The knob still wins.
    """

    def test_the_registered_dialect_is_lf_and_minimally_quoted(self) -> None:
        d = csv.get_dialect(config_mod.SHELL_CSV_DIALECT)
        assert d.lineterminator == "\n"
        assert d.quoting == csv.QUOTE_MINIMAL
        assert d.delimiter == ","

    @pytest.mark.parametrize(
        ("configured", "to_file", "expected"),
        [
            ("auto", False, "slurmwatch"),
            ("auto", True, "excel"),
            ("excel", False, "excel"),  # an explicit choice wins on a pipe
            ("unix", True, "unix"),  # ... and on a file
        ],
    )
    def test_auto_resolves_by_destination_and_an_explicit_choice_wins(
        self, configured: str, to_file: bool, expected: str
    ) -> None:
        assert config_mod.resolve_csv_dialect(configured, to_regular_file=to_file) == expected

    def test_auto_is_the_default_and_is_accepted(self) -> None:
        cfg = SlurmwatchConfig()
        assert cfg.csv_dialect == "auto"
        cfg.validate()  # must not raise

    def test_a_bogus_dialect_is_still_rejected_and_the_message_offers_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_CSV_DIALECT", "spreadsheet")
        with pytest.raises(ValueError, match="SLURMWATCH_CSV_DIALECT") as exc:
            SlurmwatchConfig.from_env()
        assert "auto" in str(exc.value), str(exc.value)

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_stdout_csv_has_no_carriage_returns(
        self, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["slurmwatch", "12345", "--once", "--format", "csv"])
        with contextlib.suppress(SystemExit):
            main(["12345", "--once", "--format", "csv"])
        out = capsysbinary.readouterr().out
        assert b"\r" not in out, out[:200]
        assert out.count(b"\n") >= 2, out[:200]

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_stdout_csv_quotes_only_when_needed(self, capsys: pytest.CaptureFixture[str]) -> None:
        with contextlib.suppress(SystemExit):
            main(["12345", "--once", "--format", "csv"])
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        # A field that needs no quoting must arrive bare — with QUOTE_ALL every one
        # of these would be '"timestamp"' and a consumer would read numbers as text.
        assert lines[0].split(",")[0] == "timestamp", lines[0][:80]
        assert not lines[1].startswith('"'), lines[1][:80]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_real_log_file_keeps_rfc4180_crlf(self, tmp_path: Path) -> None:
        """The other half of the rule: a .csv a spreadsheet opens is exactly where
        CRLF belongs, so auto must not "fix" that too."""
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "m.csv"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), "csv"))
        await _wait_for_lines(out, 2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10.0)
        raw = out.read_bytes()
        assert b"\r\n" in raw, raw[:120]
        # and it still parses as one schema
        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0][0] == "timestamp"
        assert len(rows[0]) == len(rows[1])

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_log_to_a_pipe_gets_lf(self, tmp_path: Path) -> None:
        """A fifo is not a regular file, so it takes the shell-friendly dialect —
        the same rule --log already uses to decide what to claim."""
        fifo = tmp_path / "pipe.csv"
        os.mkfifo(fifo)
        assert cli._path_is_regular_file(str(fifo)) is False
        assert (
            config_mod.resolve_csv_dialect("auto", to_regular_file=False)
            == config_mod.SHELL_CSV_DIALECT
        )

    def test_a_device_is_not_a_regular_file(self) -> None:
        """Note what is NOT asserted here: /dev/stdout follows whatever stdout is,
        and under pytest that is a real capture file — so it legitimately answers
        True in this process and False when piped to awk. The rule keys off the
        destination, so a test may not pin one answer for it."""
        assert cli._path_is_regular_file("/dev/null") is False
        assert cli._path_is_regular_file("/dev/zero") is False

    def test_a_path_that_does_not_exist_yet_counts_as_a_file(self, tmp_path: Path) -> None:
        """--log names the file it is about to create; that is a file, not a pipe."""
        assert cli._path_is_regular_file(str(tmp_path / "not-yet.csv")) is True

    def test_a_reader_handed_auto_resolves_it_itself(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The two path-readers resolve "auto" themselves, so no call site can hand
        them an unregistered dialect name. That is not belt-and-braces: when one call
        site was left passing the raw value, `csv.reader(dialect="auto")` raised
        csv.Error, which both readers swallow — so `--append`'s width warning simply
        stopped appearing, with nothing on screen to say why.
        """
        from slurmwatch.model import TelemetrySnapshot

        log = tmp_path / "agg.csv"
        log.write_text(",".join(TelemetrySnapshot.csv_header(2)) + "\r\n")
        assert cli._csv_max_gpus_from_header(str(log), "auto") == 2
        cli._csv_append_layout(str(log), "auto", 2, 4)
        assert "2 GPU column(s)" in capsys.readouterr().err

    def test_the_help_documents_the_knob(self) -> None:
        parser = _build_parser()
        text = parser.format_help()
        assert "SLURMWATCH_CSV_DIALECT" in text
        assert "CRLF" in text and "LF" in text


class TestANonUtf8StreamDoesNotEndTheRun:
    """SW-79: `sw --help` died with a traceback on any stream that is not UTF-8.

    A cluster whose locale is not UTF-8 hands Python a stream that cannot carry an em
    dash — latin-1 from an ISO-8859 locale, ASCII from PYTHONIOENCODING or an
    uncoerced C locale, both measured on this box:

        PYTHONIOENCODING=ascii        -> ascii
        LC_ALL=en_US.ISO-8859-1       -> iso8859-1

    argparse writes the epilog with a bare `file.write()`, so the failure landed on
    `--help`: the first command anyone runs on a new machine, answering with
    `UnicodeEncodeError: 'ascii' codec can't encode character '\\u2014'`. `--ascii`
    could not save it either — the epilog is rendered before any flag is read.

    Two guards, because they cover different things: ASCII text for the glyphs WE
    choose, and `backslashreplace` on the streams for text we do not (a job name with
    an accent arrives however Slurm reports it).
    """

    @pytest.mark.parametrize("encoding", ["ascii", "iso-8859-1", "utf-8"])
    def test_the_probe_recognises_what_the_stream_can_carry(
        self, encoding: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Stream:
            encoding = ""

            def reconfigure(self, **kw: object) -> None: ...

        s = _Stream()
        s.encoding = encoding
        monkeypatch.setattr(sys, "stdout", s)
        monkeypatch.setattr(sys, "stderr", s)
        assert cli._harden_output_streams() is (encoding != "utf-8")

    def test_an_unknown_encoding_name_is_treated_as_ascii_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LookupError, not UnicodeEncodeError — a stream can name a codec Python
        does not have. Assuming it copes is the wrong way to be wrong."""

        class _Stream:
            encoding = "wobble-9"

            def reconfigure(self, **kw: object) -> None: ...

        monkeypatch.setattr(sys, "stdout", _Stream())
        monkeypatch.setattr(sys, "stderr", _Stream())
        assert cli._harden_output_streams() is True

    def test_a_stream_that_cannot_be_reconfigured_is_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pytest's own capture object has no reconfigure(); nor does a closed
        stream. Neither is a reason to fail before the tool has done anything."""

        class _Stubborn:
            encoding = "utf-8"

            def reconfigure(self, **kw: object) -> None:
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(sys, "stdout", _Stubborn())
        monkeypatch.setattr(sys, "stderr", _Stubborn())
        assert cli._harden_output_streams() is False  # must not raise

    def test_the_help_epilog_is_pure_ascii_when_the_stream_is(self) -> None:
        text = _build_parser(ascii_only=True).format_help()
        bad = sorted({c for c in text if not c.isascii()})
        assert not bad, bad
        # and the em dash is still there by default, not lost for everyone
        assert not _build_parser().format_help().isascii()

    def test_the_end_of_job_note_folds_too(self) -> None:
        """The one line in the --log path that was a bare constant, printed
        regardless of --ascii."""
        assert cli._job_ended_note(ascii_mode=True).isascii()
        assert not cli._job_ended_note(ascii_mode=False).isascii()

    def test_the_streams_are_actually_reconfigured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both of them, and with the tolerant error handler — not just probed."""
        calls: list[dict[str, object]] = []

        class _Stream:
            encoding = "utf-8"

            def reconfigure(self, **kw: object) -> None:
                calls.append(kw)

        monkeypatch.setattr(sys, "stdout", _Stream())
        monkeypatch.setattr(sys, "stderr", _Stream())
        cli._harden_output_streams()
        assert calls == [{"errors": "backslashreplace"}] * 2, calls

    def test_a_job_name_we_do_not_control_cannot_kill_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard that ASCII text cannot provide: Slurm reported the name, so
        folding OUR glyphs does nothing for it. Written through the real stream after
        the real hardening call — a job name is data, and data must degrade, not
        raise."""
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        assert cli._harden_output_streams() is True  # ascii cannot carry the glyphs
        sys.stdout.write("job: caf\u00e9-tokenise\n")  # would raise, unhardened
        stream.flush()
        assert b"caf" in raw.getvalue(), raw.getvalue()
        assert b"\\xe9" in raw.getvalue() or b"\\u00e9" in raw.getvalue(), raw.getvalue()

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_main_turns_on_ascii_and_says_so_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[bool] = []
        monkeypatch.setattr(cli, "_harden_output_streams", lambda: True)
        monkeypatch.setattr(cli, "_encoding_notice_shown", False)

        def _capture(*a: object, **kw: object) -> None:
            seen.append(bool(getattr(a[1], "ascii_mode", False)))
            sys.exit(0)

        monkeypatch.setattr(cli, "_run_once", _capture)
        with contextlib.suppress(SystemExit):
            main(["12345", "--once"])
        err = capsys.readouterr().err
        assert seen == [True], seen
        assert "plain ASCII" in err, err
        # said once, not per frame (the rule the stale-env notice follows)
        with contextlib.suppress(SystemExit):
            main(["12345", "--once"])
        assert capsys.readouterr().err.count("plain ASCII") == 0


class TestAJobNameCannotDriveTheTerminal:
    """SW-82: the job name was guarded against two interpreters, not the third.

    `sbatch -J` takes arbitrary text, and this package already treats that field as
    untrusted twice: `_csv_text` prefixes a quote so a spreadsheet cannot evaluate
    `=cmd|"/bin/sh"!A1`, and `_escape_markup` neutralizes `[` so Textual's parser
    cannot be steered by it. The TERMINAL is the third interpreter of the same field,
    and it was unguarded — a name containing `\\x1b[2J` clears the screen of anyone who
    `cat`s the CSV log, and Rich passes ESC through to the dashboard verbatim (measured:
    Rich strips CR, not ESC).

    Reachability is not hypothetical: a NEWLINE in a job name is SW-1, already reported
    from a live cluster, so control characters do arrive in this field.
    """

    HOSTILE = "train\x1b[2J\x1b[31mRED\rboom\x08x"

    @pytest.mark.parametrize(
        "render",
        [
            pytest.param(lambda s: model_mod._csv_text(s), id="csv"),
            pytest.param(lambda s: cli._name_suffix(s), id="plain-text-summary"),
        ],
    )
    def test_no_control_character_reaches_the_output(self, render: Any) -> None:
        out = render(self.HOSTILE)
        assert not any(c in out for c in ("\x1b", "\r", "\x08")), repr(out)
        # escaped, not deleted: the reader can still see what the name was
        assert "x1b" in out and "train" in out, repr(out)

    def test_the_dashboard_renderer_is_covered_too(self) -> None:
        from slurmwatch.tui import _escape_markup

        out = _escape_markup(self.HOSTILE)
        assert not any(c in out for c in ("\x1b", "\r", "\x08")), repr(out)

    def test_legitimate_non_ascii_is_left_alone(self) -> None:
        """`str.isprintable()` is true for accented and CJK text, so a name in another
        language must survive intact — the guard is about control codes, not bytes."""
        from slurmwatch.units import printable_text

        for name in ("café-träning", "中文-实验", "µ-benchmark", "naïve_v2"):
            assert printable_text(name) == name

    def test_the_spreadsheet_guard_still_applies(self) -> None:
        """The formula vector this function was written for must not regress: the
        control-char pass runs first, so a leading `=` still gets its quote."""
        assert model_mod._csv_text('=cmd|"/bin/sh"!A1').startswith("'")
        assert model_mod._csv_text("+1").startswith("'")
        assert model_mod._csv_text("normal-name") == "normal-name"

    def test_a_leading_tab_or_cr_is_handled_by_the_first_pass(self) -> None:
        """Those two entries left the formula tuple deliberately: after neutralizing,
        a leading tab arrives as the printable pair `\\` `t`, so the old check could
        never fire again. Assert the OUTCOME rather than the mechanism."""
        for raw in ("\tcmd", "\rcmd"):
            out = model_mod._csv_text(raw)
            assert "\t" not in out and "\r" not in out, repr(out)
            assert out.startswith("\\"), repr(out)

    def test_the_length_cap_counts_what_the_reader_sees(self) -> None:
        """Neutralizing after the cap could slice an escape in half and leave a partial
        sequence; the cap is applied to the escaped text for that reason."""
        out = cli._name_suffix("\x1b[2J" * 30)
        assert "\x1b" not in out
        assert len(out) <= len("  name `") + 43


class TestAShortWriteCannotTruncateARecordSilently:
    """SW-83: `os.write`'s return value was discarded, so a record could be half-written.

    `_write_record`'s docstring is the log format's correctness claim — *"One record, one
    write() — the atomicity the log format depends on"* — and the function ignored the one
    thing that makes it true. `write()` may return SHORT: measured with `RLIMIT_FSIZE`, a
    4001-byte record crossing the limit wrote **999 bytes and returned**, leaving half a
    row in the file while the caller believed it had succeeded. The loop then failed on the
    NEXT record, so the reported error pointed at the wrong sample and the corrupt line was
    already on disk. A filesystem filling up or a quota boundary produces the same thing.

    This is SW-16's failure mode — *"a file that parses at the start and raises
    JSONDecodeError partway through"* — reached by a different route than the two-writer
    race that report fixed.
    """

    @staticmethod
    def _run(body: str) -> str:
        """Out-of-process, because RLIMIT_FSIZE cannot be raised again once lowered:
        setting it in the test runner would break every later test that writes a file
        (`ValueError: not allowed to raise maximum limit`)."""
        code = (
            "import os, resource, signal, sys, tempfile\n"
            f"sys.path.insert(0, {os.getcwd()!r})\n"
            "sys.path.insert(0, 'src')\n"
            "from slurmwatch.cli import _write_record\n"
            "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n" + body
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            # No bytecode writing: this child runs under a tiny RLIMIT_FSIZE, and a .pyc
            # write that trips the limit turns a 2s test into a 30s one.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=90,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout

    def test_a_record_that_cannot_be_finished_says_the_tail_is_incomplete(self) -> None:
        out = self._run(
            "path = tempfile.mktemp()\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (5000, 5000))\n"
            "fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)\n"
            "rec = b'y' * 4000 + b'\\n'\n"
            "_write_record(fd, rec)\n"
            "try:\n"
            "    _write_record(fd, rec)\n"
            "    print('NO-ERROR')\n"
            "except OSError as e:\n"
            "    print('ERR', e)\n"
            "os.close(fd); os.unlink(path)\n"
        )
        assert "NO-ERROR" not in out, out
        assert "incomplete" in out, out
        # and it says HOW MUCH went out, which is what tells a reader where to cut
        assert "wrote 999 of 4001 bytes" in out, out

    def test_a_short_write_that_can_continue_is_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recoverable case, which is the point of the loop: a sink that accepts the
        record in pieces must still receive all of it, in order."""
        chunks: list[int] = []
        real_write = os.write

        def _dribble(fd: int, data: Any) -> int:
            n = min(7, len(data))  # accept a sliver at a time
            chunks.append(n)
            return real_write(fd, bytes(data)[:n])

        target = tmp_path / "m.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            monkeypatch.setattr(os, "write", _dribble)
            cli._write_record(fd, b'{"job_id": "12345"}\n')
        finally:
            os.close(fd)
        assert len(chunks) > 1, "the test did not actually exercise a short write"
        assert target.read_bytes() == b'{"job_id": "12345"}\n'

    def test_a_sink_making_no_progress_fails_instead_of_spinning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking write returning 0 for a non-empty buffer should not happen, and
        retrying it forever would spin the executor thread hot — the hazard the poll
        loops were fixed for. It must fail with the byte count instead."""
        target = tmp_path / "m.jsonl"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            monkeypatch.setattr(os, "write", lambda *a, **k: 0)
            with pytest.raises(OSError, match="made no progress"):
                cli._write_record(fd, b"payload\n")
        finally:
            os.close(fd)

    def test_the_header_for_an_eight_gpu_node_exceeds_pipe_buf(self) -> None:
        """Pins the fact behind the documented bound: `write()` is atomic on a pipe only
        up to PIPE_BUF (4096), and `--log /dev/stdout` is a pipe. 8 GPUs per node is the
        standard HGX layout, not an edge case, so the claim needed the caveat."""
        from slurmwatch.model import TelemetrySnapshot

        header = ",".join(TelemetrySnapshot.csv_header(8))
        assert len(header.encode()) > 4096, len(header)
        assert len(",".join(TelemetrySnapshot.csv_header(0)).encode()) < 4096


class TestTheDocumentedFormatMappingIsTrue:
    """The report's round 72 asked for the CSV<->JSON naming to be documented.

    Its measurement: flattening the JSONL gives 29 fields against the CSV's 27, and the
    difference is *entirely naming* — five quantities carry two names, and `--help`
    presented the formats as interchangeable (".jsonl or .csv, inferred from the file
    extension"), which reads as one dataset in two encodings rather than two
    vocabularies.

    Aligning the names was the other option and is rejected: those column names are
    pinned by a test on purpose, so renaming them would break every existing consumer to
    fix a papercut. Documenting the rule is the non-breaking half — and a documented
    mapping that nothing checks rots, so each pair the help text claims is asserted here
    against the real schemas.
    """

    PAIRS = [
        ("cpu.usage_percent", "cpu_percent"),
        ("memory.usage_percent", "mem_percent"),
        ("memory.oom_guard_warning", "mem_oom_warning"),
        ("cpu.cores_allocated", "cpu_cores"),
        ("memory.oom_guard_critical", "mem_oom_critical"),
    ]

    @staticmethod
    def _snapshot() -> Any:
        from slurmwatch.model import CpuMetrics, MemoryMetrics, TelemetrySnapshot

        return TelemetrySnapshot(
            timestamp=1.0,
            job_id="12345",
            step_id=None,
            hostname="n",
            elapsed_seconds=10,
            cpu=CpuMetrics(cores_allocated=2, usage_ns=0, usage_percent=12.5),
            memory=MemoryMetrics(
                current_bytes=1024,
                limit_bytes=4096,
                peak_bytes=2048,
                usage_percent=25.0,
                oom_guard_warning=True,
                oom_guard_critical=False,
            ),
        )

    @pytest.mark.parametrize(("json_path", "csv_name"), PAIRS)
    def test_each_documented_pair_exists_on_both_sides(self, json_path: str, csv_name: str) -> None:
        from slurmwatch.model import TelemetrySnapshot

        payload = json.loads(self._snapshot().to_json())
        node: Any = payload
        for part in json_path.split("."):
            assert part in node, f"{json_path} missing from the JSON payload"
            node = node[part]
        assert csv_name in TelemetrySnapshot.csv_header(0), f"{csv_name} missing from the CSV"

    @pytest.mark.parametrize(("json_path", "csv_name"), PAIRS)
    def test_each_pair_carries_the_same_measurement(self, json_path: str, csv_name: str) -> None:
        from slurmwatch.model import TelemetrySnapshot

        snap = self._snapshot()
        payload = json.loads(snap.to_json())
        node: Any = payload
        for part in json_path.split("."):
            node = node[part]
        row = dict(zip(TelemetrySnapshot.csv_header(0), snap.to_csv_row(0), strict=True))
        csv_value = row[csv_name]
        if isinstance(node, bool):
            # The documented difference: true/false in JSON, 1/0 in CSV.
            assert csv_value in ("1", "0"), csv_value
            assert (csv_value == "1") is node
        else:
            assert float(csv_value) == pytest.approx(float(node)), (json_path, node, csv_value)

    def test_the_help_text_states_the_rule(self) -> None:
        text = _build_parser().format_help()
        assert "two vocabularies" in text
        for shown in ("cpu.usage_percent", "cpu_percent", "mem_oom_warning", "gpu_<N>_*"):
            assert shown in text, shown

    def test_the_topology_matrix_is_json_only_as_documented(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        assert not any("matrix" in c for c in TelemetrySnapshot.csv_header(4))


class TestTheLogFileModeIsOnePolicy:
    """SW-84: three calls opened the --log path and disagreed about its mode.

    The report's round 72 noted in passing that both log files land `-rw-rw-r--`, and
    declined to file it — the `--log` path is chosen explicitly, so writing it with
    default permissions is defensible. It is the *disagreement* that is the defect:
    the telemetry loop asks for `0o644`, and the writability PREFLIGHT — which runs
    first and therefore creates the file — used `open(path, "a")`, i.e. Python's
    `0o666 & ~umask`. Under this cluster's umask 0002 that is `0o664`, so the mode
    actually applied was chosen by the one call that was not trying to set a policy,
    and the loop's stated intent was dead.

    The consequence is not academic: a telemetry log in a shared project directory at
    0o664 can be REWRITTEN by anyone in the group — not merely read. Measurements a
    groupmate can edit are worse than measurements they can see.
    """

    @staticmethod
    def _mode_under_umask(value: int, path: Path, create: Callable[[str], None]) -> int:
        """umask is process-global; unlike RLIMIT it can be restored, so no subprocess.

        Takes the path explicitly rather than closing over a loop variable — a lambda
        that binds one is the B023 bug ruff caught here, and it would have made every
        iteration measure the last path.
        """
        old = os.umask(value)
        try:
            create(str(path))
        finally:
            os.umask(old)
        return int(stat.S_IMODE(os.stat(path).st_mode))

    def test_a_permissive_umask_does_not_make_the_log_group_writable(self, tmp_path: Path) -> None:
        target = tmp_path / "m.csv"

        def _create(path: str) -> None:
            with open(path, "a", opener=cli._log_opener):
                pass

        mode = self._mode_under_umask(0o002, target, _create)
        assert mode == 0o644, oct(mode)
        assert not mode & stat.S_IWGRP, "a groupmate could rewrite the measurements"

    def test_every_opener_of_the_log_agrees(self, tmp_path: Path) -> None:
        """The preflight, the loop's raw fd and the facts-row append must produce the
        same mode — whichever of them happens to create the file first."""

        def preflight(p: str) -> None:
            open(p, "a", opener=cli._log_opener).close()

        def loop_fd(p: str) -> None:
            os.close(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, cli._LOG_FILE_MODE))

        def facts_row(p: str) -> None:
            open(p, "a", newline="", opener=cli._log_opener).close()

        modes = {
            self._mode_under_umask(0o002, tmp_path / f"m{i}.csv", fn)
            for i, fn in enumerate((preflight, loop_fd, facts_row))
        }
        assert modes == {0o644}, [oct(m) for m in modes]

    def test_a_stricter_umask_is_still_honoured(self, tmp_path: Path) -> None:
        """The mode is a ceiling, not an override: a site that masks group/other read
        must keep getting that."""

        def _create(path: str) -> None:
            with open(path, "a", opener=cli._log_opener):
                pass

        assert self._mode_under_umask(0o077, tmp_path / "m.csv", _create) == 0o600

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_real_entry_point_lands_at_the_declared_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through `_run_headless`, NOT `_headless_loop`: the preflight that actually
        creates the file lives in the former, so a test that calls the latter directly
        passes with the bug restored. (It did — the mutant proved it.)"""

        class _OneShot:
            def __init__(self, job_ctx: object, config: object) -> None:
                self.job_ended = False

            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            def stop_sync(self) -> None: ...

            async def next_snapshot(self) -> TelemetrySnapshot:
                self.job_ended = True
                raise asyncio.TimeoutError

        monkeypatch.setattr(cli, "TelemetryCollector", _OneShot)
        target = tmp_path / "m.jsonl"
        old = os.umask(0o002)
        try:
            cli._run_headless("12345", SlurmwatchConfig(), str(target))
        finally:
            os.umask(old)
        assert target.exists(), "the preflight should have created it"
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o644, oct(mode)


class TestAppendingAcrossAReleaseBoundary:
    """The `--append` schema bridge, tested against a header a user actually has.

    Not a synthetic "old" header built by deleting a column from the current one: such a
    fixture re-derives itself every time the schema changes, so it can only test the
    machinery against itself. This is the VERBATIM 56-column header that slurmwatch
    1.2.0 writes, read out of /software/slurmwatch-1.2.0-el8-x86_64, against a build
    that writes 57 (`cpu_source`, added by SW-85).

    Both directions were verified end to end with the two real builds before this test
    was written, and neither is silent:

        newer build, older file:  "1 column(s) this build produces are not in that
                                   header and will be omitted: cpu_source"
        older build, newer file:  "1 column(s) in the file are not produced by this
                                   build and will be blank: cpu_source"

    which is what SW-25 asked for. This test exists so the NEXT column addition is
    exercised against a real historical file rather than a moving target.
    """

    RELEASED_1_2_0 = [
        "timestamp",
        "job_id",
        "job_name",
        "hostname",
        "elapsed_seconds",
        "time_limit_seconds",
        "partition",
        "owner",
        "account",
        "qos",
        "array_job_id",
        "array_task_id",
        "cpu_cores",
        "cpu_usage_ns",
        "cpu_percent",
        "cpu_effective_cores",
        "cpu_peak_effective_cores",
        "mem_current_bytes",
        "mem_limit_bytes",
        "mem_working_set_bytes",
        "mem_cache_bytes",
        "mem_source",
        "mem_cache_measured",
        "mem_percent",
        "mem_working_set_percent",
        "mem_peak_bytes",
        "mem_peak_is_lifetime",
        "mem_peak_working_set_bytes",
        "mem_oom_warning",
        "mem_oom_critical",
        "gpu_count",
        "gpu_count_requested",
        "usage_age_seconds",
        "usage_sampled",
        "gpu_active_count",
        "node_count",
        "node_index",
        "remote",
        "mock",
        "gpu_monitoring_available",
        "gpu_unavailable_reason",
        "gpu_node_count",
        "gpu_node_model",
        "gpu_allocated_indices",
        "fabric_kind",
        "fabric_link_rate_gbps",
        "fabric_link_rate_total_gbps",
        "fabric_ports",
        "fabric_rx_gbps",
        "fabric_tx_gbps",
        "gpu_interconnect",
        "gpu_interconnect_per_gpu_gbps",
        "gpu_nvlink_rx_gbps",
        "gpu_nvlink_tx_gbps",
        "gpu_pcie_rx_gbps",
        "gpu_pcie_tx_gbps",
    ]

    def test_the_released_header_is_a_subset_of_todays(self) -> None:
        """An additive-only schema is the promise that makes appending safe. If a change
        ever removes or renames a column this fails, and the release note has to say so
        — which is the point of pinning a shipped header rather than a derived one."""
        current = set(TelemetrySnapshot.csv_header(0))
        missing = [c for c in self.RELEASED_1_2_0 if c not in current]
        assert not missing, f"columns released in 1.2.0 are gone: {missing}"

    def test_todays_extra_columns_are_exactly_what_the_notes_claim(self) -> None:
        extra = [c for c in TelemetrySnapshot.csv_header(0) if c not in self.RELEASED_1_2_0]
        assert extra == ["cpu_source"], extra

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_appending_to_a_released_file_keeps_every_value_under_its_heading(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "released.csv"
        target.write_text(",".join(self.RELEASED_1_2_0) + "\r\n")
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(target), "csv", append=True))
        await _wait_for_lines(target, 2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10.0)
        with open(target, newline="") as fh:
            rows = list(csv.reader(fh))
        assert all(len(r) == 56 for r in rows), [len(r) for r in rows]
        header, data = rows[0], rows[1]
        # By NAME, not position: a positional append would shift every field after the
        # insertion point and still produce a file of the right width.
        assert data[header.index("job_id")] == "12345"
        assert data[header.index("mem_source")] in ("cgroup", "mock", "sstat", "proc")
        assert data[header.index("cpu_usage_ns")].isdigit()
        assert "cpu_source" not in header

    def test_the_difference_is_announced_not_silent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SW-25's requirement: say which columns the file cannot hold, by name."""
        target = tmp_path / "released.csv"
        target.write_text(",".join(self.RELEASED_1_2_0) + "\r\n")
        assert cli._csv_append_layout(str(target), "auto", 0, 0) is not None
        err = capsys.readouterr().err
        assert "cpu_source" in err, err
        assert "56 columns" in err and "57" in err, err


class TestTheDocumentedExitStatusIsTrue:
    """SW-88: the exit-status contract was coherent and undocumented.

    Measured against a live cluster before writing it down — same job, both formats,
    every reachable outcome:

        own running job     rc=0      foreign running job  rc=0
        step form           rc=0      PENDING job          rc=1
        already ended       rc=1      no such job          rc=1
        malformed id        rc=1      bad flag/value       rc=2

    and `csv` and `json` agree everywhere, which is the invariant SW-27 established when
    the foreign summary used to be rc=0 on one path and rc=1 on another.

    The part that needed saying is that **1 is not failure**: a PENDING job exits 1
    because there is no telemetry yet, and the payload's
    `telemetry_unavailable_reason` says which case it is. A `set -e` poller written
    against the obvious reading aborts on a perfectly normal queued job. Now stated in
    `--help`, and asserted here so the statement stays true.
    """

    @staticmethod
    def _rc(argv: list[str]) -> int:
        try:
            main(argv)
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_a_live_job_is_zero(self) -> None:
        assert self._rc(["12345", "--once"]) == 0
        assert self._rc(["12345", "--once", "--json"]) == 0

    @pytest.mark.parametrize("fmt", [[], ["--json"]], ids=["csv", "json"])
    def test_no_such_job_is_one_in_either_format(
        self, fmt: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Format-independence is SW-27's invariant; it is checked per case rather than
        once, because that finding was exactly one path disagreeing with another."""

        def _missing(*_a: object, **_k: object) -> object:
            raise JobNotFoundError("Job 99999999 not found")

        monkeypatch.setattr(cli, "resolve_job_context", _missing)
        assert self._rc(["99999999", "--once", *fmt]) == 1

    @pytest.mark.parametrize("fmt", [[], ["--json"]], ids=["csv", "json"])
    def test_a_malformed_id_is_one_not_two(
        self, fmt: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An id is DATA, not usage: argparse's 2 is reserved for a flag the tool does
        not have. Keeping them apart is what lets a script tell "you typed the wrong
        option" from "that job isn't there"."""

        def _missing(*_a: object, **_k: object) -> object:
            raise JobNotFoundError("nope")

        monkeypatch.setattr(cli, "resolve_job_context", _missing)
        assert self._rc(["not-an-id", "--once", *fmt]) == 1

    def test_bad_usage_is_two(self) -> None:
        assert self._rc(["--nosuchflag"]) == 2
        assert self._rc(["--interval", "abc"]) == 2

    def test_the_help_states_every_code_it_can_return(self) -> None:
        text = _build_parser().format_help()
        assert "exit status" in text
        for code in ("0 ", "1 ", "2 ", "128+N"):
            assert code in text, code
        # and the sentence that stops a poller mis-reading 1
        assert "A queued job is normal" in text
        assert "telemetry_unavailable_reason" in text


class TestBareSwSaysWhatIsActuallyThere:
    """SW-89: "nothing queued" and "your job is right there, unmonitorable" were one message.

    `sw` with no arguments filters the queue to R and PD, because those are the states it
    can monitor — the picker's comment says so deliberately. But when that filter emptied
    the list, the message was always:

        No running or pending Slurm jobs found for user 'youzhi'. Launch a job first or
        provide a job_id argument.

    A job in CG (COMPLETING, which can persist for minutes while a node cleans up), CF
    (CONFIGURING, during node boot) or S (SUSPENDED, after preemption) produced exactly
    that — so the tool told the user to launch a job while `squeue` sat on screen showing
    theirs. It had parsed the state and discarded it. Literally true, and it contradicts
    the command they just ran, which is the same shape as SW-28's tip telling a user to
    wait for something that can never happen.
    """

    @staticmethod
    def _message(rows: str, monkeypatch: pytest.MonkeyPatch) -> str:
        """Answers each query with the fields IT asked for, not one canned row.

        A fake that returns the same 8-field line for both calls made the state print as
        `CG|build|1|10:00|...` — which is how the loose `split("|", 1)` in the first
        version was found. A fake must mirror the call it is answering.
        """

        def _fake(cmd: list[str], *a: object, **k: object) -> str:
            fmt = cmd[cmd.index("-o") + 1] if "-o" in cmd else ""
            if fmt.count("|") > 1:  # resolve_current_jobs' wide format
                return rows
            return "\n".join(  # resolve_unmonitorable_jobs' %i|%t
                "|".join(ln.split("|")[:2]) for ln in rows.splitlines() if ln.strip()
            )

        monkeypatch.setattr(slurm_mod, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm_mod, "_run_slurm_cmd", _fake)
        monkeypatch.setattr(cli, "resolve_current_jobs", slurm_mod.resolve_current_jobs)
        monkeypatch.setattr(cli, "resolve_unmonitorable_jobs", slurm_mod.resolve_unmonitorable_jobs)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.suppress(SystemExit):
            cli._auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        return err.getvalue()

    @pytest.mark.parametrize(
        ("code", "word"),
        [("CG", "COMPLETING"), ("CF", "CONFIGURING"), ("S", "SUSPENDED"), ("PR", "PREEMPTED")],
    )
    def test_an_unmonitorable_job_is_named_with_its_state(
        self, code: str, word: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = self._message(f"54999001|{code}|build|1|10:00|1:00:00|n|my-run", monkeypatch)
        assert "54999001" in out, out
        assert word in out, out
        assert "Launch a job first" not in out, "that advice is for an empty queue"

    def test_an_empty_queue_still_gets_the_original_advice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = self._message("", monkeypatch)
        assert "No running or pending Slurm jobs found" in out
        assert "Launch a job first" in out

    def test_an_unknown_state_code_falls_back_to_the_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A code this build has never heard of must still be shown: the raw letters beat
        claiming there is nothing there."""
        out = self._message("54999009|ZZ|build|1|1|1|n|r", monkeypatch)
        assert "54999009" in out and "ZZ" in out, out

    def test_many_unmonitorable_jobs_are_capped_and_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = "\n".join(f"5499900{i}|CG|b|1|1|1|n|r" for i in range(6))
        out = self._message(rows, monkeypatch)
        assert "and 3 more" in out, out

    def test_a_running_job_is_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The filter itself must not change: an R job is still discovered, and the
        second query is not even reached."""
        out = self._message("54999010|R|build|1|10:00|1:00:00|n|my-run", monkeypatch)
        assert "monitorable state" not in out, out
