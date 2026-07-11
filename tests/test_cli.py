from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import slurmwatch.cli as cli
from slurmwatch.cli import (
    _auto_discover_job_id,
    _build_parser,
    _console_logging_suspended,
    _env_disables_hop,
    _env_output_format,
    _headless_loop,
    _hop_to_compute_node,
    _infer_use_json,
    _resolve_or_die,
    main,
)
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import (
    CgroupNotFoundError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.model import JobContext
from slurmwatch.slurm import resolve_job_context


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
        main(["--demo", "12345"])
        assert captured.get("mouse") is False

    def test_tui_mouse_env_enables_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import slurmwatch.tui as tui

        monkeypatch.setenv("SLURMWATCH_MOUSE", "1")
        captured: dict[str, object] = {}
        monkeypatch.setattr(tui.SlurmwatchApp, "run", lambda self, *a, **k: captured.update(k))
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


class TestAutoDiscover:
    """B-T8: the advertised no-job-id default is never hit under SLURMWATCH_MOCK."""

    def test_no_jobs_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [])
        with pytest.raises(SystemExit) as exc:
            _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert exc.value.code == 1

    def test_single_job_auto_attaches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "resolve_current_jobs", lambda username=None: [{"job_id": "777"}])
        assert _auto_discover_job_id(SlurmwatchConfig(), interactive=False) == "777"

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

        # cli looks these up on their modules at call time, so patching the real
        # modules (rather than re-exported names on cli) is what takes effect.
        monkeypatch.setattr("sys.stdin", _TTY())
        monkeypatch.setattr("sys.stdout", _TTY())
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun")

    def test_builds_command_and_sanitizes_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        captured: dict[str, Any] = {}

        class _Result:
            returncode = 0

        def _fake_run(cmd: list[str], env: dict[str, str] | None = None) -> _Result:
            captured["cmd"] = cmd
            captured["env"] = env
            return _Result()

        monkeypatch.setattr("subprocess.run", _fake_run)
        monkeypatch.setenv("SLURM_NTASKS", "8")
        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)

        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) is True

        cmd = captured["cmd"]
        assert "--jobid=12348" in cmd  # numeric raw id, not the 12345_3 form
        assert "--overlap" in cmd
        assert "--nodelist=cn007" in cmd
        assert "-m" in cmd and "slurmwatch" in cmd
        assert "12345_3" in cmd  # the inner positional keeps the user's form

        env = captured["env"]
        assert env is not None
        assert "SLURM_NTASKS" not in env  # surrounding allocation sizing dropped
        assert env["SLURM_CONF"] == "/etc/slurm/slurm.conf"  # but SLURM_CONF kept
        assert env["SLURMWATCH_NO_HOP"] == "1"  # child can't re-hop

    def test_no_hop_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) is False

    def test_nonzero_exit_falls_back_to_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-P8: any non-clean srun exit (attach refused, node gone) must return
        # False so the caller shows the remote summary instead of a blank screen.
        self._force_tty(monkeypatch)
        monkeypatch.delenv("SLURMWATCH_NO_HOP", raising=False)

        class _Result:
            returncode = 1

        monkeypatch.setattr("subprocess.run", lambda cmd, env=None: _Result())
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) is False


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
