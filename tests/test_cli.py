from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import threading
from pathlib import Path
from typing import Any

import pytest

import slurmwatch.cli as cli
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
    _job_owner_differs,
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

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_append_reuses_existing_header_width(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #62 regression: a run whose job has 4 GPUs, appended to a file whose
        # header was written for 2 GPUs, must write 2-GPU-wide rows so every row
        # still lines up under the header — not 4-GPU-wide rows that overflow it.
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

        class _StuckFile:
            def __enter__(self) -> _StuckFile:
                return self

            def __exit__(self, *exc: object) -> None:
                pass

            def tell(self) -> int:
                return 0

            def write(self, _data: str) -> int:
                write_started.set()
                release.wait(timeout=10.0)  # wedged sink; bounded so pytest can't hang
                return 0

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        monkeypatch.setattr(cli, "open", lambda *a, **k: _StuckFile(), raising=False)
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
        assert _hop_to_compute_node(self._ctx(), args) is True

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
        args = _build_parser().parse_args(["12345_3"])
        # If the timeout weren't caught, this would raise instead of returning.
        assert _hop_to_compute_node(self._ctx(), args) is True

    def test_no_hop_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_tty(monkeypatch)
        monkeypatch.setenv("SLURMWATCH_NO_HOP", "1")
        args = _build_parser().parse_args(["12345_3"])
        assert _hop_to_compute_node(self._ctx(), args) is False

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
        assert _hop_to_compute_node(self._ctx(), args) is False

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
        assert _hop_to_compute_node(self._ctx(), args) is True
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
        assert _hop_to_compute_node(self._ctx(), args) is True
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
        assert _hop_to_compute_node(self._ctx(), args) is False
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
        assert _hop_to_compute_node(self._ctx(), args) is True  # clean -> no summary
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
        assert _hop_to_compute_node(self._ctx(), args) is True
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

    def test_once_foreign_job_exits_nonzero_no_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # M2: --once on another user's job must not emit an all-zero telemetry row.
        # Print the honest summary to stderr and exit non-zero instead.
        ctx = self._ctx(owner="yifchen")
        ctx.uid = 5000
        monkeypatch.setattr(cli, "_resolve_running_or_pending", lambda _id: (ctx, None))
        monkeypatch.setattr("os.getuid", lambda: 4242)

        def _no_collector(*_a: Any, **_k: Any) -> None:
            raise AssertionError("must not build a collector for another user's job")

        monkeypatch.setattr(cli, "TelemetryCollector", _no_collector)
        with pytest.raises(SystemExit) as exc:
            cli._run_once("52211701_20", SlurmwatchConfig(), fmt="json")
        assert exc.value.code == 1
        cap = capsys.readouterr()
        assert cap.out == ""  # no zero JSON/CSV row on stdout
        assert "another user's job" in cap.err  # honest summary on stderr

    def test_headless_foreign_job_exits_without_writing_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # M2: --log on another user's job must not create a log of all-zero rows.
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
        assert not log.exists()  # no log file created
        assert "another user's job" in capsys.readouterr().err


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
        assert "SLURMWATCH_NO_HOP=1" in remote and "SLURMWATCH_NO_SSH=1" in remote
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

    def test_remote_exit_with_live_job_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._tty(monkeypatch, on=True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ssh")

        class _R:
            returncode = 1

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        monkeypatch.setattr(cli, "is_job_active", lambda jid: True)
        assert _ssh_to_compute_node(self._ctx(), self._args()) is False

    def test_ladder_prefers_ssh_over_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When the srun hop can't attach, the ssh rung runs BEFORE the sstat summary.
        ctx = self._ctx()
        monkeypatch.setattr(cli, "resolve_job_context", lambda job_id: ctx)
        monkeypatch.setattr("getpass.getuser", lambda: "u")
        monkeypatch.setattr(cli, "_hop_to_compute_node", lambda *a: False)
        ssh_called: dict[str, bool] = {}

        def _ssh(*a: Any, **k: Any) -> bool:
            ssh_called["yes"] = True
            return True

        monkeypatch.setattr(cli, "_ssh_to_compute_node", _ssh)

        def _no_summary(*a: Any, **k: Any) -> None:
            raise AssertionError("summary must not run when ssh succeeds")

        monkeypatch.setattr(cli, "_run_remote_summary", _no_summary)
        main(["123"])
        assert ssh_called.get("yes") is True
