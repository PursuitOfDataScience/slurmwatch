from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path

import pytest

from slurmwatch.cli import _build_parser, _headless_loop, main
from slurmwatch.config import SlurmwatchConfig
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

    def test_config_from_env_gpu_idle(self) -> None:
        os.environ["SLURMWATCH_GPU_IDLE_PCT"] = "10.0"
        try:
            config = SlurmwatchConfig.from_env()
            assert config.gpu_idle_threshold == 10.0
        finally:
            del os.environ["SLURMWATCH_GPU_IDLE_PCT"]


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


class TestHeadlessLoop:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_headless_writes_jsonl(self, tmp_path: Path) -> None:
        ctx = resolve_job_context("12345")
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        out = tmp_path / "metrics.jsonl"
        task = asyncio.create_task(_headless_loop(ctx, cfg, str(out), ""))
        await asyncio.sleep(0.3)
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
        await asyncio.sleep(0.3)
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
        await asyncio.sleep(0.3)
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
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        lines = out.read_text().strip().split("\n")
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1])["job_id"] == "12345"
