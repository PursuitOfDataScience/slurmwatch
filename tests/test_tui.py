from __future__ import annotations

import asyncio
import time

import pytest
from textual.app import App

from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot
from slurmwatch.tui import (
    CpuPanel,
    DashboardScreen,
    GpuPanel,
    MemoryPanel,
    VerdictPanel,
    _color_bar,
    _format_bytes,
    _format_duration,
    _heat_color,
    _render_sparkline,
)


class TestHelpers:
    def test_format_bytes_bytes(self) -> None:
        assert _format_bytes(0) == "0.0 B"
        assert _format_bytes(500) == "500.0 B"

    def test_format_bytes_kib(self) -> None:
        assert _format_bytes(1024) == "1.0 KiB"
        assert _format_bytes(2048) == "2.0 KiB"

    def test_format_bytes_mib(self) -> None:
        assert _format_bytes(1024 * 1024) == "1.0 MiB"

    def test_format_bytes_gib(self) -> None:
        assert _format_bytes(1024**3) == "1.0 GiB"

    def test_format_bytes_tib(self) -> None:
        assert _format_bytes(1024**4) == "1.0 TiB"

    def test_format_bytes_pib(self) -> None:
        assert _format_bytes(1024**5) == "1.0 PiB"

    def test_format_duration(self) -> None:
        assert _format_duration(0) == "00:00:00"
        assert _format_duration(3661) == "01:01:01"
        assert _format_duration(86399) == "23:59:59"

    def test_color_bar_half(self) -> None:
        assert _color_bar(50, 4, color="green") == "[green]██[/][dim]░░[/]"

    def test_color_bar_full_and_empty(self) -> None:
        assert _color_bar(100, 4, color="red") == "[red]████[/]"
        assert _color_bar(0, 4, color="red") == "[dim]░░░░[/]"

    def test_color_bar_rounded(self) -> None:
        from rich.markup import render

        bar = _color_bar(33, 6, color="green")
        rendered = str(render(bar))  # raises on invalid markup
        assert rendered.count("█") == 1
        assert rendered.count("░") == 5

    def test_color_bar_ascii(self) -> None:
        assert _color_bar(50, 4, ascii_mode=True, color="cyan") == "[cyan]##[/][dim]--[/]"

    def test_color_bar_clamps_out_of_range(self) -> None:
        # Memory usage can exceed 100% when the cgroup limit is unenforced;
        # the bar must not overflow its width.
        from rich.markup import render

        assert _color_bar(150, 4, color="red") == "[red]████[/]"
        assert _color_bar(-10, 4, color="red") == "[dim]░░░░[/]"
        assert str(render(_color_bar(150, 12, color="red"))).count("█") == 12

    def test_heat_color(self) -> None:
        assert _heat_color(10) == "green"
        assert _heat_color(70) == "yellow"
        assert _heat_color(90) == "red"

    def test_render_sparkline(self) -> None:
        from collections import deque

        vals: deque[float] = deque([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        result = _render_sparkline(vals, 5)
        assert len(result) == 5

    def test_render_sparkline_empty(self) -> None:
        from collections import deque

        result = _render_sparkline(deque(), 5)
        assert result == " " * 5

    def test_render_sparkline_scaled_to_100(self) -> None:
        from collections import deque

        vals: deque[float] = deque([5.0])
        result_5 = _render_sparkline(vals, 3)
        assert len(result_5) == 3
        vals2: deque[float] = deque([95.0])
        result_95 = _render_sparkline(vals2, 3)
        assert result_5 != result_95  # different values produce different patterns

    def test_render_sparkline_newest_at_right_edge(self) -> None:
        from collections import deque

        # 60 samples of 0 with a spike in the newest: the right edge must
        # show the spike (the old sampling could never reach the newest few).
        vals: deque[float] = deque([0.0] * 59 + [100.0], maxlen=60)
        result = _render_sparkline(vals, 16)
        assert result[-1] == "█"

    def test_render_sparkline_pads_sparse_history(self) -> None:
        from collections import deque

        # 3 samples must not be stretched into a full-width fake history.
        vals: deque[float] = deque([10.0, 50.0, 90.0])
        result = _render_sparkline(vals, 8)
        assert len(result) == 8
        assert result[:5] == " " * 5  # left-padded until history fills


class TestCpuPanel:
    def test_render_no_data(self) -> None:
        panel = CpuPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_with_data(self) -> None:
        panel = CpuPanel()
        panel.snapshot = _make_snapshot()
        rendered = panel.render()
        assert "CPU" in rendered
        assert "16 cores" in rendered
        assert "50.0%" in rendered
        assert "effective" in rendered


class TestMemoryPanel:
    def test_render_no_data(self) -> None:
        panel = MemoryPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_normal(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 50.0
        snap.memory.oom_guard_warning = False
        snap.memory.oom_guard_critical = False
        panel.snapshot = snap
        rendered = panel.render()
        assert "MEMORY" in rendered
        assert "WARNING" not in rendered
        assert "CRITICAL" not in rendered
        assert "working set" in rendered

    def test_render_warning_threshold(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 86.0
        snap.memory.oom_guard_warning = True
        snap.memory.oom_guard_critical = False
        panel.snapshot = snap
        rendered = panel.render()
        assert "WARNING" in rendered
        assert "CRITICAL" not in rendered

    def test_render_critical_threshold(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 95.0
        snap.memory.oom_guard_warning = False
        snap.memory.oom_guard_critical = True
        panel.snapshot = snap
        rendered = panel.render()
        assert "CRITICAL" in rendered


class TestGpuPanel:
    def test_render_no_data(self) -> None:
        panel = GpuPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_no_gpus(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus = []
        panel.snapshot = snap
        rendered = panel.render()
        assert "no GPUs" in rendered

    def test_render_with_gpus(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        panel.snapshot = snap
        rendered = panel.render()
        assert "GPU 0" in rendered
        assert "A100" in rendered
        assert "72.5%" in rendered

    def test_render_throttling(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus[0].throttling = True
        panel.snapshot = snap
        rendered = panel.render()
        assert "!" in rendered or "⚠" in rendered

    def test_render_process_util(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus[0].process_utilization_percent = 60.0
        snap.gpus[0].process_memory_bytes = 18 * 1024**3
        panel.snapshot = snap
        rendered = panel.render()
        assert "proc:" in rendered


class TestVerdictPanel:
    def test_render_no_data(self) -> None:
        panel = VerdictPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_with_snapshot(self) -> None:
        panel = VerdictPanel()
        panel.snapshot = _make_snapshot()
        rendered = panel.render()
        assert "Allocation Efficiency" in rendered
        assert "CPU" in rendered
        assert "Memory" in rendered
        assert "GPU" in rendered


class TestMarkupValidity:
    """Panel output must be valid Rich markup; Textual parses it on every render."""

    @staticmethod
    def _check(text: str) -> None:
        from rich.markup import render

        render(text)  # raises MarkupError on unbalanced/invalid markup

    def test_all_panels_normal(self) -> None:
        snap = _make_snapshot()
        for panel_cls in (CpuPanel, MemoryPanel, GpuPanel, VerdictPanel):
            panel = panel_cls()
            panel.snapshot = snap
            self._check(panel.render())

    def test_memory_panel_all_oom_states(self) -> None:
        for warn, crit in [(False, False), (True, False), (True, True)]:
            snap = _make_snapshot()
            snap.memory.oom_guard_warning = warn
            snap.memory.oom_guard_critical = crit
            panel = MemoryPanel()
            panel.snapshot = snap
            self._check(panel.render())

    def test_gpu_panel_idle_and_throttling(self) -> None:
        snap = _make_snapshot()
        snap.gpus[0].throttling = True
        snap.gpus[0].utilization_percent = 1.0
        snap.gpus[0].process_utilization_percent = 0.5
        panel = GpuPanel()
        panel.snapshot = snap
        self._check(panel.render())


class _StubCollector:
    def __init__(self) -> None:
        self.config = SlurmwatchConfig()

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def stop_sync(self) -> None: ...
    async def next_snapshot(self) -> TelemetrySnapshot:
        await asyncio.sleep(3600)  # UI is driven manually in the test
        raise RuntimeError


class TestDashboardIntegration:
    @pytest.mark.asyncio
    async def test_dashboard_renders_snapshot(self) -> None:
        job = JobContext(
            job_id="12345",
            username="ada",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=64 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=time.time() - 3600,
            nodelist_resolved=["cn001"],
        )
        coll = _StubCollector()

        class _App(App):  # type: ignore[type-arg]
            def __init__(self) -> None:
                super().__init__()
                self.scr = DashboardScreen(coll, job, coll.config)  # type: ignore[arg-type]

            async def on_mount(self) -> None:
                await self.push_screen(self.scr)

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            assert app.scr.query_one("#cpu-panel", CpuPanel).snapshot is not None
            assert app.scr.query_one("#verdict-panel", VerdictPanel).snapshot is not None
            header = app.scr.query_one("#header").render()
            assert "12345" in str(header)


class TestJobSelectorFlow:
    JOBS: list[dict[str, object]] = [
        {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
        {"job_id": "12345", "state": "R", "partition": "gpu", "name": "b", "nodes": "1"},
    ]

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_enter_selects_job_and_opens_dashboard(self) -> None:
        # Regression: the selector used to die with NoActiveWorker (startup
        # ran outside a Textual worker) and Enter was swallowed by ListView.
        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        app = SlurmwatchApp(jobs=self.JOBS)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            await pilot.press("down")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, DashboardScreen):
                    break
            assert isinstance(app.screen, DashboardScreen)
            assert app.screen.job_ctx.job_id == "12345"

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_escape_cancels(self) -> None:
        from slurmwatch.tui import SlurmwatchApp

        app = SlurmwatchApp(jobs=self.JOBS)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
        assert app.return_code == 0

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_selector_threads_config_through(self) -> None:
        from slurmwatch.config import SlurmwatchConfig
        from slurmwatch.tui import SlurmwatchApp

        config = SlurmwatchConfig(poll_interval=0.05, ascii_mode=True)
        app = SlurmwatchApp(jobs=self.JOBS, config=config)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, DashboardScreen):
                    break
            assert isinstance(app.screen, DashboardScreen)
            # Regression: the selector path used to discard the CLI/env
            # config and fall back to defaults.
            assert app.screen.config is config
            assert app._collector.config is config


def _make_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(
            cores_allocated=16,
            usage_ns=1_000_000_000,
            usage_percent=50.0,
            effective_cores=8.0,
        ),
        memory=MemoryMetrics(
            current_bytes=32 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=50.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=28 * 1024**3,
            cache_bytes=4 * 1024**3,
        ),
        gpus=[
            GpuMetrics(
                index=0,
                uuid="GPU-abc123",
                name="A100-SXM4-40GB",
                utilization_percent=72.5,
                memory_used_bytes=20 * 1024**3,
                memory_total_bytes=40 * 1024**3,
                memory_utilization_percent=50.0,
                power_watts=250.0,
                temperature_celsius=65.0,
                throttling=False,
                process_utilization_percent=60.0,
                process_memory_bytes=18 * 1024**3,
            ),
        ],
        gpu_count_requested=4,
        gpu_active_count=1,
    )
