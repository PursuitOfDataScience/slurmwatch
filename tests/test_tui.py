from __future__ import annotations

import asyncio
import time

import pytest
from rich.markup import render as _render_markup
from textual.app import App
from textual.geometry import Size

from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot
from slurmwatch.tui import (
    _CPU_COLOR,
    _FAINT,
    _MEM_COLOR,
    DashboardScreen,
    EfficiencyPanel,
    GpuTable,
    HistoryPanel,
    ResourceDetailScreen,
    ResourceRows,
    StatusBanner,
    _banner_segments,
    _braille_line,
    _color_bar,
    _cpu_health,
    _format_bytes,
    _format_duration,
    _gpu_health,
    _mem_health,
    _render_sparkline,
    _stretch_columns,
)


def _valid_markup(text: str) -> None:
    """Rich must be able to parse the string; Textual parses it every render."""
    _render_markup(text)  # raises MarkupError on unbalanced/invalid markup


# ---------------------------------------------------------------------------
# Formatting / drawing primitives
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_bytes(self) -> None:
        assert _format_bytes(0) == "0.0 B"
        assert _format_bytes(1024) == "1.0 KiB"
        assert _format_bytes(1024**3) == "1.0 GiB"
        assert _format_bytes(1024**5) == "1.0 PiB"

    def test_format_duration(self) -> None:
        assert _format_duration(0) == "00:00:00"
        assert _format_duration(3661) == "01:01:01"
        assert _format_duration(86399) == "23:59:59"

    def test_color_bar_wears_the_block_identity_color(self) -> None:
        # A bar's fill is its block's identity hue (passed by the caller), not a
        # health color — only the fill *length* carries magnitude. The empty
        # track is the faint neutral. Health lives in the dot/word beside it.
        bar = _color_bar(50, 4, color=_CPU_COLOR)
        assert bar == f"[{_CPU_COLOR}]██[/][{_FAINT}]░░[/]"
        bar_mem = _color_bar(50, 4, color=_MEM_COLOR)
        assert _CPU_COLOR not in bar_mem and _MEM_COLOR in bar_mem  # color follows the block
        for health in ("#6aa84f", "#e2bb4c", "#d1584f"):
            assert health not in bar  # never a health color

    def test_color_bar_clamps_out_of_range(self) -> None:
        assert str(_render_markup(_color_bar(150, 12, color=_CPU_COLOR))).count("█") == 12
        assert _color_bar(-10, 4, color=_CPU_COLOR) == f"[{_FAINT}]░░░░[/]"

    def test_color_bar_ascii(self) -> None:
        assert _color_bar(50, 4, ascii_mode=True, color=_CPU_COLOR) == (
            f"[{_CPU_COLOR}]##[/][{_FAINT}]--[/]"
        )

    def test_render_sparkline_len_and_padding(self) -> None:
        from collections import deque

        assert _render_sparkline(deque(), 5) == " " * 5
        vals: deque[float] = deque([10.0, 50.0, 90.0])
        result = _render_sparkline(vals, 8)
        assert len(result) == 8
        assert result[:5] == " " * 5  # sparse history is left-padded, not stretched

    def test_render_sparkline_newest_at_right_edge(self) -> None:
        from collections import deque

        vals: deque[float] = deque([0.0] * 59 + [100.0], maxlen=60)
        assert _render_sparkline(vals, 16)[-1] == "█"


# ---------------------------------------------------------------------------
# Health vocabulary (one scale, computed in one place)
# ---------------------------------------------------------------------------


class TestHealth:
    def test_cpu_health(self) -> None:
        good = CpuMetrics(cores_allocated=16, usage_ns=1, usage_percent=66.0, effective_cores=10.5)
        assert _cpu_health(good) == ("ok", "healthy")
        idle = CpuMetrics(cores_allocated=16, usage_ns=1, usage_percent=1.0, effective_cores=0.5)
        assert _cpu_health(idle) == ("warn", "underused")
        single = CpuMetrics(cores_allocated=1, usage_ns=1, usage_percent=1.0, effective_cores=0.01)
        assert _cpu_health(single) == ("ok", "healthy")  # 1 core can't be "underused"

    def test_mem_health(self) -> None:
        def mem(warn: bool, crit: bool) -> MemoryMetrics:
            return MemoryMetrics(
                current_bytes=1,
                limit_bytes=10,
                peak_bytes=1,
                usage_percent=10.0,
                oom_guard_warning=warn,
                oom_guard_critical=crit,
                working_set_bytes=1,
            )

        assert _mem_health(mem(False, False)) == ("ok", "healthy")
        assert _mem_health(mem(True, False)) == ("warn", "high")
        assert _mem_health(mem(True, True)) == ("crit", "near limit")

    def test_gpu_health(self) -> None:
        active = _make_gpu(util=94.0, procmem=50 * 1024**3, memused=55 * 1024**3)
        assert _gpu_health(active, 5.0) == ("ok", "active")
        idle = _make_gpu(util=1.0, procmem=0, memused=0)
        assert _gpu_health(idle, 5.0) == ("crit", "idle")
        throttling = _make_gpu(util=94.0, procmem=50 * 1024**3, memused=55 * 1024**3, throttle=True)
        assert _gpu_health(throttling, 5.0) == ("warn", "throttling")


class TestBannerSegments:
    def test_healthy_is_empty(self) -> None:
        assert _banner_segments(_make_snapshot(), SlurmwatchConfig()) == []

    def test_mem_critical_is_first_and_worst(self) -> None:
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert segs[0][0] == "crit"
        assert "OOM RISK" in segs[0][1]

    def test_gpu_idle_and_all_idle(self) -> None:
        # 1 of 2 idle -> warn; all idle -> crit.
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(94.0, 50 * 1024**3, 55 * 1024**3), _make_gpu(1.0, 0, 0)]
        snap.gpu_count_requested = 2
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert any(lvl == "warn" and "1 OF 2 GPUS IDLE" in txt for lvl, txt in segs)

        snap.gpus = [_make_gpu(1.0, 0, 0), _make_gpu(1.0, 0, 0)]
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert any(lvl == "crit" and "ALL 2 GPUS IDLE" in txt for lvl, txt in segs)

    def test_crit_ordered_before_warn(self) -> None:
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        snap.gpus = [_make_gpu(94.0, 50 * 1024**3, 55 * 1024**3), _make_gpu(1.0, 0, 0)]
        snap.gpu_count_requested = 2
        segs = _banner_segments(snap, SlurmwatchConfig())
        levels = [lvl for lvl, _ in segs]
        assert levels.index("crit") < levels.index("warn")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class TestStatusBanner:
    def test_no_data(self) -> None:
        assert "connecting" in StatusBanner().render()

    def test_all_healthy(self) -> None:
        b = StatusBanner()
        b.snapshot = _make_snapshot()
        b.config = SlurmwatchConfig()
        out = b.render()
        assert "ALL HEALTHY" in out
        _valid_markup(out)

    def test_worst_first(self) -> None:
        b = StatusBanner()
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        b.snapshot = snap
        b.config = SlurmwatchConfig()
        out = b.render()
        assert "OOM RISK" in out and "ALL HEALTHY" not in out
        _valid_markup(out)

    def test_unobservable_gpu_is_not_a_false_alarm(self) -> None:
        # gpus=[] with gpu_count_requested>0 (remote / NVML off): a neutral note,
        # not a red/yellow alarm and not a false "0 idle".
        b = StatusBanner()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_active_count = 0
        snap.gpu_count_requested = 4
        b.snapshot = snap
        b.config = SlurmwatchConfig()
        out = b.render()
        assert "unavailable" in out
        assert "IDLE" not in out
        _valid_markup(out)


class TestResourceRows:
    def test_no_data(self) -> None:
        assert "awaiting" in ResourceRows().render()

    def test_renders_cpu_mem_gpu(self) -> None:
        r = ResourceRows()
        r.snapshot = _make_snapshot()
        r.config = SlurmwatchConfig()
        out = r.render()
        assert "CPU" in out and "MEM" in out and "GPU0" in out
        assert "16 cores" in out
        assert "72" in out  # gpu utilization
        _valid_markup(out)

    def test_table_active_suppresses_gpu_rows(self) -> None:
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        r.gpu_table_active = True
        out = r.render()
        assert "GPU0" not in out  # the DataTable owns GPU rows now
        assert "CPU" in out and "MEM" in out

    def test_unobservable_gpu_note(self) -> None:
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = r.render()
        assert "unavailable" in out and "2 requested" in out
        _valid_markup(out)


class TestEfficiencyPanel:
    def test_no_data(self) -> None:
        assert "awaiting" in EfficiencyPanel().render()

    def test_recommendations_and_source(self) -> None:
        e = EfficiencyPanel()
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        snap.gpus = [
            _make_gpu(94.0, 50 * 1024**3, 55 * 1024**3, index=0),
            _make_gpu(1.0, 0, 0, index=1),
        ]
        snap.gpu_count_requested = 2
        e.snapshot = snap
        e.config = SlurmwatchConfig()
        e.source = "cgroup v2"
        out = e.render()
        assert "Allocation efficiency" in out
        assert "critical" in out and "--mem" in out  # the actionable memory advice
        assert "--gres=gpu:1" in out  # the actionable GPU sentence
        assert "GPU 1 idle" in out  # names the specific idle device, not "1 of 2"
        assert "source: cgroup v2" in out
        _valid_markup(out)

    def test_unobservable_gpu(self) -> None:
        e = EfficiencyPanel()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 4
        e.snapshot = snap
        e.config = SlurmwatchConfig()
        out = e.render()
        assert "unavailable" in out
        assert "idle" not in out.lower()
        _valid_markup(out)


class TestBrailleLine:
    """The thin braille line chart that replaced the solid block-fill area."""

    @staticmethod
    def _is_braille(s: str) -> bool:
        # Every non-space glyph must be in the braille block U+2800…U+28FF.
        return all(ch == " " or 0x2800 <= ord(ch) <= 0x28FF for ch in s)

    def test_stretch_fills_width_before_history_is_full(self) -> None:
        # Fewer samples than columns must still fill the whole width (no blank
        # left margin), oldest sample on the left, newest on the right.
        from collections import deque

        cols = _stretch_columns(deque([10.0, 90.0]), width=8)
        assert len(cols) == 8
        assert cols[0] == 10.0 and cols[-1] == 90.0
        assert None not in cols

    def test_dimensions(self) -> None:
        from collections import deque

        rows = _braille_line(deque([50.0] * 30), width=20, height=4)
        assert len(rows) == 4
        assert all(len(r) == 20 for r in rows)

    def test_draws_braille_not_solid_blocks(self) -> None:
        # The whole point of the redesign: a line, drawn with braille dots, not
        # the '█' block wall the old area chart produced.
        from collections import deque

        rows = _braille_line(deque([20.0, 80.0, 40.0, 90.0, 10.0] * 4), width=16, height=3)
        joined = "".join(rows)
        assert "█" not in joined
        assert any(ch != " " for ch in joined)  # something was drawn
        assert self._is_braille(joined)

    def test_empty_history_is_blank(self) -> None:
        from collections import deque

        rows = _braille_line(deque(), width=8, height=3)
        assert rows == ["        "] * 3

    def test_ascii_mode_has_no_braille(self) -> None:
        from collections import deque

        rows = _braille_line(deque([50.0] * 10), width=10, height=3, ascii_mode=True)
        assert len(rows) == 3 and all(len(r) == 10 for r in rows)
        assert all(not (0x2800 <= ord(ch) <= 0x28FF) for ch in "".join(rows) if ch != " ")


class _SizedHistoryPanel(HistoryPanel):
    """HistoryPanel with a fixed size so render() can be unit-tested unmounted."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._test_size = Size(width, height)

    @property
    def size(self) -> Size:
        return self._test_size


class TestHistoryPanel:
    def _panel(self, width: int, height: int) -> _SizedHistoryPanel:
        from collections import deque

        panel = _SizedHistoryPanel(width, height)
        panel.snapshot = _make_snapshot()
        panel.config = SlurmwatchConfig()
        panel.cpu_history = deque([60.0] * 40, maxlen=120)
        panel.mem_history = deque([70.0] * 40, maxlen=120)
        panel.gpu_history = {0: deque([90.0] * 40, maxlen=120)}
        return panel

    def test_renders_tall_chart_for_each_series(self) -> None:
        out = self._panel(100, 18).render()
        # A titled panel with one labeled trend per resource, in its own color.
        assert "TRENDS" in out
        assert "CPU" in out and "MEM" in out and "GPU0" in out
        assert _CPU_COLOR in out and _MEM_COLOR in out  # each series in its block hue
        # The body is a braille line, not the old '█' block wall.
        assert "█" not in out
        assert any(0x2800 <= ord(ch) <= 0x28FF for ch in out)
        _valid_markup(out)

    def test_never_overflows_its_height(self) -> None:
        for h in (4, 8, 12, 30):
            out = self._panel(100, h).render()
            assert out.count("\n") + 1 <= h

    def test_too_small_is_blank(self) -> None:
        assert self._panel(100, 2).render() == ""


class TestCpuUnderuseThreshold:
    """F4: SLURMWATCH_CPU_UNDERUSE actually drives the underused verdict."""

    def test_threshold_is_wired(self) -> None:
        cpu = CpuMetrics(cores_allocated=16, usage_ns=0, usage_percent=30.0, effective_cores=4.8)
        # ratio = 0.3: healthy under the default 0.15, underused under a 0.5 bar.
        assert _cpu_health(cpu, 0.15) == ("ok", "healthy")
        assert _cpu_health(cpu, 0.5) == ("warn", "underused")

    def test_efficiency_panel_uses_threshold(self) -> None:
        e = EfficiencyPanel()
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=16, usage_ns=0, usage_percent=30.0, effective_cores=4.8
        )
        e.snapshot = snap
        e.config = SlurmwatchConfig(cpu_underuse_threshold=0.5)
        assert "underused" in _render_markup(e.render()).plain
        e.config = SlurmwatchConfig(cpu_underuse_threshold=0.15)
        # Assert on the plain-rendered text: the grade column now carries per-
        # block colour markup between the label and the grade word.
        cpu_line = _render_markup(e.render()).plain.splitlines()[1]
        assert "CPU" in cpu_line and "good" in cpu_line and "underused" not in cpu_line


class TestMarkupValidity:
    """Every panel must emit valid Rich markup in every state Textual renders."""

    def test_all_panels_all_states(self) -> None:
        for warn, crit in [(False, False), (True, False), (True, True)]:
            snap = _make_snapshot()
            snap.memory.oom_guard_warning = warn
            snap.memory.oom_guard_critical = crit
            for cls in (StatusBanner, ResourceRows, EfficiencyPanel):
                w = cls()
                w.snapshot = snap
                w.config = SlurmwatchConfig()
                _valid_markup(w.render())

    def test_throttling_marker_has_negative_control(self) -> None:
        # B-T9: assert the throttle marker *appears* when throttling and
        # *disappears* when not, so a stray '!' can't make the test pass.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].throttling = True
        snap.gpus[0].temperature_celsius = 88.0  # hot -> '!' marker
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        hot = r.render()
        assert "throttling" in hot
        assert "88°C!" in hot

        snap.gpus[0].throttling = False
        snap.gpus[0].temperature_celsius = 60.0
        r.snapshot = snap
        cool = r.render()
        assert "throttling" not in cool
        assert "!" not in cool
        assert "60°C" in cool


# ---------------------------------------------------------------------------
# Integration (Textual Pilot)
# ---------------------------------------------------------------------------


class _StubCollector:
    def __init__(self, raise_once: bool = False) -> None:
        self.config = SlurmwatchConfig()
        self._mock = True
        self._raise_once = raise_once
        self._raised = False

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def stop_sync(self) -> None: ...

    async def next_snapshot(self) -> TelemetrySnapshot:
        if self._raise_once and not self._raised:
            self._raised = True
            raise RuntimeError("transient collector failure")
        await asyncio.sleep(3600)  # driven manually in the tests
        raise RuntimeError


class _DashApp(App[None]):
    def __init__(self, collector: _StubCollector, job: JobContext) -> None:
        super().__init__()
        self.scr = DashboardScreen(collector, job, collector.config)  # type: ignore[arg-type]

    async def on_mount(self) -> None:
        await self.push_screen(self.scr)


def _dash_app(collector: _StubCollector, gpus: int = 1) -> _DashApp:
    job = JobContext(
        job_id="12345",
        username="ada",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=gpus,
        gpu_indices=list(range(gpus)),
        step_id="0",
        uid=1001,
        job_start_time=time.time() - 3600,
        nodelist_resolved=["cn001"],
        cgroup_v2_path="/x",
    )
    return _DashApp(collector, job)


class TestDashboardIntegration:
    @pytest.mark.asyncio
    async def test_renders_snapshot_and_header(self) -> None:
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            assert app.scr.query_one(StatusBanner).snapshot is not None
            assert app.scr.latest_snapshot is not None
            assert "12345" in str(app.scr.sub_title)

    @pytest.mark.asyncio
    async def test_memory_sparkline_tracks_working_set_not_usage(self) -> None:
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            snap = _make_snapshot()  # ws=28 GiB, current=32 GiB, limit=64 GiB
            snap.memory.usage_percent = 50.0
            app.scr._update_widgets(snap)
            await pilot.pause()
            hist = app.scr.query_one(ResourceRows).mem_history
            assert hist and abs(hist[-1] - 43.75) < 0.01  # 28/64, not the 50% usage

    @pytest.mark.asyncio
    async def test_datatable_used_for_three_or_more_gpus(self) -> None:
        app = _dash_app(_StubCollector(), gpus=4)
        async with app.run_test() as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
            snap.gpu_count_requested = 4
            app.scr._update_widgets(snap)
            await pilot.pause()
            table = app.scr.query_one(GpuTable)
            assert table.display is True
            assert table.row_count == 4

    @pytest.mark.asyncio
    async def test_two_gpus_use_rows_not_table(self) -> None:
        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
            snap.gpu_count_requested = 2
            app.scr._update_widgets(snap)
            await pilot.pause()
            assert app.scr.query_one(GpuTable).display is False

    @pytest.mark.asyncio
    async def test_drill_in_opens_detail_screen(self) -> None:
        # Regression: the focus keys used to only recolor a border. Now c/m/g
        # push a real detail screen.
        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            assert isinstance(app.screen, ResourceDetailScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

    @pytest.mark.asyncio
    async def test_gpu_detail_shows_job_share_columns(self) -> None:
        # F6: drilling into GPU shows the job's per-device share (JOB% / JOB
        # VRAM), which the dashboard overview doesn't — so the keystroke pays off.
        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            assert isinstance(app.screen, ResourceDetailScreen)
            table = app.screen.query_one("#detail-table", GpuTable)
            labels = [str(c.label) for c in table.columns.values()]
            assert "JOB%" in labels and "JOB VRAM" in labels

    @pytest.mark.asyncio
    async def test_detail_chart_fits_its_box_width(self) -> None:
        # F2: the detail history chart is sized to the box's content width, so it
        # never wraps into broken fragments. Every rendered chart line must fit.
        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, ResourceDetailScreen)
            # Re-render now that the box is laid out (a real size), which is the
            # steady state the timer keeps it in.
            app.screen._refresh()
            await pilot.pause()
            chart = app.screen.query_one("#detail-chart")
            box_w = chart.size.width
            assert box_w > 0
            body_lines = _render_markup(str(chart.render())).plain.split("\n")
            # No chart row exceeds the widget width (would otherwise wrap, F2).
            assert all(len(line) <= box_w for line in body_lines)

    @pytest.mark.asyncio
    async def test_dashboard_gpu_table_has_no_row_cursor(self) -> None:
        # U5: the overview table isn't interactive, so it must not show an
        # always-on row highlight implying a selection that does nothing.
        app = _dash_app(_StubCollector(), gpus=3)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(3)]
            app.scr._update_widgets(snap)
            await pilot.pause()
            table = app.scr.query_one(GpuTable)
            assert table.cursor_type == "none"

    @pytest.mark.asyncio
    async def test_poll_loop_survives_transient_exception(self) -> None:
        # B-C7: one bad next_snapshot() must not silently kill all UI updates.
        collector = _StubCollector(raise_once=True)
        app = _dash_app(collector)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Let the poll loop hit the raising call and recover.
            for _ in range(5):
                await pilot.pause(0.05)
            assert app.scr._poll_task is not None
            assert not app.scr._poll_task.done()  # still polling, not dead


class TestJobSelectorFlow:
    JOBS: list[dict[str, object]] = [
        {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
        {"job_id": "12345", "state": "R", "partition": "gpu", "name": "b", "nodes": "1"},
    ]

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_enter_selects_job_and_opens_dashboard(self) -> None:
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
    async def test_bracketed_job_name_does_not_crash_selector(self) -> None:
        # F1: a job name with markup metacharacters must not crash the selector
        # or corrupt the render. Textual's markup parser (unlike Rich's) also
        # treats a *lone/unclosed* '[' (sbatch -J '[experiment') as a tag opener
        # and raises MarkupError mid-render — the crash class the escape must
        # cover. The mount below renders through Textual's real engine, so it
        # would raise without the fix.
        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        hostile = ["run[/]done", "sweep[3]", "[experiment", "[", "100%[x", "[red]x"]
        jobs: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "safe", "nodes": "1"}
        ]
        jobs += [
            {"job_id": str(200 + i), "state": "R", "partition": "gpu", "name": n, "nodes": "1"}
            for i, n in enumerate(hostile)
        ]
        app = SlurmwatchApp(jobs=jobs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            # Read the real rendered character grid; every name survives literally.
            app.screen.text_select_all()
            shown = app.screen.get_selected_text() or ""
        for name in hostile:
            assert name in shown, f"{name!r} not rendered literally: {shown!r}"

    def test_escape_markup_neutralizes_lone_bracket(self) -> None:
        from slurmwatch.tui import _escape_markup

        assert _escape_markup("[experiment") == r"\[experiment"
        assert _escape_markup("a[b]c") == r"a\[b]c"
        assert _escape_markup(r"back\slash[x") == "back\\\\slash\\[x"

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
            assert app.screen.config is config
            assert app._collector.config is config


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _make_gpu(
    util: float,
    procmem: int,
    memused: int,
    memtot: int = 40 * 1024**3,
    throttle: bool = False,
    index: int = 0,
) -> GpuMetrics:
    return GpuMetrics(
        index=index,
        uuid=f"GPU-{index}",
        name="A100-SXM4-40GB",
        utilization_percent=util,
        memory_used_bytes=memused,
        memory_total_bytes=memtot,
        memory_utilization_percent=round(memused / memtot * 100, 1) if memtot else 0.0,
        power_watts=250.0,
        temperature_celsius=65.0,
        throttling=throttle,
        process_utilization_percent=util if procmem > 0 else 0.0,
        process_memory_bytes=procmem,
    )


def _make_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(
            cores_allocated=16, usage_ns=1_000_000_000, usage_percent=50.0, effective_cores=8.0
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
        gpus=[_make_gpu(72.5, 18 * 1024**3, 20 * 1024**3)],
        gpu_count_requested=1,
        gpu_active_count=1,
    )
