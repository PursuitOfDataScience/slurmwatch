from __future__ import annotations

import asyncio
import os
import time

import pytest
from rich.markup import render as _render_markup
from textual.app import App
from textual.geometry import Size

from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot
from slurmwatch.tui import (
    _ACCENT,
    _CPU_COLOR,
    _FAINT,
    _GPU_COLOR,
    _HEALTH_COLOR,
    _MEM_COLOR,
    DashboardScreen,
    GpuTable,
    JobDetailsPanel,
    JobInfoBar,
    KeyFooter,
    ResourceDetailScreen,
    ResourceRows,
    StatusBanner,
    SwitchBanner,
    _banner_segments,
    _bar_cells,
    _color_bar,
    _cpu_health,
    _format_bytes,
    _format_duration,
    _gpu_health,
    _mem_health,
    _pack_chips,
    _render_sparkline,
    _shorten_path,
)


def _valid_markup(text: str) -> None:
    """Rich must be able to parse the string; Textual parses it every render."""
    _render_markup(text)  # raises MarkupError on unbalanced/invalid markup


def _plain(markup: str) -> str:
    """Rendered plain text with non-breaking spaces normalised to spaces, so
    'label value' assertions don't care that the UI binds each label to its value
    with a NBSP (which keeps a chip from wrapping apart across a line break)."""
    return _render_markup(markup).plain.replace("\N{NO-BREAK SPACE}", " ")


@pytest.fixture(autouse=True)
def _no_real_srun(monkeypatch: pytest.MonkeyPatch) -> None:
    # The node switcher streams a remote node via srun. In tests the fake node
    # lists never include the test host, so every node reads as "remote" — stub
    # the stream launcher so no test spawns a real srun (streaming tests that
    # want a fake process override this with their own setattr).
    async def _none(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr("slurmwatch.tui.open_stream", _none)


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

    def test_small_value_gauge_is_not_empty(self) -> None:
        # The reported bug: a low-but-nonzero value (e.g. MEM 4%) drew an EMPTY
        # RESOURCES gauge (int-floor with no minimum: 4% of 18 = 0 cells) while
        # showing "4%" beside it. A value that displays as >= 1% must keep at
        # least one filled cell so the bar matches its own number.
        assert _bar_cells(4.0, 18) >= 1
        assert _render_markup(_color_bar(4.0, 18, color=_MEM_COLOR)).plain.count("█") >= 1
        # A sub-0.5% value that rounds to "0%" still draws empty, matching its label.
        assert _bar_cells(0.3, 18) == 0

    def test_bar_uses_round_and_min_cell_rule(self) -> None:
        # The bar's fill length is the shared _bar_cells rule: round (not floor)
        # to the nearest cell, and a value that DISPLAYS as >= 1% keeps at least
        # one filled cell. So the on-screen bar always agrees with the whole
        # percent printed beside it, at any width.
        for width in (18, 30, 74):
            for pct in (0.0, 0.3, 1.0, 4.0, 12.0, 50.0, 99.0, 100.0):
                cells = _render_markup(_color_bar(pct, width, color=_CPU_COLOR)).plain.count("█")
                assert cells == _bar_cells(pct, width)
                if round(pct) >= 1:
                    assert cells >= 1  # a displayed >=1% is never an empty bar
                else:
                    assert cells == 0  # a genuine 0% is empty

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

    def test_render_sparkline_stretch_fills_width(self) -> None:
        from collections import deque

        # stretch=True spreads a few samples across the whole width (no blank
        # left margin), so a trend fills the row instead of hugging the right.
        out = _render_sparkline(deque([10.0, 90.0]), 8, stretch=True)
        assert len(out) == 8
        assert " " not in out  # every column drawn, no blank margin
        # Oldest sample on the left, newest on the right: a low→high history must
        # rise left→right, so the first cell is shorter than the last.
        ramp = "▁▂▃▄▅▆▇█"
        assert ramp.index(out[0]) < ramp.index(out[-1])


def _has_bar(line: str) -> bool:
    # A rendered resource row carries a horizontal magnitude bar (fill + faint
    # track), so its plain text contains the block glyphs.
    return "░" in line or "█" in line


def _fill_cells(markup: str) -> int:
    # Count the solid fill cells (the current level) in a bar's plain text.
    return _render_markup(markup).plain.count("█")


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

    def test_gpu_device_colors_distinct_up_to_eight_then_cycle(self) -> None:
        from slurmwatch.tui import _GPU_CYCLE, _gpu_device_color

        # A full DGX-class node (8 GPUs, the most the tool tabulates) must give
        # every device its own colour — no repeats.
        colors = [_gpu_device_color(i) for i in range(8)]
        assert len(set(colors)) == 8
        assert len(_GPU_CYCLE) == 8
        # A 9th device wraps rather than crashing (cosmetic, and very rare).
        assert _gpu_device_color(8) == _gpu_device_color(0)


class TestBannerSegments:
    def test_healthy_is_empty(self) -> None:
        assert _banner_segments(_make_snapshot(), SlurmwatchConfig()) == []

    def test_mem_critical_is_first_and_worst(self) -> None:
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert segs[0][0] == "crit"
        # Facts, not a verdict: the % against the limit, no "OOM RISK" tail.
        assert "MEMORY" in segs[0][1] and "of limit" in segs[0][1]
        assert "RISK" not in segs[0][1]

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

    def test_single_idle_gpu_reads_naturally(self) -> None:
        # One idle GPU should read "GPU IDLE", not the awkward "ALL 1 GPU IDLE".
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(1.0, 0, 0)]
        snap.gpu_count_requested = 1
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert any(lvl == "crit" and txt == "GPU IDLE" for lvl, txt in segs)
        assert not any("ALL 1 GPU" in txt for _lvl, txt in segs)

    def test_idle_gpu_is_not_also_reported_throttling(self) -> None:
        # A GPU can't be both idle and throttling: an idle GPU (this job isn't
        # using it) that still carries a throttle flag — a neighbour's load on a
        # shared card, or a benign clocked-down bit — must be reported idle only.
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(1.0, 0, 0, throttle=True)]  # idle for this job, throttle set
        snap.gpu_count_requested = 1
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert any("IDLE" in txt for _, txt in segs)
        assert not any("THROTTLING" in txt for _, txt in segs)  # not both at once

    def test_active_throttling_gpu_is_reported(self) -> None:
        # The throttling alarm still fires for a GPU the job IS using.
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(95.0, 50 * 1024**3, 55 * 1024**3, throttle=True)]
        snap.gpu_count_requested = 1
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert any("THROTTLING" in txt for _, txt in segs)
        assert not any("IDLE" in txt for _, txt in segs)

    def test_cpu_underuse_is_not_a_banner_alarm(self) -> None:
        # CPU underuse is often intentional (a debug shell, a data-loading stage)
        # and the CPU row already shows its own amber dot, so it must NOT raise a
        # banner headline — that just nagged and duplicated the row.
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=8, usage_ns=0, usage_percent=12.0, effective_cores=1.0
        )
        segs = _banner_segments(snap, SlurmwatchConfig())
        assert not any("CPU" in txt for _, txt in segs)

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

    def test_all_healthy_shows_no_banner(self) -> None:
        # The banner is an alarm-only strip now: a healthy job shows nothing
        # (the RESOURCES panel already tells the story), so it renders empty and
        # the widget is hidden — no redundant "ALL HEALTHY · CPU …" summary.
        b = StatusBanner()
        b.snapshot = _make_snapshot()
        b.config = SlurmwatchConfig()
        out = b.render()
        assert out.strip() == ""
        _valid_markup(out)

    def test_worst_first(self) -> None:
        b = StatusBanner()
        snap = _make_snapshot()
        snap.memory.oom_guard_critical = True
        b.snapshot = snap
        b.config = SlurmwatchConfig()
        out = b.render()
        assert "MEMORY" in out and "of limit" in out and "ALL HEALTHY" not in out
        _valid_markup(out)

    def test_unobservable_gpu_is_not_a_false_alarm(self) -> None:
        # gpus=[] with gpu_count_requested>0 (remote / NVML off): NOT a red/yellow
        # alarm and not a false "0 idle". The "telemetry unavailable" note now
        # lives on the RESOURCES GPU row (more actionable and no longer duplicated),
        # so the banner stays silent here rather than repeating it.
        b = StatusBanner()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_active_count = 0
        snap.gpu_count_requested = 4
        b.snapshot = snap
        b.config = SlurmwatchConfig()
        out = b.render()
        assert "IDLE" not in out  # no false idle alarm
        assert out.strip() == ""  # no banner clutter — the RESOURCES row carries the note
        _valid_markup(out)


class TestLabeledBar:
    """Every bar names what it measures, in a fixed-width label field so bars
    line up in a column across the CPU / MEM / GPU rows."""

    def test_labels_align_and_percent_right_justified(self) -> None:
        from slurmwatch.tui import _labeled_bar

        a = _render_markup(_labeled_bar("compute", 59.0, 10, False, "#9d78d6")).plain
        b = _render_markup(_labeled_bar("vram", 5.0, 10, False, "#9d78d6")).plain
        assert a.startswith("compute ") and b.startswith("vram   ")  # fixed 7-col label
        assert a.rstrip().endswith("59%") and b.rstrip().endswith("5%")

        def bar_start(s: str) -> int:
            return min((i for i, ch in enumerate(s) if ch in "█░"), default=-1)

        assert bar_start(a) == bar_start(b) == 8  # bars align across differing labels


class TestBannerLine:
    """B10: the headline stays one legible line even when many alerts co-occur."""

    # Real alert strings the banner emits (worst first), used to exercise the
    # line formatter's fit/collapse behaviour.
    SEGMENTS = [
        ("crit", "MEMORY 96% — OOM RISK"),
        ("warn", "2 OF 4 GPUS IDLE"),
        ("warn", "1 GPU THROTTLING"),
    ]

    def test_shows_all_when_it_fits(self) -> None:
        from slurmwatch.tui import _banner_line

        line = _render_markup(_banner_line(self.SEGMENTS, False, 200)).plain
        assert "OOM RISK" in line and "THROTTLING" in line

    def test_collapses_to_worst_plus_count_when_too_narrow(self) -> None:
        from slurmwatch.tui import _banner_line

        line = _render_markup(_banner_line(self.SEGMENTS, False, 40)).plain
        assert "OOM RISK" in line  # the single worst alert is kept
        assert "(+2 more)" in line  # the rest are summarized, not wrapped
        assert "THROTTLING" not in line  # nothing wraps mid-phrase


class _SizedRows(ResourceRows):
    """ResourceRows with a fixed width so render() can be unit-tested unmounted
    (an unmounted widget reports width 0, which the code treats as 'wide')."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self._w = width

    @property
    def size(self) -> Size:
        return Size(self._w, 40)


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
        # Every bar names the quantity it measures (no bare, ambiguous %).
        assert "usage" in out and "used" in out
        assert "compute" in out and "vram" in out
        assert "72" in out  # GPU compute utilization
        assert "20 / 40 GiB" in out  # GPU vram amount, clearly labeled
        _valid_markup(out)

    def test_gpu_compute_and_vram_merge_when_wide_stack_when_narrow(self) -> None:
        # One device, two axes: compute (SM util) and vram (fill), each an
        # explicitly-labeled bar. On a wide terminal they ride ONE line (one dense
        # row per GPU); on a narrow one they stack so nothing wraps. A full-memory
        # / moderate-compute GPU reads sensibly either way.
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(59.0, 79 * 1024**3, 79 * 1024**3, memtot=80 * 1024**3)]

        wide = _SizedRows(140)
        wide.snapshot = snap
        wide.config = SlurmwatchConfig()
        gpu_line = next(
            ln for ln in _render_markup(wide.render()).plain.splitlines() if "compute" in ln
        )
        assert "vram" in gpu_line  # compute and vram share one line when wide
        assert "59%" in gpu_line and "99%" in gpu_line
        assert "79 / 80 GiB" in gpu_line and "W" in gpu_line
        _valid_markup(wide.render())

        narrow = _SizedRows(90)
        narrow.snapshot = snap
        narrow.config = SlurmwatchConfig()
        nlines = _render_markup(narrow.render()).plain.splitlines()
        ci = next(i for i, ln in enumerate(nlines) if "compute" in ln)
        vi = next(i for i, ln in enumerate(nlines) if "vram" in ln)
        assert ci < vi  # stacked: the compute bar sits above the vram bar
        assert "59%" in nlines[ci]
        assert "99%" in nlines[vi] and "79 / 80 GiB" in nlines[vi]
        assert "W" in nlines[vi]  # power lives on the vram line when stacked
        _valid_markup(narrow.render())

    def test_no_limit_memory_has_no_contradictory_percent(self) -> None:
        # With no enforced limit, a 'used 0%' bar beside "12 GiB" would contradict
        # itself — show the amount only.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.memory.limit_bytes = 0
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        mem_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "MEM" in ln)
        assert "no limit" in mem_line
        assert "0%" not in mem_line  # no misleading empty percentage bar
        _valid_markup(r.render())

    def test_remote_snapshot_labels_memory_bar_peak(self) -> None:
        # #34: off-node the memory figure is a lifetime peak (sstat MaxRSS), not a
        # live "used". A remote snapshot labels the bar "peak" (matching the text
        # summary) and drops the redundant "· peak N GiB" suffix.
        r = _SizedRows(140)
        snap = _make_snapshot()
        snap.remote = True
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        mem_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "MEM" in ln)
        assert "peak" in mem_line
        assert "used" not in mem_line
        # The "X / Y GiB" figure appears exactly once (no duplicated peak suffix).
        assert mem_line.count("GiB") == 1
        _valid_markup(r.render())

    def test_local_snapshot_labels_memory_bar_used(self) -> None:
        # A live on-node snapshot (remote=False) keeps the "used" label.
        r = _SizedRows(140)
        snap = _make_snapshot()
        assert snap.remote is False
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        mem_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "MEM" in ln)
        assert "used" in mem_line

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

    def test_row_shows_recent_range(self) -> None:
        # The recent min–max (folded in from the old TRENDS panel) rides on the
        # resource's own row, so the current level and how much it moved live in
        # one place instead of a duplicate panel.
        from collections import deque

        r = ResourceRows()
        r.snapshot = _make_snapshot()
        r.config = SlurmwatchConfig()
        r.cpu_history = deque([10.0, 40.0, 20.0, 55.0, 30.0] * 4, maxlen=120)
        cpu_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CPU" in ln)
        assert "10–55%" in cpu_line  # the observed range
        assert "over 60s" in cpu_line  # the window it was measured over

    def test_steady_row_says_steady(self) -> None:
        # A series that barely moved reads as "steady" (no spurious range), and
        # never fabricates a window it can't justify.
        from collections import deque

        r = ResourceRows()
        r.snapshot = _make_snapshot()
        r.config = SlurmwatchConfig()
        r.cpu_history = deque([50.0] * 20, maxlen=120)
        cpu_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CPU" in ln)
        assert "steady" in cpu_line
        assert "–" not in cpu_line  # no min–max dash when steady

    def test_range_tag_dropped_on_narrow_terminal(self) -> None:
        # The range tag is secondary; like the memory peak, it's dropped on a
        # narrow terminal (< _NARROW_COLS) so a row can't wrap past its width.
        from collections import deque

        r = _SizedRows(80)
        r.snapshot = _make_snapshot()
        r.config = SlurmwatchConfig()
        r.cpu_history = deque([10.0, 40.0, 20.0, 55.0] * 4, maxlen=120)
        cpu_line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CPU" in ln)
        assert "over 60s" not in cpu_line and "steady" not in cpu_line


class TestJobInfoBar:
    def _bar(self, time_limit: int | None) -> JobInfoBar:
        b = JobInfoBar()
        b.snapshot = _make_snapshot()
        ctx = JobContext(
            job_id="51459908",
            username="youzhi",
            partition="test",
            nodelist="midway3-0372",
            hostname="midway3-0372",
            cpus_allocated=8,
            mem_limit_bytes=196 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=time.time() - 3600,
            time_limit_seconds=time_limit,
            nodelist_resolved=["midway3-0372"],
        )
        b.job_ctx = ctx
        b.config = SlurmwatchConfig()
        return b

    def test_labels_every_field(self) -> None:
        out = _plain(self._bar(24 * 3600).render())
        assert "job 12345" in out  # from the live snapshot
        assert "user youzhi" in out
        assert "partition test" in out
        assert "node midway3-0372" in out

    def test_compact_drops_the_time_budget_line(self) -> None:
        # On a short terminal (compact) the bar is a single identity line — the
        # secondary time-budget row is dropped so it doesn't starve the body.
        bar = self._bar(24 * 3600)
        bar.compact = True
        out = _plain(bar.render())
        assert "job 12345" in out and "node" in out  # identity kept
        assert "\n" not in bar.render()  # single line
        assert "left of" not in out and "limit" not in out  # time budget dropped

    def test_ascii_mode_never_leaks_a_unicode_separator(self) -> None:
        # --ascii / a non-UTF-8 terminal: the field separator is '-', never the
        # Unicode middle dot, so no stray glyph leaks anywhere in the bottom bar.
        cfg = SlurmwatchConfig()
        cfg.ascii_mode = True
        bar = self._bar(24 * 3600)
        bar.config = cfg
        out = _render_markup(bar.render()).plain
        assert "·" not in out  # no middle dot in ASCII mode
        assert " - " in out  # ...replaced by the ASCII separator

    def test_identity_values_are_coloured(self) -> None:
        # Each field value wears a distinct palette hue (not a flat grey line).
        markup = self._bar(24 * 3600).render()
        assert _ACCENT in markup  # job id
        assert _CPU_COLOR in markup and _GPU_COLOR in markup and _MEM_COLOR in markup

    def test_time_left_colour_signals_urgency(self) -> None:
        # Plenty of time left → green; almost none → red. _make_snapshot has
        # elapsed 3600s, so a 4000s limit leaves ~9% (red), a huge limit ~green.
        ok = self._bar(24 * 3600).render()
        crit = self._bar(4000).render()
        assert _HEALTH_COLOR["ok"] in ok
        assert _HEALTH_COLOR["crit"] in crit

    def test_shows_time_budget_and_end(self) -> None:
        # _make_snapshot() has elapsed 3600s; limit 24h -> 23h left.
        out = _render_markup(self._bar(24 * 3600).render()).plain
        assert "01:00:00" in out  # elapsed
        assert "24:00:00" in out and "limit" in out  # the max the job can run
        assert "23:00:00" in out and "left" in out  # time remaining
        # The end time is the wall-clock deadline (latest the job can run), not a
        # forecast — "ends by", never "ends ~" which read as a prediction.
        assert "ends by" in out
        assert "ends ~" not in out

    def test_no_time_limit_is_stated_plainly(self) -> None:
        out = _render_markup(self._bar(None).render()).plain
        assert "no wall-clock time limit" in out
        assert "left" not in out

    def test_stale_remote_sample_is_flagged(self) -> None:
        # A node streamed from elsewhere (node switcher) has an older timestamp, so
        # the bar says how stale it is; a live local node (fresh timestamp) doesn't.
        # Worded "Ns old", not "sampled" — the switcher no longer says "sampling".
        bar = self._bar(24 * 3600)
        assert bar.snapshot is not None
        bar.snapshot.timestamp = time.time() - 8
        out = _render_markup(bar.render()).plain
        assert "8s old" in out
        assert "sampl" not in out  # the confusing "sampling/sampled" word is gone
        bar.snapshot.timestamp = time.time()
        assert "s old" not in _render_markup(bar.render()).plain  # live -> no note

    def test_over_limit_clamps_without_negatives(self) -> None:
        # elapsed (from _make_snapshot: 3600s) > a 1800s limit: the bar caps at
        # 100% and remaining floors at 0 — never a negative percentage or duration.
        out = _render_markup(self._bar(1800).render()).plain
        assert "100%" in out
        assert "00:00:00" in out and "left" in out
        assert "-00:" not in out  # no negative HH:MM:SS remaining
        assert "-1" not in out.split("·")[1]  # no negative % in the time segment

    def test_multi_node_shows_node_index(self) -> None:
        b = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.node_count = 4
        snap.node_index = 2
        b.snapshot = snap
        out = _plain(b.render())
        assert "node 3 of 4" in out  # 1-based display of node_index 2


class TestKeyFooter:
    def test_each_key_wears_its_resource_color(self) -> None:
        foot = KeyFooter(
            [
                ("q", "Quit", _ACCENT),
                ("c", "CPU", _CPU_COLOR),
                ("m", "Memory", _MEM_COLOR),
                ("g", "GPU", _GPU_COLOR),
            ]
        )
        out = foot.render()
        _valid_markup(out)
        # Distinct colours (not all the coral accent): each key cap uses its hue.
        assert _CPU_COLOR in out and _MEM_COLOR in out and _GPU_COLOR in out
        # The plain text still reads the labels.
        plain = _render_markup(out).plain
        for label in ("Quit", "CPU", "Memory", "GPU"):
            assert label in plain


class TestFmtCores:
    def test_drops_pointless_trailing_zero(self) -> None:
        from slurmwatch.tui import _fmt_cores

        assert _fmt_cores(1.0) == "1"  # not "1.0"
        assert _fmt_cores(16.0) == "16"
        assert _fmt_cores(0.0) == "0"
        assert _fmt_cores(2.8) == "2.8"  # a real fraction keeps its decimal


def _provenance_ctx(**overrides: object) -> JobContext:
    base: dict[str, object] = {
        "job_id": "12345",
        "username": "ada",
        "partition": "gpu",
        "nodelist": "cn001",
        "hostname": "cn001",
        "cpus_allocated": 16,
        "mem_limit_bytes": 64 * 1024**3,
        "gpu_count_requested": 2,
        "gpu_indices": [0, 1],
        "step_id": "0",
        "uid": 1001,
        "account": "rcc-staff",
        "qos": "normal",
        "job_state": "RUNNING",
        "command": "/home/ada/proj/train.py",
        "work_dir": "/home/ada/proj/runs",
        "tres": "cpu=16,mem=64G,gres/gpu=2",
        "submit_time": 1000.0,
        "job_start_time": 1180.0,  # 180s = 3m queue wait
    }
    base.update(overrides)
    return JobContext(**base)  # type: ignore[arg-type]


class TestJobDetailsPanel:
    def _panel(self, ctx: JobContext) -> JobDetailsPanel:
        p = JobDetailsPanel()
        p.job_ctx = ctx
        p.config = SlurmwatchConfig()
        return p

    def test_shows_provenance_not_in_the_rest_of_the_ui(self) -> None:
        out = _plain(self._panel(_provenance_ctx()).render())
        assert "account rcc-staff" in out
        assert "qos normal" in out and "state RUNNING" in out
        assert "command" in out and "/home/ada/proj/train.py" in out
        assert "workdir" in out and "/home/ada/proj/runs" in out
        assert "queue wait 3m" in out  # 180s = 3 minutes

    def test_values_wear_palette_colours(self) -> None:
        # The card should read lively (coloured values), not a flat grey block:
        # account cyan, qos violet, command coral, workdir rose, and RUNNING green.
        markup = self._panel(_provenance_ctx()).render()
        assert _CPU_COLOR in markup  # account
        assert _GPU_COLOR in markup  # qos
        assert _ACCENT in markup  # command headline
        assert _MEM_COLOR in markup  # workdir
        assert _HEALTH_COLOR["ok"] in markup  # state RUNNING -> green

    def test_state_colour_reflects_the_state(self) -> None:
        assert _HEALTH_COLOR["crit"] in self._panel(_provenance_ctx(job_state="FAILED")).render()
        assert _HEALTH_COLOR["warn"] in self._panel(_provenance_ctx(job_state="PENDING")).render()

    def test_does_not_restate_allocation_facts(self) -> None:
        # The rows + bottom bar already carry allocated cores / mem / the request,
        # so this card must not repeat them (the "don't add repetitive info" rule).
        out = _render_markup(self._panel(_provenance_ctx()).render()).plain
        assert "requested" not in out  # no TRES line
        assert "cpu=16" not in out and "mem=64G" not in out
        assert "allocated" not in out and "in use" not in out

    def test_omits_absent_fields(self) -> None:
        ctx = _provenance_ctx(account="", qos="", command="", work_dir="")
        out = _plain(self._panel(ctx).render())
        assert "account" not in out and "command" not in out and "workdir" not in out
        assert "state RUNNING" in out  # what remains still renders

    def test_command_with_bracket_is_escaped(self) -> None:
        # A command containing '[' must not crash Textual's markup parser.
        ctx = _provenance_ctx(command="python train.py --shape [3,224,224]")
        panel = self._panel(ctx)
        _valid_markup(panel.render())  # raises on unbalanced markup
        assert "[3,224,224]" in _render_markup(panel.render()).plain

    def test_long_paths_are_elided_not_wrapped(self) -> None:
        # A deep script path / workdir must not wrap into a cluttered multi-line
        # block: it's shortened to root + …/ + leaf, keeping the file name.
        deep = (
            "/project/rcc/youzhi/.cache/tmp/claude-940740146/"
            "-home-youzhi-slurmwatch/00bba2f4-b926-4976-881e-2a31ff4aeeb8/scratchpad"
        )
        ctx = _provenance_ctx(command=deep + "/sw_multinode.sbatch", work_dir=deep)
        out = _render_markup(self._panel(ctx).render()).plain
        assert "…" in out  # the noisy middle is elided
        assert "sw_multinode.sbatch" in out  # the file name (what you care about) stays
        assert out.rstrip().endswith("/scratchpad") or "/scratchpad" in out  # leaf kept
        assert "00bba2f4" not in out  # the noisy hash middle is gone
        # No content line is long enough to wrap on a normal terminal.
        assert all(len(line) < 90 for line in out.splitlines())

    def test_full_paths_shows_whole_value_hard_wrapped(self) -> None:
        # With full_paths on, the elided middle is gone and the WHOLE path shows,
        # hard-wrapped so it can't overflow (no "…" ellipsis).
        deep = (
            "/project/rcc/youzhi/.cache/tmp/claude-940740146/"
            "-home-youzhi-slurmwatch/00bba2f4-b926-4976-881e-2a31ff4aeeb8/scratchpad/"
            "sw_multinode.sbatch"
        )
        panel = self._panel(_provenance_ctx(command=deep, work_dir=deep))
        panel.full_paths = True
        out = _plain(panel.render())
        assert "…" not in out  # not elided
        assert "claude-940740146" in out  # the middle that elision dropped is back
        assert "sw_multinode.sbatch" in out  # ...and the leaf
        # elided by default (toggle off)
        panel.full_paths = False
        assert "…" in _plain(panel.render())

    def test_expand_hint_only_shows_when_a_path_is_truncated(self) -> None:
        long = (
            "/project/rcc/youzhi/.cache/tmp/claude-940740146/"
            "-home-youzhi-slurmwatch/00bba2f4-b926-4976-881e-2a31ff4aeeb8/scratchpad/run.sbatch"
        )
        # Truncated -> the hint sits by the paths (not the footer).
        out = _plain(self._panel(_provenance_ctx(command=long, work_dir=long)).render())
        assert "…" in out and "for the full path" in out
        # Short paths that fit -> nothing truncated -> no hint (it'd be pointless).
        short = _plain(self._panel(_provenance_ctx(command="/a/b.sh", work_dir="/a")).render())
        assert "…" not in short and "full path" not in short and "press" not in short

    def test_expanded_paths_show_a_collapse_hint(self) -> None:
        panel = self._panel(_provenance_ctx(command="/a/b.sh", work_dir="/a"))
        panel.full_paths = True
        assert "to collapse" in _plain(panel.render())

    def test_command_with_args_is_left_intact(self) -> None:
        # A full command line (has spaces/args) is NOT treated as a path to elide,
        # so its arguments aren't mangled into fake directories.
        ctx = _provenance_ctx(command="python train.py --data /very/long/unused")
        out = _render_markup(self._panel(ctx).render()).plain
        assert "python train.py --data /very/long/unused" in out


class TestPackChips:
    """`_pack_chips`: wrap a labelled strip between chips, never inside one."""

    def test_wraps_between_chips_never_inside(self) -> None:
        chips = ["account a-very-long-account-value", "qos a-long-qos-value", "state RUNNING"]
        out = _pack_chips(chips, " · ", width=36)
        lines = out.split("\n")
        assert len(lines) > 1  # it wrapped
        for chip in chips:  # every chip lands intact on some line (never split)
            assert any(chip in ln for ln in lines)

    def test_single_line_when_it_fits(self) -> None:
        assert _pack_chips(["a 1", "b 2", "c 3"], " · ", width=100) == "a 1 · b 2 · c 3"

    def test_zero_width_falls_back_to_join(self) -> None:
        # Unmounted (size 0): don't crash, just join.
        assert _pack_chips(["a", "b"], " · ", width=0) == "a · b"


class TestShortenPath:
    """`_shorten_path`: fit a long path to a column budget without wrapping."""

    def test_keeps_root_and_leaf_elides_middle(self) -> None:
        p = "/project/rcc/u/.cache/tmp/hash123456/deep/scratchpad/run.sbatch"
        out = _shorten_path(p, budget=40, keep=2)
        assert out.startswith("/project/…/") or out.startswith("/…/")
        assert out.endswith("scratchpad/run.sbatch")  # parent + file kept
        assert "hash123456" not in out
        assert len(out) <= 40

    def test_directory_keeps_only_the_leaf(self) -> None:
        p = "/project/rcc/u/.cache/tmp/hash123456/scratchpad"
        out = _shorten_path(p, budget=30, keep=1)
        assert out.endswith("/scratchpad") and "hash123456" not in out
        assert len(out) <= 30

    def test_short_path_is_unchanged(self) -> None:
        assert _shorten_path("/a/b/c.sh", budget=40) == "/a/b/c.sh"

    def test_home_collapses_to_tilde(self) -> None:
        home = os.path.expanduser("~")
        assert _shorten_path(home + "/work/run.sh", budget=40) == "~/work/run.sh"

    def test_ascii_ellipsis(self) -> None:
        p = "/project/rcc/u/.cache/tmp/hash123456/deep/scratchpad/run.sbatch"
        assert "..." in _shorten_path(p, budget=40, keep=2, ell="...")


class TestCpuUnderuseThreshold:
    """F4: SLURMWATCH_CPU_UNDERUSE drives the CPU row's health dot colour."""

    def test_threshold_is_wired(self) -> None:
        cpu = CpuMetrics(cores_allocated=16, usage_ns=0, usage_percent=30.0, effective_cores=4.8)
        # ratio = 0.3: healthy under the default 0.15, underused under a 0.5 bar.
        assert _cpu_health(cpu, 0.15) == ("ok", "healthy")
        assert _cpu_health(cpu, 0.5) == ("warn", "underused")

    def test_threshold_drives_the_row_dot(self) -> None:
        # Facts-only: the threshold surfaces as the CPU row's health DOT colour,
        # never an "underused" verdict word. Below the bar -> amber warn dot;
        # above -> green ok dot.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=16, usage_ns=0, usage_percent=30.0, effective_cores=4.8
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig(cpu_underuse_threshold=0.5)
        cpu_block = next(b for b in r.render().split("\n\n") if "CPU" in b)
        assert _HEALTH_COLOR["warn"] in cpu_block  # amber dot when under the bar
        assert "underused" not in _render_markup(cpu_block).plain  # never a word
        r.config = SlurmwatchConfig(cpu_underuse_threshold=0.15)
        cpu_block = next(b for b in r.render().split("\n\n") if "CPU" in b)
        assert _HEALTH_COLOR["ok"] in cpu_block  # green dot when above the bar


class TestMarkupValidity:
    """Every panel must emit valid Rich markup in every state Textual renders."""

    def test_all_panels_all_states(self) -> None:
        for warn, crit in [(False, False), (True, False), (True, True)]:
            snap = _make_snapshot()
            snap.memory.oom_guard_warning = warn
            snap.memory.oom_guard_critical = crit
            for cls in (StatusBanner, ResourceRows):
                w = cls()
                w.snapshot = snap
                w.config = SlurmwatchConfig()
                _valid_markup(w.render())

    def test_throttling_shows_as_health_dot_negative_control(self) -> None:
        # B-T9 (facts-only): a throttling GPU is flagged by its amber health dot,
        # not a "throttling" verdict word. Negative control: the amber dot
        # disappears when the GPU isn't throttling. Uses a cool temp so the
        # hot-temp colour can't stand in for the throttle dot.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].throttling = True
        snap.gpus[0].temperature_celsius = 60.0  # cool -> only the dot is amber
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        block = next(b for b in r.render().split("\n\n") if "GPU0" in b)
        assert _HEALTH_COLOR["warn"] in block  # amber dot = throttling
        assert "throttling" not in _render_markup(block).plain  # no verdict word

        snap.gpus[0].throttling = False
        r.snapshot = snap
        block = next(b for b in r.render().split("\n\n") if "GPU0" in b)
        assert _HEALTH_COLOR["warn"] not in block  # active + cool -> no amber

    def test_hot_temp_marker_has_negative_control(self) -> None:
        # The '⚠' hot-temperature marker (matching the GPU table) appears at/above
        # the threshold and disappears below it, so a stray marker can't pass it.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].throttling = False
        snap.gpus[0].temperature_celsius = 88.0  # hot -> '⚠' marker
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        assert "88 °C ⚠" in r.render()

        snap.gpus[0].temperature_celsius = 60.0
        r.snapshot = snap
        cool = r.render()
        assert "⚠" not in cool
        assert "60 °C" in cool

    def test_hot_temp_marker_is_ascii_in_ascii_mode(self) -> None:
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].temperature_celsius = 88.0
        r.snapshot = snap
        cfg = SlurmwatchConfig()
        cfg.ascii_mode = True
        r.config = cfg
        out = r.render()
        assert "88 C !" in out and "⚠" not in out and "·" not in out


# ---------------------------------------------------------------------------
# Integration (Textual Pilot)
# ---------------------------------------------------------------------------


class _StubCollector:
    def __init__(self, raise_once: bool = False) -> None:
        self.config = SlurmwatchConfig()
        self._mock = True
        self._raise_once = raise_once
        self._raised = False
        self.job_ended = False  # mirrors TelemetryCollector.job_ended (#28)

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
    async def test_gpu_table_rows_are_compact_single_height(self) -> None:
        # Efficient spacing: one line per device (no blank-row gap). Adjacent GPUs
        # stay separable via their coloured index cell / per-device hue, and zebra
        # stripes are off so nothing fills the rows with a background band.
        app = _dash_app(_StubCollector(), gpus=4)
        async with app.run_test() as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
            snap.gpu_count_requested = 4
            app.scr._update_widgets(snap)
            await pilot.pause()
            table = app.scr.query_one(GpuTable)
            assert table.zebra_stripes is False
            assert all(row.height == 1 for row in table.rows.values())

    @pytest.mark.asyncio
    async def test_bottom_bar_pinned_to_terminal_floor(self) -> None:
        # A job with little to show must not leave the bottom bar floating mid
        # screen: the job-info + key bar are docked to the terminal floor, with
        # the key bar on the very last row and the job-info bar directly above it.
        app = _dash_app(_StubCollector(), gpus=0)
        async with app.run_test(size=(100, 45)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = []
            snap.gpu_count_requested = 0
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.pause()
            keybar = app.scr.query_one("#keybar")
            jobinfo = app.scr.query_one(JobInfoBar)
            assert keybar.region.y + keybar.region.height == 45  # last row of the terminal
            assert jobinfo.region.y + jobinfo.region.height == keybar.region.y  # directly above

    @pytest.mark.asyncio
    async def test_tall_content_scrolls_body_bar_stays_pinned(self) -> None:
        # When many GPUs overflow a short terminal, the BODY scrolls (not the
        # screen), so the docked bottom bar stays pinned to the floor and visible.
        app = _dash_app(_StubCollector(), gpus=8)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(8)]
            snap.gpu_count_requested = 8
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.pause()
            body = app.scr.query_one("#body")
            assert body.max_scroll_y > 0  # the body scrolls to reveal the rest
            keybar = app.scr.query_one("#keybar")
            assert keybar.region.y + keybar.region.height == 24  # bar still pinned to the floor

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
    async def test_detail_chart_shows_min_avg_max(self) -> None:
        # The drill-in's added value over the dashboard sparkline is the summary
        # stats line; drive a known history and assert the computed figures.
        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, ResourceDetailScreen)
            rows = app.scr.query_one(ResourceRows)
            rows.cpu_history.clear()
            rows.cpu_history.extend([10.0, 50.0, 90.0])
            app.screen._refresh()
            await pilot.pause()
            chart = _render_markup(str(app.screen.query_one("#detail-chart").render())).plain
            assert "min  10%" in chart and "avg  50%" in chart and "max  90%" in chart

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

    @pytest.mark.asyncio
    async def test_jobinfo_bar_mounted_below_body_and_wired(self) -> None:
        # The bottom bar must actually be composed (after #body, before Footer)
        # and fed the live snapshot/ctx by _update_widgets.
        app = _dash_app(_StubCollector())
        async with app.run_test(size=(120, 34)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            bar = app.scr.query_one(JobInfoBar)
            assert bar.snapshot is not None and bar.job_ctx is not None
            out = _plain(str(bar.render()))
            assert "job 12345" in out and "user ada" in out
            # Composed before the keybinding footer (so it sits above it).
            ids = [type(w).__name__ for w in app.scr.walk_children()]
            assert ids.index("JobInfoBar") < ids.index("KeyFooter")

    @pytest.mark.asyncio
    async def test_job_card_mounted_and_fed(self) -> None:
        # The JOB provenance card must be composed inside #body (below RESOURCES)
        # and fed by _update_widgets — and must NOT restate allocation facts.
        app = _dash_app(_StubCollector(), gpus=2)
        app.scr.job_ctx.account = "rcc-staff"
        app.scr.job_ctx.command = "/home/ada/train.py"
        async with app.run_test(size=(128, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 40 * 1024**3, 40 * 1024**3, index=i) for i in range(2)]
            snap.gpu_count_requested = 2
            app.scr._update_widgets(snap)
            await pilot.pause()
            job = app.scr.query_one(JobDetailsPanel)
            out = _render_markup(job.render()).plain
            assert "rcc-staff" in out
            assert "allocated" not in out and "requested" not in out  # no duplication
            # It sits inside the scrolling body.
            body_ids = [type(w).__name__ for w in app.scr.query_one("#body").walk_children()]
            assert "JobDetailsPanel" in body_ids

    @staticmethod
    def _multinode_app(nodes: list[str]) -> _DashApp:
        job = JobContext(
            job_id="12345",
            username="ada",
            partition="gpu",
            nodelist="cn[001-002]",
            hostname=nodes[0],
            cpus_allocated=8,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            step_id="0",
            uid=1001,
            nodelist_resolved=nodes,
        )
        return _DashApp(_StubCollector(), job)

    def test_history_window_sizes_to_remote_cadence(self) -> None:
        # #55: the history deque holds `history_seconds` of the DISPLAYED node's
        # samples. The local node is served at poll_interval (0.5s) -> 120 slots
        # for a 60s window; a remote node is streamed at 1.0s -> 60 slots, so the
        # "over 60s" trend tag is honest (it used to keep 120 remote samples
        # spanning ~120s under a label that claimed 60s).
        app = self._multinode_app(["cn001", "cn002"])
        scr = app.scr
        scr._local_node = "cn001"  # pretend this process runs on cn001
        scr._selected_node = "cn001"
        assert scr._history_maxlen() == 120  # 60s / 0.5s local cadence
        scr._selected_node = "cn002"
        assert scr._history_maxlen() == 60  # 60s / 1.0s remote stream cadence

    @pytest.mark.asyncio
    async def test_node_switcher_number_keys_and_arrows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Number keys jump straight to a node ("press 1-N"); Left/Right also step.
        # Stub the remote stream so switching to a non-local node doesn't srun.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002", "cn003"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            await pilot.press("3")  # jump straight to node 3
            await pilot.pause()
            assert scr._selected_node == "cn003"
            await pilot.press("1")  # and back to node 1
            await pilot.pause()
            assert scr._selected_node == "cn001"
            await pilot.press("right")  # arrows step next/prev too
            await pilot.pause()
            assert scr._selected_node == "cn002"
            await pilot.press("left")
            await pilot.pause()
            assert scr._selected_node == "cn001"
            await pilot.press("9")  # out-of-range digit is ignored, not a crash
            await pilot.pause()
            assert scr._selected_node == "cn001"
            footer = _render_markup(app.scr.query_one("#keybar", KeyFooter).render()).plain
            assert "1-3" in footer and "Node" in footer  # advertises "press 1-3"

    @pytest.mark.asyncio
    async def test_single_node_has_no_switcher(self) -> None:
        app = self._multinode_app(["cn001"])  # one node
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            before = scr._selected_node
            scr.action_next_node()  # no-op with a single node
            await pilot.pause()
            assert scr._selected_node == before
            footer = _render_markup(app.scr.query_one("#keybar", KeyFooter).render()).plain
            assert "Node" not in footer  # not advertised

    @pytest.mark.asyncio
    async def test_scales_to_100_nodes_via_typed_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the viewed node is ever streamed (O(1)), and you TYPE a node number
        # to jump straight there, so a 100-node job reaches any node in a couple of
        # keystrokes — no per-node setup, no arrow-mashing.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        nodes = [f"cn{i:03d}" for i in range(1, 101)]
        app = self._multinode_app(nodes)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            # "55" is unambiguous (no node 550+), so it commits on the 2nd digit.
            await pilot.press("5", "5")
            await pilot.pause()
            assert scr._selected_node == nodes[54]  # node 55
            # "100" reaches the last node.
            await pilot.press("1", "0", "0")
            await pilot.pause()
            assert scr._selected_node == nodes[99]  # node 100
            await pilot.press("left")  # arrows still step to the neighbour
            await pilot.pause()
            assert scr._selected_node == nodes[98]  # node 99
            # An ambiguous prefix ("2" could be node 2 or 20-29) commits on Enter.
            await pilot.press("2", "enter")
            await pilot.pause()
            assert scr._selected_node == nodes[1]  # node 2
            footer = _render_markup(app.scr.query_one("#keybar", KeyFooter).render()).plain
            assert "1-100" in footer  # the full typeable range is advertised

    @pytest.mark.asyncio
    async def test_arrow_cancels_a_pending_typed_jump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Typing an ambiguous prefix then using the arrows must NOT leave a stale
        # pause-timer that later yanks the view to the abandoned number.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app([f"cn{i:03d}" for i in range(1, 21)])  # 20 nodes
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            await pilot.press("1")  # ambiguous (node 1 vs 10-19) -> buffer "1", timer armed
            await pilot.pause()
            assert scr._node_input == "1"
            await pilot.press("right")  # arrow-step -> must clear the buffer + timer
            await pilot.pause()
            assert scr._node_input == ""  # not left dangling
            assert scr._node_input_timer is None
            assert scr._selected_node == "cn002"  # the arrow won, not a late jump to node 1

    @pytest.mark.asyncio
    async def test_typed_node_jump_edge_cases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        # 12-node job: a prefix that overshoots restarts from the latest digit.
        app = self._multinode_app([f"cn{i:03d}" for i in range(1, 13)])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            await pilot.press("1", "5")  # "15" > 12 -> restart to "5" -> node 5
            await pilot.pause()
            assert scr._selected_node == "cn005"
        # 5-node job: a single digit beyond the count is ignored (no crash, no move).
        app2 = self._multinode_app([f"cn{i:03d}" for i in range(1, 6)])
        async with app2.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app2.scr
            await pilot.press("9")  # 9 > 5 -> ignored
            await pilot.pause()
            assert scr._selected_node == "cn001"  # unchanged
            await pilot.press("3")  # valid -> node 3
            await pilot.pause()
            assert scr._selected_node == "cn003"

    @pytest.mark.asyncio
    async def test_short_terminal_compacts_the_bottom_bar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A short terminal collapses the docked bar (single line, no padding/border)
        # so the RESOURCES gauges keep their rows; a tall one keeps the full bar.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        short = self._multinode_app(["cn001", "cn002"])
        async with short.run_test(size=(80, 14)) as pilot:
            await pilot.pause()
            assert short.scr.query_one(JobInfoBar).compact is True
            assert "compact" in short.scr.query_one("#bottombar").classes
        tall = self._multinode_app(["cn001", "cn002"])
        async with tall.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            assert tall.scr.query_one(JobInfoBar).compact is False
            assert "compact" not in tall.scr.query_one("#bottombar").classes

    @pytest.mark.asyncio
    async def test_footer_degrades_gracefully_when_narrow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On a narrow terminal the footer drops the most self-evident labels first
        # (q/c…), keeps every coloured key cap, and retains the least-obvious "Node"
        # label longest — so nothing wraps or clips a label off the right edge.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(48, 20)) as pilot:
            await pilot.pause()
            foot = app.scr.query_one("#keybar", KeyFooter)
            out = _render_markup(foot.render()).plain
            assert "Quit" not in out  # the obvious label goes first
            assert "Node" in out  # the cryptic node cap keeps its word longest
            for cap in ("q", "c", "m", "g", "1-2"):  # every key cap survives
                assert cap in out
            assert _ACCENT in foot.render() and _CPU_COLOR in foot.render()  # colours kept

    @pytest.mark.asyncio
    async def test_switch_shows_banner_and_dims_until_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The mounted wiring: a switch shows the SwitchBanner and dims the body,
        # and the target node's own frame (via _show) clears both.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            await pilot.press("2")
            await pilot.pause()
            banner = scr.query_one(SwitchBanner)
            assert scr._switch_target == "cn002"
            assert banner.display is True
            assert "switching" in scr.query_one("#body").classes  # body dimmed
            frame = _make_snapshot()
            frame.hostname = "cn002"
            frame.node_count, frame.node_index = 2, 1
            scr._show(frame, "cn002")
            await pilot.pause()
            assert scr._switch_target is None  # the node's frame ended the switch
            assert banner.display is False
            assert "switching" not in scr.query_one("#body").classes  # un-dimmed

    @pytest.mark.asyncio
    async def test_switch_slow_note_appears_after_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _tick_switch flips the reassuring "slow" note once the attach passes the
        # slow threshold (but before the stuck threshold), and is a no-op when no
        # switch is pending.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            scr._tick_switch()  # no switch pending -> no crash, no banner
            assert scr.query_one(SwitchBanner).slow is False
            await pilot.press("2")
            await pilot.pause()
            banner = scr.query_one(SwitchBanner)
            assert banner.slow is False  # not yet
            scr._switch_started = time.monotonic() - 6  # past slow (4s), before stuck (12s)
            scr._tick_switch()
            assert banner.slow is True and banner.stuck is False

    @pytest.mark.asyncio
    async def test_switch_shows_the_banner_in_both_directions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every switch confirms the key press with the banner — including back to
        # the local node (which previously showed nothing, reading as "did it
        # work?"). It clears the instant that node's frame lands.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            scr._local_node = "cn001"  # make node 1 the live local node
            scr._selected_node = "cn002"  # pretend we're on node 2
            scr._set_node("cn001")  # ...and switch back to the local node
            await pilot.pause()
            assert scr._switch_target == "cn001"  # banner is up for the local switch too
            assert scr.query_one(SwitchBanner).display is True
            assert "switching" in scr.query_one("#body").classes
            # the local node's own frame clears it
            frame = _make_snapshot()
            frame.hostname = "cn001"
            scr._show(frame, "cn001")
            await pilot.pause()
            assert scr._switch_target is None

    @pytest.mark.asyncio
    async def test_switch_to_unreachable_node_unblocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A node that never streams must NOT freeze the session on a dim, spinning
        # screen: past the stuck threshold the body un-dims and the banner warns,
        # while the switch stays pending (the poll loop keeps retrying).
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            await pilot.press("2")
            await pilot.pause()
            scr._switch_started = time.monotonic() - 30  # pretend it hung
            scr._tick_switch()
            await pilot.pause()
            banner = scr.query_one(SwitchBanner)
            assert banner.stuck is True
            assert banner.display is True  # a warning is still shown
            assert "switching" not in scr.query_one("#body").classes  # no longer dimmed
            assert scr._switch_target == "cn002"  # still trying in the background

    @pytest.mark.asyncio
    async def test_narrow_mem_row_never_overflows(self) -> None:
        # Regression: a big-memory job (3-digit GiB) must not push the MEM row
        # past an 80-col terminal and soft-wrap onto a second line.
        app = _dash_app(_StubCollector())
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.memory = MemoryMetrics(
                current_bytes=520 * 1024**3,
                limit_bytes=512 * 1024**3,
                peak_bytes=500 * 1024**3,
                usage_percent=98.0,
                oom_guard_warning=True,
                oom_guard_critical=True,
                working_set_bytes=502 * 1024**3,
                cache_bytes=0,
            )
            app.scr._update_widgets(snap)
            await pilot.pause()
            rows = app.scr.query_one(ResourceRows)
            width = rows.size.width
            mem_line = next(
                ln for ln in _render_markup(str(rows.render())).plain.splitlines() if "MEM" in ln
            )
            assert len(mem_line) <= width  # fits the content region, no soft-wrap
            assert "peak" not in mem_line  # secondary detail dropped when narrow

    @pytest.mark.asyncio
    async def test_banner_collapses_on_narrow_terminal(self) -> None:
        # B10: the mounted banner must feed its real *width* into _banner_line so
        # concurrent alerts collapse instead of wrapping. The pure-function test
        # can't catch a width/height/0 wiring regression; this does.
        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test(size=(40, 20)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.memory.oom_guard_critical = True
            snap.gpus = [_make_gpu(1.0, 0, 0, index=0), _make_gpu(1.0, 0, 0, index=1)]
            snap.gpu_count_requested = 2
            app.scr._update_widgets(snap)
            await pilot.pause()
            banner = app.scr.query_one(StatusBanner)
            rendered = _render_markup(str(banner.render())).plain
            assert "(+" in rendered and "more)" in rendered  # collapsed, not wrapped
            assert "IDLE" not in rendered  # the lower-priority segment is summarized


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


class TestNodeStreaming:
    """The switcher's remote path: stream a node via srun and cache per node."""

    @staticmethod
    def _screen(nodes: list[str]) -> DashboardScreen:
        job = JobContext(
            job_id="12345",
            username="ada",
            partition="gpu",
            nodelist=",".join(nodes),
            hostname=nodes[0],
            cpus_allocated=8,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=nodes,
        )
        return DashboardScreen(_StubCollector(), job, SlurmwatchConfig())  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_read_remote_streams_and_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scr = self._screen(["cn001", "cn002"])
        snap = _make_snapshot()
        snap.hostname = "cn002"
        snap.node_count = 2
        snap.node_index = 1
        line = (snap.to_json() + "\n").encode()

        class _Out:
            def __init__(self) -> None:
                self.n = 0

            async def readline(self) -> bytes:
                self.n += 1
                if self.n == 1:
                    return line
                await asyncio.sleep(3600)  # then block: no more frames
                return b""

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        proc = _Proc()

        async def _open(*_a: object, **_k: object) -> _Proc:
            return proc

        monkeypatch.setattr("slurmwatch.tui.open_stream", _open)
        got = await scr._read_remote("cn002")
        assert got is not None and got.hostname == "cn002" and got.node_index == 1
        await scr._stop_stream()  # reap the fake proc

    def test_switch_shows_cached_node_instantly(self) -> None:
        # Switching to a node we've seen shows its last snapshot immediately from
        # cache (no wait for the stream), so a re-visit feels instant.
        scr = self._screen(["cn001", "cn002"])
        cached = _make_snapshot()
        cached.hostname = "cn002"
        scr._node_cache["cn002"] = cached
        scr._set_node("cn002")
        assert scr._selected_node == "cn002"
        assert scr.latest_snapshot is cached

    def test_switch_begins_and_a_matching_frame_ends_it(self) -> None:
        # A switch enters the pending state (so the banner can show); the first
        # frame for the *target* node clears it and renders.
        scr = self._screen(["cn001", "cn002"])
        scr._set_node("cn002")
        assert scr._switch_target == "cn002"  # switch is in flight
        frame = _make_snapshot()
        frame.hostname = "cn002"
        scr._show(frame, "cn002")
        assert scr._switch_target is None  # the node's own frame ends the switch
        assert scr.latest_snapshot is frame

    def test_show_drops_a_stale_frame_from_the_old_node(self) -> None:
        # The bug this guards: after switching away, an already-in-flight frame
        # for the *previous* node must NOT overwrite the new node's view (and must
        # not end the pending switch).
        scr = self._screen(["cn001", "cn002"])
        scr._set_node("cn002")  # now waiting on cn002
        stale = _make_snapshot()
        stale.hostname = "cn001"  # a late frame from the node we left
        scr._show(stale, "cn001")
        assert scr.latest_snapshot is not stale  # not rendered
        assert scr._node_cache["cn001"] is stale  # but still cached for a re-visit
        assert scr._switch_target == "cn002"  # switch still pending

    def test_show_renders_by_requested_node_not_self_reported_hostname(self) -> None:
        # A frame is keyed + gated by the node it was REQUESTED for, not by
        # snapshot.hostname — so a cluster where Slurm's NodeName differs from the
        # node's gethostname (aliases / kept domain / case) still renders instead
        # of blanking the dashboard.
        scr = self._screen(["gpu-a100-01", "gpu-a100-02"])
        scr._selected_node = "gpu-a100-01"
        frame = _make_snapshot()
        frame.hostname = "nid001234"  # the node's gethostname != Slurm NodeName
        scr._show(frame, "gpu-a100-01")  # requested for the selected node
        assert scr.latest_snapshot is frame  # rendered despite the hostname mismatch
        assert scr._node_cache["gpu-a100-01"] is frame  # cached under the requested node

    def test_switch_banner_animates_and_names_the_target(self) -> None:
        banner = SwitchBanner()
        assert banner.render() == ""  # idle: nothing shown
        banner.target_label = "node 2 of 2"
        banner.node = "cn002"
        first = _render_markup(banner.render()).plain
        assert "switching to node 2 of 2" in first and "cn002" in first
        assert "sampl" not in first  # no "sampling" jargon
        banner.frame = 1  # the spinner glyph advances between frames
        second = _render_markup(banner.render()).plain
        assert first != second

    def test_switch_banner_slow_note_and_stuck_warning(self) -> None:
        banner = SwitchBanner()
        banner.target_label = "node 2 of 2"
        banner.node = "cn002"
        banner.slow = True
        assert "few seconds" in _render_markup(banner.render()).plain  # reassuring note
        banner.stuck = True  # escalated: an unreachable-looking node
        stuck = _render_markup(banner.render()).plain
        assert "still reaching" in stuck and "retrying" in stuck
        assert _HEALTH_COLOR["warn"] in banner.render()  # amber warning, not violet

    def test_switch_banner_go_to_node_prompt(self) -> None:
        banner = SwitchBanner()
        banner.prompt = "199"
        banner.total = "200"  # own field, so it can't corrupt an in-flight switch's `node`
        out = _render_markup(banner.render()).plain
        assert "go to node" in out and "199" in out and "200" in out  # echoes what's typed
        # a switch (target_label) still shows the switching form when no prompt
        banner.prompt = ""
        banner.target_label = "node 3 of 200"
        assert "switching to node 3 of 200" in _render_markup(banner.render()).plain

    def test_switch_banner_ascii_mode(self) -> None:
        banner = SwitchBanner()
        banner.target_label = "node 2 of 2"
        banner.node = "cn002"
        banner.ascii = True
        out = _render_markup(banner.render()).plain
        assert "->" in out and "…" not in out and "→" not in out  # ASCII arrow/tail
        banner.stuck = True
        assert "!" in _render_markup(banner.render()).plain  # ASCII warning mark, not ⚠


class TestDemoModeSelectsLocalNode:
    """Regression for #27: `--demo` must render the mock collector's frames.

    The dashboard serves `_local_node` from the local collector and streams any
    other node over srun. When the mock nodelist contained no local host, the
    screen selected an unreachable node[0], srun failed, and the dashboard sat on
    "awaiting telemetry…" forever while the collector's frames went nowhere.
    """

    def _screen(self, monkeypatch: pytest.MonkeyPatch) -> DashboardScreen:
        from slurmwatch import slurm

        # A synthetic FQDN (not this machine's real hostname) so the test is
        # obviously host-independent, and the kept ".example.org" suffix also
        # proves node[0] is the *short* local name.
        monkeypatch.setattr("socket.gethostname", lambda: "testnode-01.example.org")
        ctx = slurm._make_mock_job_context("12345")
        collector = _StubCollector()
        return DashboardScreen(collector, ctx, collector.config)  # type: ignore[arg-type]

    def test_selected_node_is_the_local_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scr = self._screen(monkeypatch)
        # Equality here is what routes the poll loop to the fast local collector
        # (`node == self._local_node`) instead of srun-streaming a fake node.
        assert scr._selected_node == scr._local_node
        assert scr._selected_node == "testnode-01"  # short name, domain stripped

    def test_local_node_is_selectable_in_the_switcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `_set_node` refuses any node outside `_node_list`; if the local node
        # weren't listed the user could never switch *back* to live local data.
        scr = self._screen(monkeypatch)
        assert scr._local_node in scr._node_list

    @pytest.mark.asyncio
    async def test_demo_dashboard_renders_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        # A synthetic FQDN (not this machine's real hostname) so the test is
        # obviously host-independent, and the kept ".example.org" suffix also
        # proves node[0] is the *short* local name.
        monkeypatch.setattr("socket.gethostname", lambda: "testnode-01.example.org")
        ctx = slurm._make_mock_job_context("12345")
        collector = _StubCollector()

        class _App(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.scr = DashboardScreen(collector, ctx, collector.config)  # type: ignore[arg-type]

            async def on_mount(self) -> None:
                await self.push_screen(self.scr)

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Feed one frame the way the local-collector branch of the poll loop
            # would, then assert the dashboard left the "awaiting" state.
            app.scr._show(_make_snapshot(), app.scr._local_node)
            await pilot.pause()
            assert app.scr.latest_snapshot is not None
            rows = app.scr.resource_rows
            assert rows is not None
            assert "awaiting telemetry" not in _plain(rows.render())


class TestJobEndedBanner:
    """#28: when the collector reports the job ended, the dashboard shows a final
    persistent banner, freezes the last numbers, and stops polling — but stays
    open so the user can read them and press q."""

    def test_banner_renders_ended_notice(self) -> None:
        b = SwitchBanner()
        b.ended = True
        b.ended_job = "12345"
        out = _plain(b.render())
        assert "JOB 12345 ENDED" in out
        assert "press q to quit" in out

    def test_ended_banner_outranks_switch_and_prompt(self) -> None:
        # The ended notice is terminal: it must win over an in-flight switch
        # spinner and any half-typed "go to node" prompt.
        b = SwitchBanner()
        b.target_label = "node 2 of 4"
        b.node = "cn-002"
        b.prompt = "3"
        b.ended = True
        assert "ENDED" in _plain(b.render())

    def test_ended_banner_ascii_has_no_unicode(self) -> None:
        b = SwitchBanner()
        b.ended = True
        b.ended_job = "9"
        b.ascii = True
        out = b.render()
        assert "⚑" not in out and "JOB 9 ENDED" in _plain(out)

    @pytest.mark.asyncio
    async def test_node_switch_disabled_after_job_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #50: once the job ends the poll loop stops, so a node switch could never
        # be un-dimmed and would only corrupt the frozen final view. Arrows, typed
        # digits, and commit must all be inert, and the frozen screen must stay
        # bright (no "switching" dim) under the terminal JOB ENDED notice.
        from slurmwatch import slurm

        monkeypatch.setattr("socket.gethostname", lambda: "testnode-01.example.org")

        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        ctx = slurm._make_mock_job_context("12345")  # 4 nodes; node 0 is local
        collector = _StubCollector()

        class _App(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.scr = DashboardScreen(collector, ctx, collector.config)  # type: ignore[arg-type]

            async def on_mount(self) -> None:
                await self.push_screen(self.scr)

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            scr._show(_make_snapshot(), scr._local_node)
            await pilot.pause()
            scr._show_job_ended()
            await pilot.pause()
            assert scr._job_ended is True
            before = scr._selected_node
            scr.action_next_node()  # arrow: inert
            await pilot.press("2")  # typed digit: inert
            scr.action_commit_node_input()
            await pilot.pause()
            assert scr._selected_node == before
            assert scr._switch_target is None
            assert scr._node_input == ""
            assert not scr.query_one("#body").has_class("switching")  # stays bright
            assert scr.query_one(SwitchBanner).ended is True  # terminal notice intact

    @pytest.mark.asyncio
    async def test_poll_loop_shows_banner_and_stops_when_job_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch import slurm

        monkeypatch.setattr("socket.gethostname", lambda: "testnode-01.example.org")
        ctx = slurm._make_mock_job_context("12345")
        collector = _StubCollector()

        class _App(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.scr = DashboardScreen(collector, ctx, collector.config)  # type: ignore[arg-type]

            async def on_mount(self) -> None:
                await self.push_screen(self.scr)

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Render a last frame, then signal the job ended.
            app.scr._show(_make_snapshot(), app.scr._local_node)
            await pilot.pause()
            assert app.scr.latest_snapshot is not None
            collector.job_ended = True
            for _ in range(30):
                await asyncio.sleep(0.02)
                if app.scr.query_one(SwitchBanner).ended:
                    break
            await pilot.pause()
            banner = app.scr.query_one(SwitchBanner)
            assert banner.ended is True
            assert banner.display is True
            # Last numbers stay frozen on screen; the app has not exited.
            assert app.scr.latest_snapshot is not None
            rows = app.scr.resource_rows
            assert rows is not None
            assert "awaiting telemetry" not in _plain(rows.render())
            # Poll task has stopped.
            for _ in range(30):
                if app.scr._poll_task is None or app.scr._poll_task.done():
                    break
                await asyncio.sleep(0.02)
            assert app.scr._poll_task is None or app.scr._poll_task.done()
