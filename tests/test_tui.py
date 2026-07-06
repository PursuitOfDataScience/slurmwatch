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
    _ACCENT,
    _CPU_COLOR,
    _FAINT,
    _GPU_COLOR,
    _HEALTH_COLOR,
    _MEM_COLOR,
    DashboardScreen,
    EfficiencyPanel,
    GpuTable,
    HistoryPanel,
    JobInfoBar,
    KeyFooter,
    ResourceDetailScreen,
    ResourceRows,
    StatusBanner,
    _banner_segments,
    _color_bar,
    _cpu_health,
    _format_bytes,
    _format_duration,
    _gpu_health,
    _mem_health,
    _render_sparkline,
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

    def test_cpu_underused_is_plain_language_no_cryptic_ratio(self) -> None:
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=8, usage_ns=0, usage_percent=12.0, effective_cores=1.0
        )
        segs = _banner_segments(snap, SlurmwatchConfig())
        cpu_seg = next(txt for lvl, txt in segs if "CPU" in txt)
        assert cpu_seg == "CPU UNDERUSED"  # no cryptic "1/8" in the headline
        assert "/" not in cpu_seg

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

    SEGMENTS = [
        ("crit", "MEMORY 96% — OOM RISK"),
        ("warn", "2 OF 4 GPUS IDLE"),
        ("warn", "1 GPU THROTTLING"),
        ("warn", "CPU UNDERUSED"),
    ]

    def test_shows_all_when_it_fits(self) -> None:
        from slurmwatch.tui import _banner_line

        line = _render_markup(_banner_line(self.SEGMENTS, False, 200)).plain
        assert "OOM RISK" in line and "THROTTLING" in line

    def test_collapses_to_worst_plus_count_when_too_narrow(self) -> None:
        from slurmwatch.tui import _banner_line

        line = _render_markup(_banner_line(self.SEGMENTS, False, 40)).plain
        assert "OOM RISK" in line  # the single worst alert is kept
        assert "(+3 more)" in line  # the rest are summarized, not wrapped
        assert "THROTTLING" not in line  # nothing wraps mid-phrase


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

    def test_gpu_row_shows_compute_and_vram_separately(self) -> None:
        # The reported confusion: an unlabeled bar next to "VRAM 79/80G" read as a
        # contradiction. Now compute (SM util) and vram (fill) are two explicitly
        # labeled bars on distinct lines, so a full-memory / moderate-compute GPU
        # reads sensibly.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(59.0, 79 * 1024**3, 79 * 1024**3, memtot=80 * 1024**3)]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        _valid_markup(r.render())
        lines = _render_markup(r.render()).plain.splitlines()
        ci = next(i for i, ln in enumerate(lines) if "compute" in ln)
        vi = next(i for i, ln in enumerate(lines) if "vram" in ln)
        assert ci < vi  # the compute bar sits above the vram bar
        assert "59%" in lines[ci]
        assert "99%" in lines[vi] and "79 / 80 GiB" in lines[vi]  # 79/80 fill
        assert "W" in lines[vi]  # power lives on the vram line

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

    def test_flags_real_problems_only(self) -> None:
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
        assert "Recommendations" in out
        assert "--mem" in out  # the actionable memory advice
        assert "--gres=gpu:1" in out  # the actionable GPU sentence
        assert "GPU 1 idle" in out  # names the specific idle device, not "1 of 2"
        _valid_markup(out)

    def test_no_grade_word_and_no_cgroup_jargon(self) -> None:
        # The reframe: never a context-free "good", and the on-node data source
        # is not spelled out as confusing "cgroup v1/v2" plumbing.
        e = EfficiencyPanel()
        snap = _make_snapshot()  # a healthy snapshot: no problems to flag
        e.snapshot = snap
        e.config = SlurmwatchConfig()
        e.source = "cgroup v2"
        out = e.render()
        assert "good" not in out.lower()  # no value judgment
        assert "cgroup" not in out.lower()  # no plumbing jargon
        assert "nothing to change" in out  # neutral all-clear
        _valid_markup(out)

    def test_remote_source_is_flagged_in_plain_language(self) -> None:
        e = EfficiencyPanel()
        e.snapshot = _make_snapshot()
        e.config = SlurmwatchConfig()
        e.source = "sstat (remote)"
        out = e.render()
        assert "remote estimate" in out and "compute node" in out
        assert "cgroup" not in out.lower()
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

        # Varying histories so the auto-scaled sparklines actually draw (a
        # constant series renders as a "steady" rule, tested separately).
        panel = _SizedHistoryPanel(width, height)
        panel.snapshot = _make_snapshot()
        panel.config = SlurmwatchConfig()
        wave = [50.0 + 20.0 * (i % 5) / 4 for i in range(40)]  # 50→70 sawtooth
        panel.cpu_history = deque(wave, maxlen=120)
        panel.mem_history = deque(wave, maxlen=120)
        panel.gpu_history = {0: deque(wave, maxlen=120)}
        return panel

    def test_renders_a_sparkline_per_series(self) -> None:
        out = self._panel(100, 8).render()
        # A titled panel with one labeled sparkline per resource, in its own color.
        assert "TRENDS" in out
        assert "CPU" in out and "MEM" in out and "GPU0" in out
        assert _CPU_COLOR in out and _MEM_COLOR in out  # each series in its block hue
        # The body is a one-row block sparkline (▁..█), not braille or a block wall.
        assert any(ch in "▁▂▃▄▅▆▇█" for ch in out)
        assert not any(0x2800 <= ord(ch) <= 0x28FF for ch in out)  # no braille
        _valid_markup(out)

    def test_compact_height_independent_of_panel(self) -> None:
        # height:auto — the panel emits the same compact block (title + a blank +
        # one line per series with a blank between) regardless of how tall it is,
        # rather than stretching to fill a 1fr box.
        short = self._panel(100, 8).render().count("\n")
        tall = self._panel(100, 40).render().count("\n")
        assert short == tall
        # 3 series -> title, blank, s, blank, s, blank, s = 7 lines (6 newlines).
        assert short == 6

    def test_labels_say_what_the_percent_means(self) -> None:
        # A bare "62%" is ambiguous; each row names its metric (busy/used/compute)
        # and the title explains the axis.
        out = self._panel(100, 12).render()
        assert "CPU busy" in out and "MEM used" in out and "GPU0 compute" in out
        assert "TRENDS" in out and "last" in out

    def test_autoscale_shows_range_when_moving(self) -> None:
        # A varying series is auto-scaled and annotated with its min–max range.
        from collections import deque

        panel = self._panel(100, 12)
        panel.cpu_history = deque([10.0, 40.0, 20.0, 55.0, 30.0] * 4, maxlen=120)
        out = _render_markup(panel.render()).plain
        assert "10–55%" in out  # the observed range, not a bare current %

    def test_steady_band_sits_at_absolute_height(self) -> None:
        # Steady lines draw at their ABSOLUTE height, so different values sit at
        # different heights (a steady 99% is a full band, a steady 4% a low one) —
        # not all collapsed to the same mid-height bar.
        from collections import deque

        panel = self._panel(100, 12)
        # The band height follows the *current* value (from the snapshot); make
        # CPU read 99% and MEM ~4%, with constant histories so both are "steady".
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=8, usage_ns=0, usage_percent=99.0, effective_cores=7.9
        )
        snap.memory = MemoryMetrics(
            current_bytes=4 * 1024**3,
            limit_bytes=100 * 1024**3,
            peak_bytes=4 * 1024**3,
            usage_percent=4.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=4 * 1024**3,
            cache_bytes=0,
        )
        panel.snapshot = snap
        panel.cpu_history = deque([99.0] * 40, maxlen=120)
        panel.mem_history = deque([4.0] * 40, maxlen=120)
        panel.gpu_history = {}
        lines = _render_markup(panel.render()).plain.splitlines()
        cpu_line = next(ln for ln in lines if "CPU busy" in ln)
        mem_line = next(ln for ln in lines if "MEM used" in ln)
        assert "steady" in cpu_line and "steady" in mem_line

        def peak(line: str) -> int:
            return max(("▁▂▃▄▅▆▇█".index(c) for c in line if c in "▁▂▃▄▅▆▇█"), default=0)

        assert peak(cpu_line) > peak(mem_line)  # 99% band clearly taller than 4%

    def test_steady_band_ripples_and_travels(self) -> None:
        # A steady band still *moves*: it has more than one glyph height (a ripple,
        # not a dead-flat row), and advancing the frame counter shifts it.
        from collections import deque

        panel = self._panel(100, 12)
        panel.cpu_history = deque([50.0] * 40, maxlen=120)
        panel.mem_history = deque([50.0] * 40, maxlen=120)
        panel.gpu_history = {}

        def cpu_band(frame: int) -> str:
            panel.frame = frame
            line = next(
                ln for ln in _render_markup(panel.render()).plain.splitlines() if "CPU busy" in ln
            )
            return line

        b0 = cpu_band(0)
        assert len({c for c in b0 if c in "▁▂▃▄▅▆▇█"}) >= 2  # rippled, not flat
        assert cpu_band(3) != b0  # travels with the frame → looks alive over time

    def test_sparklines_are_a_tight_group(self) -> None:
        # The series form a compact group (one blank line between), not scattered
        # across the panel with big gaps.
        panel = self._panel(100, 20)
        body = _render_markup(panel.render()).plain.splitlines()[1:]  # drop title
        spark_rows = [i for i, ln in enumerate(body) if any(c in "▁▂▃▄▅▆▇█" for c in ln)]
        assert len(spark_rows) == 3
        gaps = [b - a for a, b in zip(spark_rows, spark_rows[1:], strict=False)]
        assert all(g == 2 for g in gaps)  # exactly one blank line between each


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
        out = _render_markup(self._bar(24 * 3600).render()).plain
        assert "job 12345" in out  # from the live snapshot
        assert "user youzhi" in out
        assert "partition test" in out
        assert "node midway3-0372" in out

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
        assert "ends ~" in out

    def test_no_time_limit_is_stated_plainly(self) -> None:
        out = _render_markup(self._bar(None).render()).plain
        assert "no wall-clock time limit" in out
        assert "left" not in out

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
        out = _render_markup(b.render()).plain
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
        # ratio = 0.3: flagged (recommend fewer cores) under a 0.5 bar; not flagged
        # under the default 0.15 bar — and never graded "good" either way.
        e.config = SlurmwatchConfig(cpu_underuse_threshold=0.5)
        flagged = _render_markup(e.render()).plain
        assert "CPU barely used" in flagged and "--cpus-per-task" in flagged
        e.config = SlurmwatchConfig(cpu_underuse_threshold=0.15)
        clean = _render_markup(e.render()).plain
        assert "--cpus-per-task" not in clean  # CPU no longer flagged
        assert "good" not in clean.lower()  # and never graded "good"


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
        assert "88 °C!" in hot

        snap.gpus[0].throttling = False
        snap.gpus[0].temperature_celsius = 60.0
        r.snapshot = snap
        cool = r.render()
        assert "throttling" not in cool
        assert "!" not in cool
        assert "60 °C" in cool


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
    async def test_gpu_table_rows_have_a_separating_gap(self) -> None:
        # Adjacent GPUs (often identical when a job saturates every device) must
        # be visually separable: each row is height 2 (a one-line gap) and zebra
        # stripes are off so the gap isn't filled with a background band.
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
            assert all(row.height == 2 for row in table.rows.values())

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
            out = _render_markup(str(bar.render())).plain
            assert "job 12345" in out and "user ada" in out
            # Composed before the keybinding footer (so it sits above it).
            ids = [type(w).__name__ for w in app.scr.walk_children()]
            assert ids.index("JobInfoBar") < ids.index("KeyFooter")

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
