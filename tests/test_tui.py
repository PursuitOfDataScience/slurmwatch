from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from typing import Any

import pytest
from rich.markup import render as _render_markup
from textual.app import App
from textual.css.query import NoMatches
from textual.geometry import Size

from slurmwatch import tui as tuimod
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import (
    CpuMetrics,
    GpuInterconnect,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    NodeFabric,
    TelemetrySnapshot,
)
from slurmwatch.tui import (
    _ACCENT,
    _CPU_COLOR,
    _FAINT,
    _GPU_COLOR,
    _HEALTH_COLOR,
    _MEM_COLOR,
    DashboardScreen,
    ForeignJobApp,
    ForeignJobScreen,
    ForeignJobView,
    JobDetailsPanel,
    JobInfoBar,
    KeyFooter,
    LogViewScreen,
    MonitorNote,
    ResourceDetailScreen,
    ResourceRows,
    SwitchBanner,
    _area_chart,
    _bar_cells,
    _color_bar,
    _cpu_health,
    _format_bytes,
    _format_duration,
    _gpu_health,
    _gpu_model,
    _interconnect_block,
    _interconnect_label,
    _interconnect_summary,
    _interconnect_traffic_glance,
    _mem_health,
    _pack_chips,
    _shorten_path,
    _topo_legend,
    _topo_matrix_lines,
    _topo_traffic_lines,
)


def _valid_markup(text: str) -> None:
    """Rich must be able to parse the string; Textual parses it every render."""
    _render_markup(text)  # raises MarkupError on unbalanced/invalid markup


def _plain(markup: str) -> str:
    """Rendered plain text with non-breaking spaces normalised to spaces, so
    'label value' assertions don't care that the UI binds each label to its value
    with a NBSP (which keeps a chip from wrapping apart across a line break)."""
    return _render_markup(markup).plain.replace("\N{NO-BREAK SPACE}", " ")


def _textual_plain(markup: str) -> str:
    """Plain text as TEXTUAL's markup parser sees it — the engine that runs every
    real render, and the one whose behaviour differs from Rich's on hostile text
    (it raises MarkupError on an unbalanced ``[/]`` and silently CONSUMES a valid
    tag like ``[red]``, shrinking the visible width). Use this, not ``_plain``,
    whenever a test needs to prove externally-supplied text was escaped."""
    from textual.markup import to_content

    return to_content(markup).plain.replace("\N{NO-BREAK SPACE}", " ")


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


class TestAreaChart:
    def test_shape_and_fill_levels(self) -> None:
        from collections import deque

        # A constant series at the top/bottom/middle of the 0-100 scale.
        full = _area_chart(deque([100.0] * 10), width=6, height=4)
        assert len(full) == 4 and all(len(r) == 6 for r in full)
        assert full[0] == "█" * 6 and full[-1] == "█" * 6  # 100% fills every row

        empty = _area_chart(deque([0.0] * 10), width=6, height=4)
        assert all(row == " " * 6 for row in empty)  # 0% draws nothing

        mid = _area_chart(deque([50.0] * 10), width=6, height=4)
        assert mid[0] == " " * 6 and mid[-1] == "█" * 6  # bottom half filled

    def test_empty_history_is_blank(self) -> None:
        from collections import deque

        rows = _area_chart(deque(), width=8, height=3)
        assert rows == [" " * 8] * 3

    def test_ascii_mode_has_no_unicode(self) -> None:
        from collections import deque

        rows = _area_chart(deque([70.0] * 4), width=5, height=4, ascii_mode=True)
        assert all(c.isascii() for row in rows for c in row)


class TestHelpers:
    def test_format_bytes(self) -> None:
        assert _format_bytes(0) == "0 B"  # a byte count carries no fraction
        assert _format_bytes(1024) == "1.0 KiB"
        assert _format_bytes(1024**3) == "1.0 GiB"
        assert _format_bytes(1024**5) == "1.0 PiB"
        # A5: a value just under a power of 1024 promotes instead of "1024.0 X".
        assert _format_bytes(1073741800) == "1.0 GiB"  # ~1 GiB, was "1024.0 MiB"
        assert _format_bytes(1099460000000) == "1.0 TiB"  # was "1024.0 GiB"

    def test_format_duration(self) -> None:
        assert _format_duration(0) == "00:00:00"
        assert _format_duration(3661) == "01:01:01"
        assert _format_duration(86399) == "23:59:59"
        assert _format_duration(-5) == "00:00:00"  # clock-skew clamp, no odd negatives

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

    def test_bar_full_only_at_rounded_100(self) -> None:
        # Note 2: a completely full bar must agree with a "100%" label — a value that
        # rounds to 99% leaves the last cell unfilled even on a narrow gauge, so the
        # bar and the label beside it never disagree.
        assert _color_bar(99.4, 10).count("█") < 10  # not solid-full at 99%
        assert _color_bar(100.0, 10).count("█") == 10  # full only at a rounded 100%
        assert _bar_cells(99.4, 10) == 9
        assert _bar_cells(100.0, 10) == 10

    def test_bar_at_nonpositive_width_draws_nothing(self) -> None:
        # A width of 0 (or negative) has no slot to draw into, so the "keep at least
        # one cell so a non-zero % isn't drawn empty" rule must not fire — it would
        # emit a 1-cell bar into a 0-cell column and push the row's later fields out
        # of alignment by one.
        for width in (0, -1, -8):
            assert _bar_cells(50.0, width) == 0
            assert _bar_cells(100.0, width) == 0
            assert _render_markup(_color_bar(50.0, width, color=_CPU_COLOR)).plain == ""
            assert _render_markup(_color_bar(50.0, width, ascii_mode=True)).plain == ""

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
        # least a sliver of fill (a partial cap is fine) so the bar matches its number.
        assert _bar_cells(4.0, 18) >= 1
        assert _fill_eighths(_render_markup(_color_bar(4.0, 18, color=_MEM_COLOR)).plain) >= 1
        # A sub-0.5% value that rounds to "0%" still draws empty, matching its label.
        assert _bar_cells(0.3, 18) == 0
        assert _fill_eighths(_render_markup(_color_bar(0.3, 18, color=_MEM_COLOR)).plain) == 0
        # A NaN percent must not crash round() (min/max don't neutralize NaN); inf
        # still clamps to a full bar.
        assert _bar_cells(float("nan"), 18) == 0
        assert _bar_cells(float("inf"), 18) == 18

    def test_bar_is_eighth_accurate_and_never_falsely_empty(self) -> None:
        # The fill lands on its true length to one-eighth of a cell (round, not
        # floor), so the on-screen bar agrees with the percent printed beside it at
        # any width. A value that DISPLAYS as >= 1% keeps at least a one-eighth
        # sliver; a genuine 0% is empty. Fill + faint track always span the width.
        for width in (18, 30, 74):
            for pct in (0.0, 0.3, 1.0, 4.0, 12.0, 50.0, 70.0, 99.0, 100.0):
                plain = _render_markup(_color_bar(pct, width, color=_CPU_COLOR)).plain
                assert len(plain) == width
                eighths = _fill_eighths(plain)
                if round(pct) < 1:
                    assert eighths == 0  # a genuine 0% is empty
                else:
                    assert eighths >= 1  # a displayed >=1% is never an empty bar
                    expected = min(width * 8, round(pct / 100 * width * 8))
                    assert eighths == max(1, expected)  # lands on its true length
                # ASCII has no partial glyphs: it rounds to whole cells (_bar_cells).
                ascii_fill = _render_markup(
                    _color_bar(pct, width, ascii_mode=True, color=_CPU_COLOR)
                ).plain.count("#")
                assert ascii_fill == _bar_cells(pct, width)

    def test_color_bar_draws_a_partial_end_cap(self) -> None:
        # 70% of an 18-cell bar is 12.6 cells: a whole-cell bar would snap to 13,
        # but the fractional bar draws 12 whole cells + a partial cap (▋ = 5/8) so
        # its length matches the printed "70%" instead of over-reporting.
        plain = _render_markup(_color_bar(70.0, 18, color=_CPU_COLOR)).plain
        assert plain.count("█") == 12  # 12 whole cells
        assert plain[12] in _FILL_GLYPHS[1:]  # a partial cap follows, not a whole cell
        assert _fill_eighths(plain) == round(70 / 100 * 18 * 8)  # exact to the eighth
        # An exact multiple of a cell has NO partial cap (25% of 8 = exactly 2).
        exact = _render_markup(_color_bar(25.0, 8, color=_CPU_COLOR)).plain
        assert exact == "██░░░░░░"


# Every glyph a bar can draw as FILL: a whole cell (█) plus the left-partial caps
# (▏…▉) a fractional bar uses for sub-cell precision. Index in this string doubles
# as the eighth value of a partial (▏ = index 1 = 1/8 … ▉ = index 7 = 7/8).
_FILL_GLYPHS = "█▏▎▍▌▋▊▉"


# `_has_bar` and `_fill_cells` used to sit here and nothing called either. The
# second was superseded by `_fill_eighths` below, which counts the same fill at
# eighth-of-a-cell precision and so subsumes a whole-cell count; the first was
# simply orphaned. `_FILL_GLYPHS` above is still needed -- `_fill_eighths` indexes
# it to turn a partial cap into its eighth value.
def _fill_eighths(plain: str) -> int:
    # A bar's total fill in eighths of a cell: a whole "█" is 8, a partial cap
    # (▏…▉) contributes its 1..7 eighths. Lets a test assert a fractional bar's
    # true on-screen length rather than only whole cells.
    total = 0
    for ch in plain:
        if ch == "█":
            total += 8
        elif ch in _FILL_GLYPHS[1:]:
            total += _FILL_GLYPHS.index(ch)
    return total


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
        # Throttling is NOT a status word (jargon, reads as alarming) — a throttling
        # but still-running device reads as plain "active"; power/temp give context.
        throttling = _make_gpu(util=94.0, procmem=50 * 1024**3, memused=55 * 1024**3, throttle=True)
        assert _gpu_health(throttling, 5.0) == ("ok", "active")


class TestGpuModel:
    """The short device model shown beside "CUDA N" (H100 / A100 / …)."""

    def test_strips_vendor_and_memory_size(self) -> None:
        # NVML's product name carries a vendor prefix and often the VRAM size; the
        # size is already on the block's second line, so it's pure wasted width.
        assert _gpu_model("NVIDIA GH200 480GB") == "GH200"
        assert _gpu_model("NVIDIA GeForce RTX 4090") == "RTX 4090"
        assert _gpu_model("Quadro RTX 6000") == "RTX 6000"

    def test_strips_the_bus_and_form_factor(self) -> None:
        # The label answers exactly one question — WHICH GPU is this — so the bus /
        # form factor / memory technology goes: the interconnect label already names
        # the bus (beside its live rate), making a per-device "PCIe" a second copy of
        # the same fact, and the power cap + VRAM total on these two lines already
        # separate one model's classes.
        assert _gpu_model("NVIDIA H100 PCIe") == "H100"
        assert _gpu_model("NVIDIA A100-PCIE-40GB") == "A100"
        assert _gpu_model("Tesla P100-PCIE-16GB") == "P100"
        assert _gpu_model("NVIDIA A100-SXM4-80GB") == "A100"
        assert _gpu_model("Tesla V100-SXM2-16GB") == "V100"
        assert _gpu_model("NVIDIA H100 NVL") == "H100"
        assert _gpu_model("NVIDIA H100 80GB HBM3") == "H100"
        assert _gpu_model("NVIDIA H200 141GB HBM3e") == "H200"

    def test_keeps_model_tokens_that_only_look_like_a_form_factor(self) -> None:
        # The drop list is exact-token, so a model whose name merely CONTAINS one of
        # those strings survives intact.
        assert _gpu_model("NVIDIA SXM9000") == "SXM9000"
        assert _gpu_model("NVIDIA NVL40") == "NVL40"

    def test_bare_models_pass_through(self) -> None:
        assert _gpu_model("NVIDIA H200") == "H200"
        assert _gpu_model("NVIDIA L40S") == "L40S"
        assert _gpu_model("NVIDIA A40") == "A40"
        assert _gpu_model("NVIDIA RTX A6000") == "RTX A6000"

    def test_empty_when_nothing_left(self) -> None:
        # A vendor-only or blank name must yield "", so the caller omits the label
        # instead of rendering a stray separator with nothing after it.
        assert _gpu_model("") == ""
        assert _gpu_model("NVIDIA") == ""
        assert _gpu_model("   ") == ""

    def test_truncates_with_mode_appropriate_marker(self) -> None:
        from slurmwatch.tui import _GPU_MODEL_MAX

        long = _gpu_model("NVIDIA SomeVeryLongExperimentalAccelerator 9000 PCIe")
        assert len(long) <= _GPU_MODEL_MAX
        assert long.endswith("…")
        ascii_long = _gpu_model("NVIDIA SomeVeryLongExperimentalAccelerator 9000 PCIe", True)
        assert len(ascii_long) <= _GPU_MODEL_MAX
        assert ascii_long.endswith("...")
        assert all(ord(c) < 128 for c in ascii_long)  # --ascii purity

    def test_real_names_fit_without_truncation(self) -> None:
        # Every datacentre GPU we expect to meet must render in full — truncation is
        # a defensive path, not the normal case.
        from slurmwatch.tui import _GPU_MODEL_MAX

        for name in (
            "NVIDIA H100 PCIe",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H200",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100-PCIE-40GB",
            "Tesla V100-SXM2-16GB",
            "NVIDIA L40S",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA GH200 480GB",
        ):
            model = _gpu_model(name)
            assert model and len(model) <= _GPU_MODEL_MAX
            assert "…" not in model


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class TestMonitorNote:
    def test_calls_out_the_stall(self) -> None:
        # The note is contextual: it only ever renders the "a launch looks stuck"
        # warning (shown by the screen only while a launch is actually stuck). There
        # is no always-on "monitoring via a job step" note any more.
        out = MonitorNote().render()
        assert "stuck" in out and "quit" in out
        _valid_markup(out)

    def test_ascii_mode_is_pure_ascii(self) -> None:
        n = MonitorNote()
        n.ascii_mode = True
        out = n.render()
        out.encode("ascii")  # raises UnicodeEncodeError if a glyph leaked through
        _valid_markup(out)


class TestSwitchBanner:
    def test_escapes_bracket_in_node_and_job(self) -> None:
        # SwitchBanner interpolates node/job values; a stray '[' must be escaped, not
        # crash the markup parser — it was the one widget that skipped _escape_markup.
        b = SwitchBanner()
        b.ended = True
        b.ended_job = "12345_[3]"
        _valid_markup(b.render())  # ended notice
        b.ended = False
        b.target_label = "node [x]"
        b.node = "cn[01]"
        b.stuck = True
        _valid_markup(b.render())  # stuck/amber variant
        b.stuck = False
        _valid_markup(b.render())  # animated switching variant


class TestLabeledBar:
    """Every bar names what it measures, in a fixed-width label field so bars
    line up in a column across the CPU / MEM / GPU rows."""

    def test_labels_align_and_percent_right_justified(self) -> None:
        from slurmwatch.tui import _labeled_bar

        a = _render_markup(_labeled_bar("compute", 59.0, 10, False, "#9d78d6")).plain
        b = _render_markup(_labeled_bar("VRAM", 5.0, 10, False, "#9d78d6")).plain
        assert a.startswith("compute ") and b.startswith("VRAM   ")  # fixed 7-col label
        assert a.rstrip().endswith("59%") and b.rstrip().endswith("5%")

        def bar_start(s: str) -> int:
            return min((i for i, ch in enumerate(s) if ch in _FILL_GLYPHS + "░"), default=-1)

        assert bar_start(a) == bar_start(b) == 8  # bars align across differing labels


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
        assert "CPU" in out and "MEM" in out
        assert "GPU" in out and "CUDA 0" in out  # GPU section head + device-0 block
        assert "16 cores" in out
        # Every bar names the quantity it measures (no bare, ambiguous %).
        assert out.count("used") >= 2  # CPU and MEM bars both labelled "used"
        assert "compute" in out and "VRAM" in out
        assert "72" in out  # GPU compute utilization
        assert "20 / 40 GiB" in out  # GPU vram amount, clearly labeled
        _valid_markup(out)

    def test_gpu_compute_and_vram_always_stack(self) -> None:
        # One device, two axes: compute (SM util) and vram (fill), each an
        # explicitly-labeled bar with its own %. They ALWAYS stack — compute bar
        # directly above the vram bar — at both wide and narrow widths, so 'how
        # busy' and 'how full' read as two comparable gauges (the vertical room is
        # spent on clarity rather than packing both onto one dense line).
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(59.0, 79 * 1024**3, 79 * 1024**3, memtot=80 * 1024**3)]
        for width in (140, 90):
            r = _SizedRows(width)
            r.snapshot = snap
            r.config = SlurmwatchConfig()
            lines = _render_markup(r.render()).plain.splitlines()
            ci = next(i for i, ln in enumerate(lines) if "compute" in ln)
            vi = next(i for i, ln in enumerate(lines) if "VRAM" in ln)
            assert ci + 1 == vi  # the vram bar sits directly below the compute bar
            assert "59%" in lines[ci]
            assert "99%" in lines[vi] and "79 / 80 GiB" in lines[vi]
            assert "W" in lines[ci]  # power/temp trails the compute (top) line
            _valid_markup(r.render())

    def test_gpu_amount_columns_align_across_devices(self) -> None:
        # Same-unit facts stack into a column across devices: the shorter values
        # are right-justified so the "/" (VRAM), the "W" (power) and the "°C"
        # (temp) line up down the GPU section instead of hanging at ragged offsets.
        snap = _make_snapshot()
        snap.gpus = [
            _make_gpu(96.0, 50 * 1024**3, 57 * 1024**3, memtot=80 * 1024**3, index=0),
            _make_gpu(4.0, 0, 2 * 1024**3, memtot=80 * 1024**3, index=1),
        ]
        snap.gpus[0].power_watts, snap.gpus[0].temperature_celsius = 356.0, 71.0
        snap.gpus[1].power_watts, snap.gpus[1].temperature_celsius = 58.0, 36.0
        r = _SizedRows(140)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()

        vram_lines = [ln for ln in lines if "VRAM" in ln]  # MEM's "GiB /" line excluded
        assert len(vram_lines) == 2
        assert len({ln.index("/") for ln in vram_lines}) == 1  # slashes stack
        assert "57 / 80 GiB" in vram_lines[0]
        assert " 2 / 80 GiB" in vram_lines[1]  # shorter used space-padded, not shifted

        compute_lines = [ln for ln in lines if "°C" in ln]  # row 1 carries power + temp
        assert len(compute_lines) == 2
        assert len({ln.index("W") for ln in compute_lines}) == 1  # watts unit aligned
        assert len({ln.index("°C") for ln in compute_lines}) == 1  # temp unit aligned
        assert " 58 W" in compute_lines[1]  # 2-digit watts right-justified to width 3

    def test_cpu_mem_used_columns_align(self) -> None:
        # The CPU and MEM "used" figures share a right-justified width so the two
        # rows' "/" stack ("8" -> " 8" to line up under MEM's two-digit "28").
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 0  # keep GiB out of the GPU section
        snap.cpu = CpuMetrics(
            cores_allocated=16, usage_ns=0, usage_percent=50.0, effective_cores=8.0
        )  # MEM default: working set 28 GiB / limit 64 GiB
        r = _SizedRows(140)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        cpu_line = next(ln for ln in lines if "cores" in ln)
        mem_line = next(ln for ln in lines if "GiB" in ln)
        assert cpu_line.index("/") == mem_line.index("/")  # slashes stack into a column
        assert " 8 / 16 cores" in cpu_line  # CPU "8" padded to MEM's width
        assert "28 / 64 GiB" in mem_line

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

    def test_multi_gpu_renders_inline_device_blocks(self) -> None:
        # 3+ GPUs render inline as spacious per-device blocks (a compute bar over a
        # vram bar), NOT a separate table. Each device is its own numbered block, so
        # every GPU carries a compute AND a vram gauge with its own %.
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = _render_markup(r.render()).plain
        assert "CPU" in out and "MEM" in out
        # One labeled compute bar and one labeled vram bar per device.
        assert out.count("compute") == 4
        assert out.count("VRAM") == 4
        for i in range(4):
            assert f"CUDA {i}" in out  # each device labelled "CUDA N" in its own block
        _valid_markup(r.render())

    def test_multi_gpu_renders_aligned_gpu_section_head(self) -> None:
        # The GPU group gets the same marker · label section head the CPU/MEM rows
        # carry, aligned with them, so GPU reads as a first-class resource. Device
        # and active counts are facts; the reader judges from them.
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.gpus = [
            _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=0),
            _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=1),
            _make_gpu(0.0, 0, 55 * 1024**3, index=2),  # idle
        ]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        _valid_markup(r.render())
        out = _render_markup(r.render()).plain
        gpu_line = next(ln for ln in out.splitlines() if "devices" in ln)
        cpu_line = next(ln for ln in out.splitlines() if "CPU" in ln)
        assert "3 devices" in gpu_line and "2 active" in gpu_line  # 1 idle
        # The label starts at exactly the same column as the CPU/MEM labels.
        assert gpu_line.index("GPU") == cpu_line.index("CPU")
        # The section-head marker is the GPU IDENTITY colour (decorative), never a
        # health grade: an idle device must NOT turn the head red — colour asserts
        # no verdict (the per-device status word carries the fact instead).
        raw_gpu_line = next(ln for ln in r.render().split("\n") if "devices" in ln)
        assert f"[{_GPU_COLOR}]" in raw_gpu_line
        assert f"[{_HEALTH_COLOR['crit']}]" not in raw_gpu_line
        assert f"[{_HEALTH_COLOR['warn']}]" not in raw_gpu_line

    def test_gpu_block_vram_is_a_labeled_bar_with_percent(self) -> None:
        # The whole point of this view: VRAM is a colored bar + % (like compute),
        # not bare "used / total" text — so 'how full' reads as the same kind of
        # gauge as 'how busy', and an idle-but-VRAM-held device is obvious.
        r = _SizedRows(150)
        snap = _make_snapshot()
        # 0% compute, ~86% VRAM (120 of 140 GiB) — the idle-but-holding pattern.
        snap.gpus = [_make_gpu(0.0, 0, 120 * 1024**3, memtot=140 * 1024**3, index=0)]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        compute_ln = next(ln for ln in lines if "compute" in ln)
        vram_ln = next(ln for ln in lines if "VRAM" in ln)
        assert "0%" in compute_ln and "86%" in vram_ln
        assert "█" in vram_ln  # a filled vram bar (86%)
        assert "█" not in compute_ln  # empty compute bar (0%)
        assert "120 / 140 GiB" in vram_ln  # the amount the bar summarises
        # The two bars are stacked and their labels align in one column.
        assert compute_ln.index("compute") == vram_ln.index("VRAM")

    def test_gpu_block_compute_and_vram_bars_are_different_colours(self) -> None:
        # The two stacked bars use two DIFFERENT hues — compute the GPU violet,
        # vram a calm teal — so they read as distinct, comfortable colours (never
        # two shades of one hue, which looked either too similar or too bright).
        from slurmwatch.tui import _GPU_COLOR, _GPU_VRAM_BAR

        assert _GPU_VRAM_BAR != _GPU_COLOR  # genuinely different colours
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=1)]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        raw = r.render()
        assert f"[{_GPU_COLOR}]" in raw  # compute bar in the GPU violet
        assert f"[{_GPU_VRAM_BAR}]" in raw  # vram bar in the distinct teal

    def test_gpu_compute_shows_na_when_util_unreadable(self) -> None:
        # A2: when NVML can't read device util (a MIG slice / a transient failure),
        # the compute row shows "n/a" — never a false "0%" bar that would contradict
        # the "active" status. VRAM is still readable, so its bar stays.
        r = _SizedRows(150)
        snap = _make_snapshot()
        g = _make_gpu(0.0, 0, 55 * 1024**3, index=0)
        g.utilization_available = False
        snap.gpus = [g]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        compute_ln = next(ln for ln in lines if "compute" in ln)
        assert "n/a" in compute_ln
        assert "0%" not in compute_ln  # no false zero
        vram_ln = next(ln for ln in lines if "VRAM" in ln)
        assert "n/a" not in vram_ln  # VRAM is still readable

    def test_gpu_vram_shows_na_when_memory_unreadable(self) -> None:
        # The VRAM twin of the compute-unreadable case above: nvidia-ml-py >= 11.510
        # raises FunctionNotFound from nvmlDeviceGetMemoryInfo_v2 against a pre-510
        # driver, so a failed VRAM read must not render a false "0%" bar or "0 / 0
        # GiB" beside an otherwise-active device. Compute is still readable, so its
        # bar stays.
        r = _SizedRows(150)
        snap = _make_snapshot()
        g = _make_gpu(90.0, 0, 0, index=0)
        g.memory_available = False
        snap.gpus = [g]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        compute_ln = next(ln for ln in lines if "compute" in ln)
        assert "n/a" not in compute_ln  # compute is still readable
        vram_ln = next(ln for ln in lines if "VRAM" in ln)
        # The vram bar and the trailing GiB amount are on the same row.
        assert "0%" not in vram_ln  # no false zero bar
        assert "n/a / n/a GiB" in vram_ln
        assert "0 / 0 GiB" not in vram_ln  # no false zero amount

    def test_unreadable_readings_still_stack_in_their_columns(self) -> None:
        # An "n/a" is 3 cells wide but the reading it replaces may be 1-2, so the
        # column widths must be measured from the RENDERED string, not the raw
        # number — otherwise the "n/a" overflows its column and shoves the "/",
        # the degree mark and the " GiB" out of line between devices, which is the
        # exact stacking _GpuCols exists to guarantee. Asserting on substrings
        # alone can't see this; the column OFFSETS have to line up.
        blind = _make_gpu(90.0, 0, 0, index=0)
        blind.memory_available = False
        blind.temperature_available = False
        snap = _make_snapshot()
        # A readable neighbour with narrower figures than "n/a" (8 GiB used, 80
        # total, 65 °C) — the case that actually went ragged.
        snap.gpus = [
            blind,
            _make_gpu(45.0, 20 * 1024**3, 8 * 1024**3, memtot=80 * 1024**3, index=1),
        ]
        r = _SizedRows(150)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()

        vram = [ln for ln in lines if "VRAM" in ln]
        assert len(vram) == 2
        # " / " (spaced) is the separator; the "/" inside "n/a" is unspaced.
        assert len({ln.index(" / ") for ln in vram}) == 1, vram
        assert len({ln.index(" GiB") for ln in vram}) == 1, vram

        # Same for the temperature: the °C device must not be shifted by the n/a one.
        dev = [ln for ln in lines if "CUDA" in ln]
        assert len(dev) == 2
        assert len({ln.index(" W") for ln in dev}) == 1, dev

    def test_mixed_capacity_devices_align_their_vram_totals(self) -> None:
        # The capacity half of "used / total GiB" also needs a measured width: on a
        # genuinely mixed-capacity node (an 80 GiB card beside a 48 GiB one) the
        # unpadded total left the trailing " GiB" ragged.
        snap = _make_snapshot()
        snap.gpus = [
            _make_gpu(90.0, 40 * 1024**3, 55 * 1024**3, memtot=80 * 1024**3, index=0),
            _make_gpu(45.0, 20 * 1024**3, 8 * 1024**3, memtot=48 * 1024**3, index=1),
            _make_gpu(20.0, 10 * 1024**3, 4 * 1024**3, memtot=816 * 1024**3, index=2),
        ]
        r = _SizedRows(150)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        vram = [ln for ln in _render_markup(r.render()).plain.splitlines() if "VRAM" in ln]
        assert len(vram) == 3
        assert len({ln.index(" / ") for ln in vram}) == 1, vram
        assert len({ln.index(" GiB") for ln in vram}) == 1, vram

    def test_gpu_block_shows_power_against_the_cap(self) -> None:
        # #7: the enforced power cap is shown as "used / cap W" so headroom-to-cap is
        # visible — a GPU pegged near its cap is being fully driven, not sick, and
        # it's the context for a benign SwPowerCap throttle. Without a readable cap
        # (older pynvml -> 0.0) the row falls back to a bare "W".
        r = _SizedRows(150)
        snap = _make_snapshot()
        g = _make_gpu(100.0, 50 * 1024**3, 55 * 1024**3, index=0)
        g.power_watts = 348.0
        g.power_limit_watts = 350.0
        snap.gpus = [g]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CUDA 0" in ln)
        assert "348 / 350 W" in line
        g.power_limit_watts = 0.0
        line = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CUDA 0" in ln)
        assert "348 W" in line
        assert "348 / " not in line  # no cap -> a bare "W", not a half-empty ratio

    def test_mem_ws_pct_clamped_to_100(self) -> None:
        # P3: on a ConstrainRAMSpace=no node (or an imbalanced off-node step) the
        # working set can exceed the REPORTED limit (the allocation). The gauge is
        # clamped so it can't read a confusing ">100% of the request"; whether the
        # job is actually near OOM is decided separately, against the true kernel
        # kill point, so the clamp can't hide a real near-OOM.
        from slurmwatch.tui import _mem_ws_pct

        over = MemoryMetrics(
            current_bytes=12 << 30,
            limit_bytes=8 << 30,
            peak_bytes=12 << 30,
            usage_percent=100.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=12 << 30,  # 150% of the reported limit
        )
        assert _mem_ws_pct(over) == 100.0
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.memory = over
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        mem_ln = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "MEM" in ln)
        assert "100%" in mem_ln
        assert "150%" not in mem_ln
        # No limit at all -> 0, not a ZeroDivisionError.
        over.limit_bytes = 0
        assert _mem_ws_pct(over) == 0.0

    def test_cpu_peak_suffix_hidden_off_node(self) -> None:
        # Off-node the peak is a running max of an AVERAGE that normally equals the
        # figure right beside it, so the "· peak N" suffix would just restate it —
        # exactly why the MEM row drops its own suffix there. On-node it stays.
        snap = _make_snapshot()
        snap.cpu.effective_cores = 3.0
        snap.cpu.peak_effective_cores = 3.0
        for remote, expected in ((False, True), (True, False)):
            snap.remote = remote
            r = _SizedRows(150)
            r.snapshot = snap
            r.config = SlurmwatchConfig()
            cpu_ln = next(ln for ln in _render_markup(r.render()).plain.splitlines() if "CPU" in ln)
            assert ("peak" in cpu_ln) is expected, f"remote={remote}"

    def test_gpu_share_line_names_the_model(self) -> None:
        # The drill-in carries the model too, so the card is named even when the
        # dashboard was too narrow to show it.
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        g = _make_gpu(30.0, 4 * 1024**3, 8 * 1024**3, index=1)
        g.name = "NVIDIA H100 PCIe"
        line = _render_markup(screen._gpu_share_line(g, SlurmwatchConfig())).plain
        assert "CUDA 1" in line and "H100" in line
        assert line.index("CUDA 1") < line.index("H100") < line.index("this job")
        # A nameless device just omits it — no stray separator.
        g.name = ""
        bare = _render_markup(screen._gpu_share_line(g, SlurmwatchConfig())).plain
        assert "CUDA 1" in bare and "this job" in bare

    def test_device_label_is_the_cuda_ordinal_not_the_nvml_index(self) -> None:
        # C2: "CUDA N" claims to be the number the job's code uses. On a cluster with
        # no device-cgroup isolation NVML sees the whole node, so a job holding the
        # node's GPUs 2 and 3 was labelled "CUDA 2"/"CUDA 3" while its own code
        # addresses them as cuda:0/cuda:1. The label follows the ordinal.
        r = _SizedRows(150)
        snap = _make_snapshot()
        gpus = []
        for ordinal, index in ((0, 2), (1, 3)):
            g = _make_gpu(90.0, 20 * 1024**3, 30 * 1024**3, index=index)
            g.cuda_ordinal = ordinal
            gpus.append(g)
        snap.gpus = gpus
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        plain = _render_markup(r.render()).plain
        assert "CUDA 0" in plain and "CUDA 1" in plain
        assert "CUDA 2" not in plain and "CUDA 3" not in plain

    def test_device_label_falls_back_to_the_index_when_ordinal_unknown(self) -> None:
        # A remote node running a build from before the field exists sends no ordinal
        # (-1); the label then reads NVML's index, exactly as the whole UI used to.
        r = _SizedRows(150)
        snap = _make_snapshot()
        g = _make_gpu(90.0, 20 * 1024**3, 30 * 1024**3, index=5)
        assert g.cuda_ordinal == -1  # the default a skewed remote leaves behind
        snap.gpus = [g]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        assert "CUDA 5" in _render_markup(r.render()).plain

    def test_gpu_share_line_names_both_numbers_only_when_they_differ(self) -> None:
        # The drill-in has room the dashboard doesn't, so where the two diverge it
        # names the smi index too — that's what you'd type to cross-check. On an
        # isolated cluster (the common case) they're equal and it stays quiet.
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        cfg = SlurmwatchConfig()
        g = _make_gpu(30.0, 4 * 1024**3, 8 * 1024**3, index=2)
        g.cuda_ordinal = 0
        line = _render_markup(screen._gpu_share_line(g, cfg)).plain
        assert "CUDA 0" in line and "smi 2" in line
        same = _make_gpu(30.0, 4 * 1024**3, 8 * 1024**3, index=1)
        same.cuda_ordinal = 1
        assert "smi" not in _render_markup(screen._gpu_share_line(same, cfg)).plain

    def test_drillin_chart_titles_use_the_cuda_ordinal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The per-device history charts are captioned "CUDA N compute" / "CUDA N VRAM";
        # they must name the same N as the dashboard row and the share line above them.
        from collections import deque

        from slurmwatch.tui import ResourceDetailScreen

        class _Capture:
            text = ""

            def update(self, markup: str) -> None:
                self.text = markup

        chart = _Capture()
        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        monkeypatch.setattr(
            ResourceDetailScreen, "query_one", lambda self, sel, cls=None: chart, raising=False
        )
        monkeypatch.setattr(ResourceDetailScreen, "_chart_area_w", lambda self, w: 60)
        monkeypatch.setattr(ResourceDetailScreen, "_chart_height", lambda self: 8)
        g = _make_gpu(90.0, 20 * 1024**3, 30 * 1024**3, index=3)
        g.cuda_ordinal = 1
        hist = {3: deque([50.0, 90.0])}
        screen._render_gpu_charts([g], hist, hist, SlurmwatchConfig())
        plain = _render_markup(chart.text).plain
        assert "CUDA 1 compute" in plain and "CUDA 1 VRAM" in plain
        assert "CUDA 3 compute" not in plain

    def test_topology_grid_headers_use_the_cuda_ordinal(self) -> None:
        # The grid is keyed on NVML indices (the handle lookups need them), so without
        # the remap it would label devices differently from every other "CUDA N".
        from slurmwatch.tui import _topo_matrix_lines

        ic = GpuInterconnect(
            fabric="nvlink", devices=[2, 3], matrix=[["self", "NV4"], ["NV4", "self"]]
        )
        header = _render_markup(_topo_matrix_lines(ic, {2: 0, 3: 1})[0]).plain
        assert "CUDA0" in header and "CUDA1" in header
        # No map (or an unmapped device) keeps the index rather than inventing one.
        assert "CUDA2" in _render_markup(_topo_matrix_lines(ic)[0]).plain

    def test_gpu_share_line_compute_dash_on_mig(self) -> None:
        # A2 residual: the drill-in "this job" share line shows "—" for compute on a
        # MIG slice (util unsupported), matching its VRAM half, not a false "0%".
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        cfg = SlurmwatchConfig()
        mig = _make_gpu(0.0, 0, 8 * 1024**3, index=0)
        mig.utilization_supported = False
        line = screen._gpu_share_line(mig, cfg)
        assert "compute" in line and "0% compute" not in line and "—" in line
        # A util-capable device shows the real per-process compute %.
        normal = _make_gpu(30.0, 4 * 1024**3, 8 * 1024**3, index=1)
        assert "30% compute" in screen._gpu_share_line(normal, cfg)

    def test_gpu_blocks_align_across_devices_with_mixed_status_widths(self) -> None:
        # status_w pads every device's status word to the WIDEST present ("active"=6
        # vs "idle"=4), so a device with a shorter status word must not shift its
        # bars left — every device's compute/vram bars start in the SAME column.
        # Guards the inter-device alignment the fixed indent provides.
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.gpus = [
            _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=0),  # active (6)
            _make_gpu(0.0, 0, 0, index=1),  # idle (4)
            _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=2),  # active (6)
        ]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        assert any("idle" in ln for ln in lines)  # both status widths are present
        assert any("active" in ln for ln in lines)
        compute_cols = {ln.index("compute") for ln in lines if "compute" in ln}
        vram_cols = {ln.index("VRAM") for ln in lines if "VRAM" in ln}
        assert len(compute_cols) == 1  # all three compute bars in one column
        assert len(vram_cols) == 1  # all three vram bars in one column
        assert compute_cols == vram_cols  # compute and vram share the column

    def test_gpu_block_names_the_device_model(self) -> None:
        # WHICH card this is (H100 vs A100) is the one GPU fact the block can't
        # otherwise show, and it's what gives the power/VRAM numbers a frame of
        # reference — so the model sits right beside the "CUDA N" label.
        r = _SizedRows(150)
        snap = _make_snapshot()
        g = _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=0)
        g.name = "NVIDIA H100 PCIe"
        snap.gpus = [g]
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        cuda_ln = next(ln for ln in lines if "CUDA 0" in ln)
        assert "H100" in cuda_ln
        # The bus is NOT repeated here — the interconnect label already names it.
        assert "PCIe" not in cuda_ln
        # Between the ordinal and the status word, so the block reads
        # "which device · what card · is it working".
        assert cuda_ln.index("CUDA 0") < cuda_ln.index("H100") < cuda_ln.index("active")
        _valid_markup(r.render())

    def test_gpu_blocks_align_across_mixed_device_models(self) -> None:
        # A mixed-device node pads every model to the widest present, so a device
        # with a short model ("H100") must not shift its bars left — the same
        # inter-device alignment invariant the status column upholds.
        r = _SizedRows(150)
        snap = _make_snapshot()
        gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(3)]
        names = ["NVIDIA H100 PCIe", "NVIDIA RTX A6000", "NVIDIA GH200 480GB"]
        for g, nm in zip(gpus, names, strict=True):
            g.name = nm
        snap.gpus = gpus
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        # Three models of three different widths (4 / 9 / 5), so padding matters.
        assert any("H100" in ln for ln in lines) and any("RTX A6000" in ln for ln in lines)
        compute_cols = {ln.index("compute") for ln in lines if "compute" in ln}
        vram_cols = {ln.index("VRAM") for ln in lines if "VRAM" in ln}
        assert len(compute_cols) == 1
        assert len(vram_cols) == 1
        assert compute_cols == vram_cols

    def test_gpu_power_column_aligns_with_mixed_cap_readability(self) -> None:
        # "345 / 350 W" and a bare "346 W" (cap unreadable) are different LENGTHS, so
        # without justifying the whole power string the "W" — and the temperature
        # after it — went ragged between devices. Caps of different digit counts
        # (350 vs 1000 W) are the same trap.
        r = _SizedRows(150)
        snap = _make_snapshot()
        gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(3)]
        gpus[0].power_watts, gpus[0].power_limit_watts = 345.0, 350.0
        gpus[1].power_watts, gpus[1].power_limit_watts = 346.0, 0.0  # cap unreadable
        gpus[2].power_watts, gpus[2].power_limit_watts = 342.0, 1000.0
        snap.gpus = gpus
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = [ln for ln in _render_markup(r.render()).plain.splitlines() if "CUDA" in ln]
        assert len(lines) == 3
        assert len({ln.index(" W") for ln in lines}) == 1  # one "W" column
        assert len({ln.rindex("C") for ln in lines}) == 1  # so temperature aligns too

    def test_gpu_model_dropped_on_narrow_terminal(self) -> None:
        # The model is identity, not a live number, so it's the first thing dropped
        # when the terminal can't spare the width — the bars must never be pushed
        # off an 80-column SSH session to make room for it. The threshold is the one
        # that also narrows the bars (_NARROW_COLS = 100).
        snap = _make_snapshot()
        g = _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=0)
        g.name = "NVIDIA H100 PCIe"
        snap.gpus = [g]
        for width, expected in ((80, False), (99, False), (100, True), (150, True)):
            r = _SizedRows(width)
            r.snapshot = snap
            r.config = SlurmwatchConfig()
            plain = _render_markup(r.render()).plain
            assert ("H100" in plain) is expected, f"width {width}"
            assert max(len(ln) for ln in plain.splitlines()) <= width, f"width {width} overflows"

    def test_gpu_temp_carries_a_fahrenheit_reading(self) -> None:
        # NVML reports Celsius only; the block trails it with the same reading in
        # Fahrenheit so a reader who doesn't think in Celsius needn't convert. Both
        # units, correct arithmetic, and the °F column right-justified across devices
        # (a 2-digit °F under a 3-digit one) so the degree marks stack like the "W".
        snap = _make_snapshot()
        snap.gpus = [
            _make_gpu(96.0, 50 * 1024**3, 57 * 1024**3, memtot=80 * 1024**3, index=0),
            _make_gpu(4.0, 0, 2 * 1024**3, memtot=80 * 1024**3, index=1),
        ]
        snap.gpus[0].temperature_celsius = 74.0  # -> 165.2 -> 165°F
        snap.gpus[1].temperature_celsius = 30.0  # -> 86.0 ->  86°F
        r = _SizedRows(140)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = [ln for ln in _render_markup(r.render()).plain.splitlines() if "°C" in ln]
        assert len(lines) == 2
        assert "74°C (165°F)" in lines[0]
        assert "30°C ( 86°F)" in lines[1]  # shorter reading padded, not shifted
        assert len({ln.index("°F") for ln in lines}) == 1

    def test_unreadable_temperature_shows_na_not_zero_celsius(self) -> None:
        # NVML returns NOT_SUPPORTED for temperature on a MIG slice. Reading that as
        # 0 °C rendered a below-freezing card — and the °F conversion turned the
        # fabricated zero into a second, derived-looking measurement, "0°C (32°F)".
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(96.0, 50 * 1024**3, 57 * 1024**3, memtot=80 * 1024**3, index=0)]
        snap.gpus[0].temperature_celsius = 0.0
        snap.gpus[0].temperature_available = False
        r = _SizedRows(140)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        plain = _render_markup(r.render()).plain
        assert "0°C" not in plain
        assert "32°F" not in plain
        assert "n/a" in plain

    def test_fahrenheit_is_the_first_thing_dropped_on_a_narrow_terminal(self) -> None:
        # The °F reading is the same number said twice — the one figure in the block
        # that adds no information — so it goes before even the model label, and its
        # threshold sits above _NARROW_COLS. Below it, only Celsius remains.
        from slurmwatch.tui import _FAHRENHEIT_COLS, _NARROW_COLS

        assert _FAHRENHEIT_COLS > _NARROW_COLS  # dropped before the model label
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=0)]
        snap.gpus[0].temperature_celsius = 65.0
        for width, expected in ((80, False), (_FAHRENHEIT_COLS - 1, False), (150, True)):
            r = _SizedRows(width)
            r.snapshot = snap
            r.config = SlurmwatchConfig()
            plain = _render_markup(r.render()).plain
            assert "65°C" in plain, f"width {width} lost the Celsius reading"
            assert ("149°F" in plain) is expected, f"width {width}"
            assert max(len(ln) for ln in plain.splitlines()) <= width, f"width {width} overflows"

    def test_widest_gpu_block_with_fahrenheit_fits_at_its_threshold(self) -> None:
        # _FAHRENHEIT_COLS is derived as the label-threshold worst case plus the eight
        # cells " (196°F)" costs, so the same worst case must fit EXACTLY there —
        # otherwise the °F pushes the bars off a terminal that opted into showing it.
        from slurmwatch.tui import _FAHRENHEIT_COLS

        snap = _make_snapshot()
        gpus = []
        for i in (8, 9, 10):
            g = _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i)
            g.name = "NVIDIA SuperAccelerator 9000"  # trims to the 14-char ceiling
            g.temperature_celsius = 91.0  # hot -> 196°F and a trailing "⚠"
            g.power_watts = 1000.0
            g.power_limit_watts = 1000.0
            gpus.append(g)
        snap.gpus = gpus
        for ascii_mode in (False, True):
            r = _SizedRows(_FAHRENHEIT_COLS)
            r.snapshot = snap
            r.config = SlurmwatchConfig(ascii_mode=ascii_mode)
            plain = _render_markup(r.render()).plain
            assert any(("196F" if ascii_mode else "196°F") in ln for ln in plain.splitlines())
            widest = max(len(ln) for ln in plain.splitlines())
            over = widest - _FAHRENHEIT_COLS
            assert widest <= _FAHRENHEIT_COLS, f"ascii={ascii_mode} overflows by {over}"

    def test_widest_gpu_block_fits_at_the_label_threshold(self) -> None:
        # The label is gated on _NARROW_COLS, so the WORST case must still fit
        # exactly there: a max-length model, a two-digit index, the longer status
        # word, a hot-marked temperature and a "used / cap W" pair on every device.
        from slurmwatch.tui import _NARROW_COLS

        snap = _make_snapshot()
        gpus = []
        for i in (8, 9, 10):
            g = _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i)
            g.name = "NVIDIA SuperAccelerator 9000"  # trims to the 14-char ceiling
            g.temperature_celsius = 91.0  # hot -> trailing "⚠"
            g.power_watts = 1000.0
            g.power_limit_watts = 1000.0
            gpus.append(g)
        snap.gpus = gpus
        for ascii_mode in (False, True):
            r = _SizedRows(_NARROW_COLS)
            r.snapshot = snap
            r.config = SlurmwatchConfig(ascii_mode=ascii_mode)
            plain = _render_markup(r.render()).plain
            assert any("CUDA 10" in ln for ln in plain.splitlines())
            widest = max(len(ln) for ln in plain.splitlines())
            over = widest - _NARROW_COLS
            assert widest <= _NARROW_COLS, f"ascii={ascii_mode} overflows by {over}"

    def test_gpu_model_markup_is_escaped(self) -> None:
        # The model comes from the NVIDIA driver — the first driver-supplied text this
        # block feeds into console markup. Validated with TEXTUAL's parser, not
        # Rich's: Textual is the engine that runs every render, and it's stricter
        # ("[/]" -> MarkupError, killing the dashboard) *and* looser in a worse way
        # ("[red]" is silently consumed as a tag, so the label's VISIBLE width
        # shrinks and every device's bars slide out of column). Escaping must
        # therefore happen, and must happen AFTER padding so the width still counts
        # visible cells.
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        for hostile, visible in (
            ("NVIDIA H100 [/]x", "H100 [/]x"),  # unbalanced close tag -> MarkupError
            ("NVIDIA H100 [red]x", "H100 [red]x"),  # a real tag -> swallowed silently
            ("NVIDIA [experiment", "[experiment"),  # lone opener
        ):
            snap = _make_snapshot()
            gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
            gpus[0].name = hostile
            gpus[1].name = "NVIDIA H200"
            snap.gpus = gpus
            r = _SizedRows(150)
            r.snapshot = snap
            r.config = SlurmwatchConfig()
            plain = _textual_plain(r.render())  # raises MarkupError if unescaped
            assert visible in plain, f"{hostile!r} not rendered literally: {plain!r}"
            # Visible width preserved -> every device's bars still share one column.
            compute_cols = {ln.index("compute") for ln in plain.splitlines() if "compute" in ln}
            vram_cols = {ln.index("VRAM") for ln in plain.splitlines() if "VRAM" in ln}
            assert len(compute_cols) == 1, f"{hostile!r} misaligned: {compute_cols}"
            assert compute_cols == vram_cols
            # The drill-in share line carries the same text and must escape it too.
            share = _textual_plain(screen._gpu_share_line(gpus[0], SlurmwatchConfig()))
            assert visible in share

        # ...and specifically with the hostile model SHORTER than its sibling, so the
        # label genuinely needs padding: escaping before padding would count the
        # invisible backslash as a cell and under-pad, sliding the bars out of column.
        snap = _make_snapshot()
        gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
        gpus[0].name = "NVIDIA [x"  # -> "[x" (2 visible)
        gpus[1].name = "NVIDIA RTX A6000"  # -> "RTX A6000" (9 visible)
        snap.gpus = gpus
        r = _SizedRows(150)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        plain = _textual_plain(r.render())
        assert "[x" in plain and "RTX A6000" in plain
        compute_cols = {ln.index("compute") for ln in plain.splitlines() if "compute" in ln}
        assert len(compute_cols) == 1, f"short escaped label misaligned: {compute_cols}"

    def test_gpu_model_long_name_cannot_overflow(self) -> None:
        # A name longer than the ceiling is truncated (with an ASCII-safe marker
        # under --ascii) rather than pushing the bars past the terminal edge.
        snap = _make_snapshot()
        g = _make_gpu(50.0, 1 << 30, 4 << 30, index=0)
        g.name = "NVIDIA SomeVeryLongExperimentalAcceleratorName 9000 PCIe 128GB"
        snap.gpus = [g]
        for ascii_mode in (False, True):
            cfg = SlurmwatchConfig()
            cfg.ascii_mode = ascii_mode
            r = _SizedRows(124)
            r.snapshot = snap
            r.config = cfg
            plain = _render_markup(r.render()).plain
            assert max(len(ln) for ln in plain.splitlines()) <= 124
            if ascii_mode:  # --ascii purity: no stray Unicode ellipsis
                assert all(ord(c) < 128 for c in plain)

    def test_unobservable_gpu_note(self) -> None:
        # On the node (remote=False) but no readable GPU: we got here via the
        # --gres=none fallback (GPU held by the job's own step). Don't tell the
        # user to "run on the compute node" — they already are.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.remote = False
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = r.render()
        assert "2 requested" in out
        assert "srun step" in out  # names the real cause + the fix
        assert "run on the compute node" not in out
        _valid_markup(out)

    def test_unobservable_gpu_note_remote(self) -> None:
        # Off the node (remote summary path): the fix really is "go to the node".
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.remote = True
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = r.render()
        assert "2 requested" in out and "run on the compute node" in out
        _valid_markup(out)

    def test_unobservable_gpu_note_nvml_unavailable(self) -> None:
        # F3: on-node but NVML/pynvml couldn't start (no driver, pynvml not
        # installed, or a non-NVIDIA GPU) -> say so honestly, NOT the misleading
        # "GPU held by your srun step; run without srun" (a wild goose chase).
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.remote = False
        snap.gpu_monitoring_available = False
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = r.render()
        assert "2 requested" in out
        assert "no NVIDIA GPU telemetry" in out
        assert "srun step" not in out
        _valid_markup(out)

    def _denied_snapshot(self) -> TelemetrySnapshot:
        """A monitor step beside a job holding all its GPUs: the real-world case.

        Job 54117243 on beagle3-0015 (2-node, gpu:2 per node, A100): Slurm gave the
        job GPU indices 0 and 2 of the node's 4, NVML enumerated zero devices
        because the monitor step was allocated none, and slurmwatch called that a
        missing driver.
        """
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.remote = False
        snap.gpu_monitoring_available = False
        snap.gpu_unavailable_reason = "devices_denied"
        snap.gpu_node_count = 4
        snap.gpu_node_model = "NVIDIA A100-PCIE-40GB"
        snap.gpu_allocated_indices = [0, 2]
        return snap

    def test_devices_denied_names_the_hardware_and_not_a_missing_driver(self) -> None:
        r = ResourceRows()
        r.snapshot = self._denied_snapshot()
        r.config = SlurmwatchConfig()
        out = _plain(r.render())
        # It must state what the job actually holds on this node...
        assert "A100-PCIE-40GB" in out
        assert "idx 0,2" in out
        # ...and must NOT blame a missing driver or a non-NVIDIA GPU, which is what
        # sent the user hunting for a broken install / a mis-sized 2-node request.
        assert "no driver" not in out
        assert "non-NVIDIA" not in out
        assert "no NVIDIA GPU telemetry" not in out
        # Nor tell them to go to the compute node — this IS the compute node.
        assert "run on the compute node" not in out
        _valid_markup(r.render())

    def test_devices_denied_line_stays_inside_the_panel(self) -> None:
        """The GPU row shares one line with the reason; a wordy message overflows it.

        Asserting on rendered WIDTH, not on substrings: a message that spills past
        the panel still contains every expected substring, so only the measurement
        catches it (the same blind spot that let an unreadable GPU reading break
        column alignment before).
        """
        r = ResourceRows()
        r.snapshot = self._denied_snapshot()
        r.config = SlurmwatchConfig()
        gpu_lines = [ln for ln in _plain(r.render()).splitlines() if "GPU" in ln]
        assert gpu_lines, "no GPU row rendered"
        # The narrowest terminal the dashboard targets leaves ~100 cols inside the
        # RESOURCES border; the previous wording used 82.
        assert max(len(ln) for ln in gpu_lines) <= 100, gpu_lines

    def test_reason_specific_notes_do_not_collapse_together(self) -> None:
        """Each cause gets its own sentence; that is the whole point of the field."""
        from slurmwatch.tui import _gpu_unavailable_note

        snap = self._denied_snapshot()
        notes = {}
        for reason in ("devices_denied", "no_pynvml", "nvml_error", "no_driver", ""):
            snap.gpu_unavailable_reason = reason
            notes[reason] = _gpu_unavailable_note(snap, "-", long=False)
        assert "pynvml is not installed" in notes["no_pynvml"]
        assert "NVML failed to start" in notes["nvml_error"]
        # An unknown/absent reason (an older node streaming to a new UI) keeps the
        # original wording rather than inventing a cause it does not know.
        assert notes[""] == notes["no_driver"]
        assert "no NVIDIA GPU telemetry" in notes["no_driver"]
        assert len(set(notes.values())) == 4

    def test_denied_note_names_the_srun_cause_and_the_remedy(self) -> None:
        """ "Can't read it" is a dead end; the reader needs the cause and the fix.

        The blocker is the job's own inner `srun` step holding every GPU — a job that
        launches its work directly in the batch script (a single-node torchrun) keeps
        its GPUs readable and shows full utilization. Stating only the symptom sent
        the user hunting for a slurmwatch bug that wasn't there.
        """
        from slurmwatch.tui import _gpu_unavailable_note

        snap = self._denied_snapshot()
        short = _gpu_unavailable_note(snap, "-", long=False)
        assert "srun step" in short
        assert "g" in short  # points at the drill-in, where the remedy fits
        long_note = _gpu_unavailable_note(snap, "-", long=True)
        assert "without an inner srun" in long_note.lower()
        assert "batch script" in long_note
        # And the escape hatch when an inner srun is unavoidable (multi-node).
        assert "slurmwatch --log" in long_note or "nvidia-smi" in long_note

    def test_long_form_adds_the_node_total_and_the_way_out(self) -> None:
        from slurmwatch.tui import _gpu_hardware_label, _gpu_unavailable_note

        snap = self._denied_snapshot()
        label = _gpu_hardware_label(snap, False, long=True)
        assert "2 of the node's 4" in label and "idx 0,2" in label
        long_note = _gpu_unavailable_note(snap, "-", long=True)
        # The drill-in has room to say why AND what would actually work.
        assert "cannot share a GPU" in long_note
        assert "nvidia-smi" in long_note

    def test_offnode_sample_never_names_the_local_gpus(self) -> None:
        """An sstat estimate must not label a login node's hardware as the job's.

        The hardware label describes the machine the collector ran on. Off-node that
        is not where the job is, so the row falls back to the plain count and the
        "go to the compute node" note.
        """
        r = ResourceRows()
        snap = self._denied_snapshot()
        snap.remote = True
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = _plain(r.render())
        assert "A100-PCIE-40GB" not in out
        assert "idx 0,2" not in out
        assert "2 requested" in out
        assert "run on the compute node" in out
        _valid_markup(r.render())

    def test_ascii_mode_uses_no_multiplication_sign(self) -> None:
        from slurmwatch.tui import _gpu_hardware_label

        snap = self._denied_snapshot()
        assert "\u00d7" not in _gpu_hardware_label(snap, True)
        assert "x A100-PCIE-40GB" in _gpu_hardware_label(snap, True)

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


def _nvlink_ic(devices: int = 4, version: int = 3) -> GpuInterconnect:
    n = devices
    matrix = [["self" if i == j else "NV12" for j in range(n)] for i in range(n)]
    return GpuInterconnect(
        fabric="nvlink",
        nvlink_version=version,
        links_per_gpu=12,
        link_speed_gbps=25.0,
        per_gpu_gbps=600.0,
        nvswitch=True,
        devices=list(range(n)),
        matrix=matrix,
        nvlink_rx_gbps=[50.0] * n,
        nvlink_tx_gbps=[40.0] * n,
    )


def _pcie_ic() -> GpuInterconnect:
    return GpuInterconnect(fabric="pcie", devices=[0, 1], matrix=[["self", "SYS"], ["SYS", "self"]])


class TestInterconnectRendering:
    def test_label_variants(self) -> None:
        assert _interconnect_label(_nvlink_ic(version=3)) == "NVLink 3"
        assert _interconnect_label(_nvlink_ic(version=4)) == "NVLink 4"
        assert _interconnect_label(_pcie_ic()) == "PCIe"
        mixed = GpuInterconnect(fabric="mixed", nvlink_version=4, devices=[0, 1])
        assert _interconnect_label(mixed) == "NVLink 4 + PCIe"
        # version 0 (unknown gen) still labels as NVLink, not "NVLink 0"
        assert _interconnect_label(GpuInterconnect(fabric="nvlink")) == "NVLink"

    def test_summary_has_speed_and_bandwidth(self) -> None:
        out = _plain(_interconnect_summary(_nvlink_ic(), False))
        assert "NVLink 3" in out
        assert "12 links/GPU" in out
        assert "25 GB/s per link" in out
        assert "600 GB/s per GPU" in out
        assert "NVSwitch" in out
        _valid_markup(_interconnect_summary(_nvlink_ic(), False))

    def test_summary_pcie_explains_no_nvlink(self) -> None:
        pcie = _pcie_ic()
        out = _plain(_interconnect_summary(pcie, False))
        assert "PCIe" in out and "not NVLink-connected" in out
        _valid_markup(_interconnect_summary(pcie, False))

    def test_matrix_is_square_and_labeled(self) -> None:
        lines = _topo_matrix_lines(_nvlink_ic(4))
        _valid_markup("\n".join(lines))
        plain = [_render_markup(ln).plain for ln in lines]
        # header + 4 device rows
        assert len(plain) == 5
        for i in range(4):
            assert f"CUDA{i}" in plain[0]  # every device is a column
            assert plain[i + 1].split()[0] == f"CUDA{i}"  # and a row label
        # self-diagonal renders as X, off-diagonal as the NVLink count
        assert plain[1].count("NV12") == 3  # CUDA0 row: 3 peers
        assert "X" in plain[1]

    def test_matrix_columns_align(self) -> None:
        # Ragged cell widths (NV12 vs SYS vs X) must be padded to a common column so
        # the grid reads as a table, not a jumble.
        lines = [_render_markup(ln).plain for ln in _topo_matrix_lines(_nvlink_ic(3))]
        # Every rendered row is the same visual width once padded.
        assert len({len(ln) for ln in lines}) == 1

    def test_legend_explains_each_present_code(self) -> None:
        # An all-NVLink fabric: only the NV code, no PCIe codes, and no meaningless
        # "fast→slow" ordering.
        nv = _plain(_topo_legend(_nvlink_ic(), False))
        assert "NVn" in nv and "n NVLinks" in nv
        assert "fast" not in nv and "slow" not in nv
        assert "PCIe switch" not in nv  # no PCIe cells present to explain
        # A mixed fabric: the NV code plus each PCIe code with its real meaning.
        mixed = GpuInterconnect(
            fabric="mixed",
            devices=[0, 1, 2],
            matrix=[["self", "NV4", "PHB"], ["NV4", "self", "PHB"], ["PHB", "PHB", "self"]],
        )
        m = _plain(_topo_legend(mixed, False))
        assert "NVn" in m
        assert "PHB = via the CPU (host bridge)" in m
        # A PCIe-only fabric with PIX cells gets the PIX gloss.
        pix = GpuInterconnect(
            fabric="pcie", devices=[0, 1], matrix=[["self", "PIX"], ["PIX", "self"]]
        )
        assert "PIX = one PCIe switch" in _plain(_topo_legend(pix, False))
        # A pair whose common ancestor NVML wouldn't report renders "?" in the grid;
        # it must be glossed like any other code, not left as an unexplained cell.
        unknown = GpuInterconnect(
            fabric="pcie", devices=[0, 1], matrix=[["self", "?"], ["?", "self"]]
        )
        assert "? = NVML wouldn't say" in _plain(_topo_legend(unknown, False))

    def test_pcie_class_asymmetry_is_reported_verbatim(self) -> None:
        # Different cells for different pairs is the HARDWARE, not a bug: two cards
        # under one PCIe switch are PIX, a third on the other socket's root complex is
        # SYS (verified identical to `nvidia-smi topo -m` on midway3-0372). The grid
        # must report each pair as measured, and the legend must explain both codes,
        # so a reader can tell the fast pair from the slow one.
        ic = GpuInterconnect(
            fabric="pcie",
            devices=[0, 1, 2],
            matrix=[
                ["self", "PIX", "SYS"],
                ["PIX", "self", "SYS"],
                ["SYS", "SYS", "self"],
            ],
        )
        rows = [_plain(ln) for ln in _topo_matrix_lines(ic)]
        assert "PIX" in rows[1] and "SYS" in rows[1]  # CUDA0: PIX to 1, SYS to 2
        assert rows[3].count("SYS") == 2  # CUDA2: SYS to both
        legend = _plain(_topo_legend(ic, False))
        assert "PIX = one PCIe switch" in legend and "SYS = across NUMA nodes" in legend

    def test_traffic_lines_sum_and_pick_fabric(self) -> None:
        # NVLink fabric → one NVLink line summed over devices, in GB/s.
        ic = _nvlink_ic(4)  # tx 40*4 = 160, rx 50*4 = 200 GB/s
        lines = _topo_traffic_lines(ic, False)
        assert len(lines) == 1
        out = _plain(lines[0])
        assert "NVLink" in out and "160.0" in out and "200.0" in out and "GB/s" in out
        # No counters (older driver / no permission) → no line at all.
        ic.nvlink_rx_gbps = []
        ic.nvlink_tx_gbps = []
        assert _topo_traffic_lines(ic, False) == []
        # PCIe fabric → a PCIe line instead.
        pcie = GpuInterconnect(
            fabric="pcie",
            devices=[0, 1],
            matrix=[["self", "PIX"], ["PIX", "self"]],
            pcie_rx_gbps=[8.0, 6.0],
            pcie_tx_gbps=[12.0, 10.0],
        )
        p = _topo_traffic_lines(pcie, False)
        assert len(p) == 1
        pout = _plain(p[0])
        assert "PCIe" in pout and "22.0" in pout and "14.0" in pout  # tx 22, rx 14

    def test_pcie_traffic_line_says_host_copies_are_included(self) -> None:
        # NVML's PCIe counter is the whole link — host↔GPU copies plus any peer-to-
        # peer — while the NVLink counter really is the links' own (GPU↔GPU) traffic.
        # Under a "these GPUs are not NVLink-connected" heading an unqualified PCIe
        # number reads as inter-GPU traffic, so the PCIe line (and only it) says what
        # the counter actually covers.
        pcie = GpuInterconnect(
            fabric="pcie",
            devices=[0, 1],
            matrix=[["self", "PIX"], ["PIX", "self"]],
            pcie_rx_gbps=[8.0, 6.0],
            pcie_tx_gbps=[12.0, 10.0],
        )
        assert "host copies included" in _plain(_topo_traffic_lines(pcie, False)[0])
        nv = _plain(_topo_traffic_lines(_nvlink_ic(4), False)[0])
        assert "NVLink" in nv and "host copies" not in nv
        # --ascii keeps the line pure ASCII.
        ascii_line = _plain(_topo_traffic_lines(pcie, True)[0])
        assert "host copies included" in ascii_line
        assert all(ord(c) < 128 for c in ascii_line)

    def test_traffic_glance_sums_all_fabrics(self) -> None:
        # The compact head tag: total up/down over the fabric, summed across devices.
        nv = _plain(_interconnect_traffic_glance(_nvlink_ic(4), False))  # tx 160, rx 200
        assert "↑ 160.0" in nv and "↓ 200.0" in nv and "GB/s" in nv
        _valid_markup(_interconnect_traffic_glance(_nvlink_ic(4), False))
        # A mixed fabric adds NVLink and PCIe together into one head total.
        mixed = GpuInterconnect(
            fabric="mixed",
            devices=[0, 1],
            nvlink_rx_gbps=[10.0, 10.0],
            nvlink_tx_gbps=[5.0, 5.0],
            pcie_rx_gbps=[1.0, 1.0],
            pcie_tx_gbps=[2.0, 2.0],
        )
        m = _plain(_interconnect_traffic_glance(mixed, False))
        assert "↑ 14.0" in m and "↓ 22.0" in m  # tx 5+5+2+2, rx 10+10+1+1
        # ASCII mode swaps the arrows for ^/v.
        assert "^ 14.0" in _plain(_interconnect_traffic_glance(mixed, True))

    def test_traffic_switches_to_mbps_instead_of_showing_zero(self) -> None:
        # A pair rendered as "%.1f GB/s" prints "0.0" for anything under 50 MB/s, so
        # a fabric genuinely moving tens of MB/s read as completely idle on the
        # dashboard head. Below 0.1 GB/s the pair switches to MB/s instead.
        low = GpuInterconnect(
            fabric="pcie",
            devices=[0, 1, 2],
            matrix=[["self"]],
            pcie_rx_gbps=[0.021, 0.025, 0.025],  # 71 MB/s total
            pcie_tx_gbps=[0.015, 0.014, 0.010],  # 39 MB/s total
        )
        head = _plain(_interconnect_traffic_glance(low, False))
        assert "MB/s" in head and "GB/s" not in head
        assert "↑ 39.0" in head and "↓ 71.0" in head
        assert "0.0" not in head  # the old rendering said "↑ 0.0 ↓ 0.1 GB/s"
        # The drill-in line must agree with the head (same unit, same numbers).
        drill = _plain(_topo_traffic_lines(low, False)[0])
        assert "↑ 39.0" in drill and "↓ 71.0" in drill and "MB/s (all devices" in drill
        # Both values share ONE unit, chosen from the larger: a pair straddling the
        # boundary stays comparable rather than mixing GB/s with MB/s.
        straddle = GpuInterconnect(
            fabric="pcie", devices=[0], matrix=[["self"]], pcie_rx_gbps=[4.0], pcie_tx_gbps=[0.002]
        )
        s = _plain(_interconnect_traffic_glance(straddle, False))
        assert "↑ 0.0" in s and "↓ 4.0" in s and "GB/s" in s
        # A genuine zero still reads zero (in MB/s) — no fake floor.
        zero = GpuInterconnect(
            fabric="pcie", devices=[0], matrix=[["self"]], pcie_rx_gbps=[0.0], pcie_tx_gbps=[0.0]
        )
        assert "↑ 0.0" in _plain(_interconnect_traffic_glance(zero, False))
        # Real GB/s-scale traffic is unchanged, and ASCII mode still swaps arrows.
        assert "↑ 160.0" in _plain(_interconnect_traffic_glance(_nvlink_ic(4), False))
        assert "^ 39.0" in _plain(_interconnect_traffic_glance(low, True))
        _valid_markup(_interconnect_traffic_glance(low, False))
        _valid_markup(_topo_traffic_lines(low, True)[0])

    def test_traffic_glance_empty_when_no_counters(self) -> None:
        # PCIe-only fabric whose live counters aren't readable → no tag at all, so
        # the head shows just the bare "PCIe" label.
        assert _interconnect_traffic_glance(_pcie_ic(), False) == ""
        # Half-present (rx but no tx) is also treated as unreadable, matching the
        # drill-in's both-or-nothing rule.
        half = GpuInterconnect(fabric="nvlink", devices=[0, 1], nvlink_rx_gbps=[1.0, 1.0])
        assert _interconnect_traffic_glance(half, False) == ""

    def test_block_is_valid_markup_in_both_modes(self) -> None:
        for ascii_mode in (False, True):
            _valid_markup(_interconnect_block(_nvlink_ic(), ascii_mode))
            _valid_markup(_interconnect_block(_nvlink_ic(8), ascii_mode))

    def test_dashboard_head_tags_multi_gpu_fabric(self) -> None:
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
        snap.interconnect = _nvlink_ic(4)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        out = _render_markup(r.render()).plain
        gpu_line = next(ln for ln in out.splitlines() if "devices" in ln)
        assert "NVLink 3" in gpu_line
        # …and the live fabric traffic sits right after the label (tx 40*4, rx 50*4).
        assert "↑ 160.0" in gpu_line and "↓ 200.0" in gpu_line and "GB/s" in gpu_line
        assert gpu_line.index("NVLink 3") < gpu_line.index("↑ 160.0")
        _valid_markup(r.render())

    def test_dashboard_head_omits_fabric_for_single_gpu(self) -> None:
        # A single-GPU job has no interconnect (nothing to wire) — the head must not
        # sprout an NVLink/PCIe tag.
        r = _SizedRows(150)
        snap = _make_snapshot()  # one GPU, interconnect None
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        lines = _render_markup(r.render()).plain.splitlines()
        gpu_line = next(ln for ln in lines if "device" in ln)
        assert "NVLink" not in gpu_line and "PCIe" not in gpu_line


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
            job_name="argonne35-pretrain",
        )
        b.job_ctx = ctx
        b.config = SlurmwatchConfig()
        return b

    def test_multinode_bar_names_the_node_switch_key(self) -> None:
        """ "node 1 of 2" must not advertise other nodes without naming the key.

        The bar stated that more nodes existed and said nothing about how to reach
        them, so a 2-node job read as "there is no per-node data" when the other
        node's CPU/MEM/GPU was one arrow key away. `[`/`]` are NOT bound, so a user
        guessing at brackets gets silence.
        """
        b = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.node_count = 2
        snap.node_index = 0
        b.snapshot = snap
        out = _plain(b.render())
        assert "node 1 of 2" in out
        assert "switch node" in out
        _valid_markup(b.render())

    def test_single_node_bar_has_no_switch_hint(self) -> None:
        """Nowhere else to go, so the hint would be noise."""
        b = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.node_count = 1
        b.snapshot = snap
        out = _plain(b.render())
        assert "switch node" not in out

    def test_switch_hint_is_ascii_safe(self) -> None:
        b = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.node_count = 2
        b.snapshot = snap
        b.config = SlurmwatchConfig(ascii_mode=True)
        out = _plain(b.render())
        assert "switch node" in out
        assert "\u2190" not in out and "\u2192" not in out

    def test_labels_every_field(self) -> None:
        out = _plain(self._bar(24 * 3600).render())
        assert "job 12345" in out  # from the live snapshot
        assert "user youzhi" in out
        assert "partition test" in out
        assert "node midway3-0372" in out

    def test_job_name_rides_beside_the_id(self) -> None:
        # This bar is the only thing guaranteed on screen — the JOB card that carries
        # the name in full is the first thing off the bottom on a short terminal. So
        # the name sits right after the id, which alone can't say WHICH experiment.
        out = _plain(self._bar(24 * 3600).render())
        assert "argonne35-pretrain" in out
        assert out.index("job 12345") < out.index("argonne35-pretrain") < out.index("user youzhi")
        # It survives the compact (single-line) bar too — that's the case it's for.
        bar = self._bar(24 * 3600)
        bar.compact = True
        assert "argonne35-pretrain" in _plain(bar.render())

    def test_job_name_absent_or_hostile(self) -> None:
        from slurmwatch.tui import _JOB_NAME_MAX

        # No name from Slurm -> no chip, no stray separator before "user".
        bar = self._bar(24 * 3600)
        assert bar.job_ctx is not None
        bar.job_ctx.job_name = ""
        out = _plain(bar.render())
        assert "job 12345" in out and "user youzhi" in out
        # A markup-smuggling name renders literally instead of crashing the TUI...
        bar.job_ctx.job_name = "exp[/]a35"
        _valid_markup(bar.render())
        assert "exp[/]a35" in _plain(bar.render())
        # ...and a very long one is capped, so it can't push the other chips to row 2.
        bar.job_ctx.job_name = "x" * 200
        assert "x" * 200 not in _plain(bar.render())
        assert "x" * (_JOB_NAME_MAX - 1) in _plain(bar.render())

    def test_markup_in_identity_fields_does_not_crash(self) -> None:
        # audit-3 #1/#7: a job name can smuggle a `Partition=[/]` token into
        # scontrol's first line, poisoning ctx.partition; an unescaped `[/]` used
        # to crash the whole TUI via Textual's markup parser. Every identity field
        # must be escaped and render the value literally.
        b = self._bar(24 * 3600)
        assert b.job_ctx is not None and b.snapshot is not None
        b.job_ctx.partition = "[/]"
        b.job_ctx.username = "[red]evil[/]"
        b.job_ctx.nodelist = "cn[/bold]"
        b.snapshot.job_id = "12[/]45"
        b.snapshot.node_count = 1  # so node = ctx.nodelist
        out = _plain(b.render())  # must not raise MarkupError
        assert "[/]" in out and "[red]evil[/]" in out and "cn[/bold]" in out

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

    def test_a_negative_elapsed_never_renders_a_negative_percentage(self) -> None:
        """This site took `snap.elapsed_seconds` raw; its sibling clamps.

        `_time_budget` (tui.py:5749) computes `elapsed = max(0, ...)` before the
        identical `frac` / `remaining` / `frac_left` arithmetic. This one did not.
        `min(100.0, ...)` caps the top of the percentage and not the bottom, so a
        negative arrived as "-200%", and `max(0, limit - elapsed)` reported more
        time remaining than the limit it was measured against.

        `from_dict` now clamps too, so a negative can no longer arrive over the
        node hop -- this keeps the render site honest on its own terms, which is
        the state its sibling was already in.
        """
        bar = self._bar(3600)
        assert bar.snapshot is not None
        bar.snapshot.elapsed_seconds = -7196  # 2h of clock skew, as measured
        out = _render_markup(bar.render()).plain
        assert "-" not in out.split("%")[0].split()[-1], out
        assert "%" in out
        # And the impossible pair: never more remaining than the limit.
        assert "02:59:56" not in out, out
        assert "01:00:00" in out and "limit" in out, out

    def test_a_normal_elapsed_still_renders_its_real_percentage(self) -> None:
        """CONTROL -- passes in both states; the clamp must not flatten real times.

        `_make_snapshot` has elapsed 3600s, so a 3600s limit is exactly 100%. A
        `max(0, ...)` that also capped or zeroed the value would break this.
        """
        bar = self._bar(3600)
        out = _render_markup(bar.render()).plain
        assert "100%" in out, out
        assert "01:00:00" in out, out

    def test_identity_and_time_lines_breathe(self) -> None:
        # The docked bar's two blocks are separated by a blank line (not crammed
        # together crushed against the footer). The identity block may itself wrap
        # between chips on a narrow terminal, so assert on the SEPARATOR rather than
        # a fixed line count.
        lines = self._bar(24 * 3600).render().split("\n")
        assert "" in [ln.strip() for ln in lines], lines
        blank = [i for i, ln in enumerate(lines) if not ln.strip()]
        assert len(blank) == 1, lines
        assert 0 < blank[0] < len(lines) - 1, "the blank line separates two blocks"

    def test_the_bar_names_its_transport(self) -> None:
        """SW-20: the dashboard never said which data source it was on, while the
        prose summary always did ("source: sstat (remote; …)"). The two are not
        interchangeable — off-node, MaxRSS stands in for the working set, there is no
        cache breakdown, and the guard measures a high-water mark — and the choice
        isn't deterministic: the hop is bounded by --immediate, so falling back is
        normal. Two materially different views looked identical."""
        bar = self._bar(24 * 3600)
        assert "source cgroup" in _plain(bar.render())

        snap = _make_snapshot()
        snap.remote = True
        bar.snapshot = snap
        remote = _plain(bar.render())
        assert "source sstat" in remote, remote
        assert "no cache" in remote, "the degradation has to be named, not just the source"
        assert "cgroup" not in remote

    def test_the_transport_says_when_it_measures_nothing_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a JobAcctGatherType=none cluster sstat gathers nothing, so every
        off-node row reads zero permanently. "(peaks, no cache)" would describe a
        measurement that was never taken."""
        import slurmwatch.tui as tui_mod

        monkeypatch.setattr(tui_mod, "acct_gather_disabled", lambda: True)
        bar = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.remote = True
        bar.snapshot = snap
        out = _plain(bar.render())
        assert "source sstat" in out
        assert "gathers nothing" in out, out
        assert "peaks, no cache" not in out

    def test_a_gathering_cluster_keeps_the_ordinary_caveat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import slurmwatch.tui as tui_mod

        monkeypatch.setattr(tui_mod, "acct_gather_disabled", lambda: False)
        bar = self._bar(24 * 3600)
        snap = _make_snapshot()
        snap.remote = True
        bar.snapshot = snap
        out = _plain(bar.render())
        assert "peaks, no cache" in out and "gathers nothing" not in out

    def test_on_node_never_consults_the_gather_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cgroup path measures directly; sstat's config is irrelevant there."""
        import slurmwatch.tui as tui_mod

        def _boom() -> bool:
            raise AssertionError("must not be consulted on the cgroup path")

        monkeypatch.setattr(tui_mod, "acct_gather_disabled", _boom)
        assert "source cgroup" in _plain(self._bar(24 * 3600).render())

    def test_a_short_terminal_keeps_its_gauges(self) -> None:
        """The label must never cost a RESOURCES row: compact stays one line, and the
        transport is still legible there from the MEM bar's "peak" label."""
        bar = self._bar(24 * 3600)
        bar.compact = True
        assert "\n" not in bar.render()

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
        from slurmwatch.units import format_cores as _fmt_cores

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
        "job_name": "train-llama-8b",
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


class TestJobAnchor:
    """The header's short "which job is this" string."""

    def test_prefers_the_name(self) -> None:
        from slurmwatch.tui import _job_anchor

        assert _job_anchor("argonne35-pretrain", "52638815") == "argonne35-pretrain"

    def test_falls_back_to_the_id_without_a_name(self) -> None:
        from slurmwatch.tui import _job_anchor

        # Exactly what the header showed before the name existed.
        assert _job_anchor("", "52638815") == "job 52638815"
        # ...and the id already carries the array task there ("52330903_7").
        assert _job_anchor("", "52330903_7", array_task_id="7") == "job 52330903_7"

    def test_array_task_index_disambiguates_a_shared_name(self) -> None:
        from slurmwatch.tui import _job_anchor

        assert _job_anchor("sweep", "52330903_7", array_task_id="7") == "sweep · task 7"
        assert _job_anchor("sweep", "52330903_7", True, "7") == "sweep - task 7"

    def test_long_name_is_capped(self) -> None:
        from slurmwatch.tui import _JOB_NAME_MAX, _job_anchor

        anchor = _job_anchor("x" * 200, "1")
        assert anchor == "x" * (_JOB_NAME_MAX - 1) + "…"

    def test_ascii_mode_never_leaks_a_unicode_glyph(self) -> None:
        from slurmwatch.tui import _job_anchor

        for anchor in (
            _job_anchor("x" * 200, "1", ascii_mode=True),
            _job_anchor("sweep", "1", True, "7"),
            _job_anchor("", "1", ascii_mode=True),
        ):
            assert anchor.isascii(), anchor


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

    def test_shows_the_job_name_first(self) -> None:
        # The name is the one label the user chose, so it's what answers "which of my
        # jobs is this" — it led the selector and the pending view but was missing from
        # the running-job dashboard entirely. It heads the identity group.
        out = _plain(self._panel(_provenance_ctx()).render())
        assert "name train-llama-8b" in out
        assert out.index("name train-llama-8b") < out.index("account rcc-staff")

    def test_job_name_omitted_when_slurm_gives_none(self) -> None:
        # No name -> no chip and no stray separator; the group still starts cleanly.
        out = _plain(self._panel(_provenance_ctx(job_name="")).render())
        assert "name " not in out
        assert out.lstrip().startswith("account rcc-staff")

    def test_long_job_name_is_elided_not_wrapped(self) -> None:
        # Sweep scripts generate very long names; one must not claim the whole line
        # that account / qos / state share.
        from slurmwatch.tui import _JOB_NAME_MAX

        long_name = "sweep-lr3e4-seed7-wd0.01-warmup2000-cosine-run17-final"
        assert len(long_name) > _JOB_NAME_MAX  # the case this test is about
        out = _plain(self._panel(_provenance_ctx(job_name=long_name)).render())
        # Elided to exactly the cap, so the chip's width is bounded no matter what
        # sbatch -J was given; the raw name appears nowhere.
        expected = long_name[: _JOB_NAME_MAX - 1] + "…"
        assert f"name {expected}" in out
        assert long_name not in out
        # It displaced nothing: the other chips are all still present (they may wrap
        # to the next line, which is _pack_chips keeping each chip whole).
        for chip in ("account rcc-staff", "qos normal", "state RUNNING"):
            assert chip in out

    def test_job_name_markup_is_escaped(self) -> None:
        # sbatch -J takes anything: a lone "[" is a Textual MarkupError (the whole
        # dashboard dies) and a real tag like "[red]" would be silently swallowed.
        for hostile in ("exp[red]-a35", "run[/]x", "[experiment"):
            panel = self._panel(_provenance_ctx(job_name=hostile))
            _valid_markup(panel.render())
            assert hostile in _plain(panel.render())

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

    def test_shows_stdout_and_stderr_log_paths(self) -> None:
        # The log files a user tails are exactly the paths the rest of the UI never
        # carries, so the card points straight at them.
        ctx = _provenance_ctx(
            std_out="/home/ada/proj/runs/job-9.out",
            std_err="/home/ada/proj/runs/job-9.err",
        )
        out = _plain(self._panel(ctx).render())
        assert "stdout" in out and "/home/ada/proj/runs/job-9.out" in out
        assert "stderr" in out and "/home/ada/proj/runs/job-9.err" in out

    def test_merges_stdout_and_stderr_when_they_are_the_same_file(self) -> None:
        # Slurm merges the two streams by default; one "output" row then says it
        # all — a second identical line would waste space, not add information.
        same = "/home/ada/proj/runs/slurm-9.out"
        out = _plain(self._panel(_provenance_ctx(std_out=same, std_err=same)).render())
        assert "output" in out and same in out
        assert "stdout" not in out and "stderr" not in out
        assert out.count(same) == 1  # the path is shown once, not twice

    def test_omits_log_paths_when_absent(self) -> None:
        # An interactive job has no log files (scontrol StdOut/StdErr empty) — the
        # card drops the rows rather than showing a blank label.
        out = _plain(self._panel(_provenance_ctx(std_out="", std_err="")).render())
        assert "stdout" not in out and "stderr" not in out and "output" not in out

    def test_paths_pack_two_columns_when_wide(self) -> None:
        # On a wide card the paths pack TWO per row to use the horizontal space:
        # command|workdir on one line, stdout|stderr on the next. The left column
        # (command, stdout) aligns, the right column (workdir, stderr) aligns, and
        # the two columns are genuinely distinct (side by side).
        v = {
            "command": "/opt/aaa/cmd.sh",
            "workdir": "/opt/bbb",
            "stdout": "/opt/ccc/o.out",
            "stderr": "/opt/ddd/e.err",
        }
        ctx = _provenance_ctx(
            command=v["command"], work_dir=v["workdir"], std_out=v["stdout"], std_err=v["stderr"]
        )
        lines = _plain(self._panel(ctx).render()).splitlines()

        def col(value: str) -> int:
            return next(ln.index(value) for ln in lines if value in ln)

        # command pairs with workdir on one row; stdout pairs with stderr on the next.
        assert any(v["command"] in ln and v["workdir"] in ln for ln in lines)
        assert any(v["stdout"] in ln and v["stderr"] in ln for ln in lines)
        left = {col(v["command"]), col(v["stdout"])}
        right = {col(v["workdir"]), col(v["stderr"])}
        assert len(left) == 1 and len(right) == 1  # each column aligns
        assert left != right  # two distinct columns, not stacked

    def test_pressing_p_reflows_paths_to_a_single_full_width_column(self) -> None:
        # Expanding (p) drops the two-column packing so each full untruncated path
        # gets the whole width — command / workdir / stdout / stderr each own a row.
        v = {
            "command": "/opt/aaa/cmd.sh",
            "workdir": "/opt/bbb",
            "stdout": "/opt/ccc/o.out",
            "stderr": "/opt/ddd/e.err",
        }
        panel = self._panel(
            _provenance_ctx(
                command=v["command"],
                work_dir=v["workdir"],
                std_out=v["stdout"],
                std_err=v["stderr"],
            )
        )
        panel.full_paths = True
        lines = _plain(panel.render()).splitlines()
        # No row carries two different path values side by side any more.
        assert not any(v["command"] in ln and v["workdir"] in ln for ln in lines)
        assert not any(v["stdout"] in ln and v["stderr"] in ln for ln in lines)
        # Each value still starts in the same (single) column.
        cols = {
            next(ln.index(val) for ln in lines if val in ln)
            for val in (v["command"], v["workdir"], v["stdout"], v["stderr"])
        }
        assert len(cols) == 1

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
        assert "…" in out and "for full paths" in out
        # Short paths that fit -> nothing truncated -> no hint (it'd be pointless).
        short = _plain(self._panel(_provenance_ctx(command="/a/b.sh", work_dir="/a")).render())
        assert "…" not in short and "full path" not in short and "press" not in short

    def test_expanded_paths_show_a_collapse_hint(self) -> None:
        panel = self._panel(_provenance_ctx(command="/a/b.sh", work_dir="/a"))
        panel.full_paths = True
        assert "to collapse" in _plain(panel.render())

    def test_command_with_args_is_not_path_shortened(self) -> None:
        # A full command line (has spaces/args) is NOT treated as a path to elide,
        # so its arguments are never mangled into fake "/…/" directories. In the
        # wide compact view a long one is cut from the RIGHT (start + args kept in
        # order); expanded with p it shows the whole command line.
        cmd = "python train.py --data /very/long/unused"
        ctx = _provenance_ctx(command=cmd)
        out = _render_markup(self._panel(ctx).render()).plain
        assert "python train.py --data" in out  # start preserved, args in order
        assert "/very/…/unused" not in out  # never middle-elided like a path
        panel = self._panel(ctx)
        panel.full_paths = True
        assert cmd in _plain(panel.render())  # expanded: the whole command line


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

    def test_row_marker_is_decorative_never_a_health_verdict(self) -> None:
        # Colour-is-decorative: the CPU row's marker is the CPU IDENTITY colour and
        # never a health grade. A would-be "underused" CPU (well under the bar)
        # shows the same cyan dot as a busy one — no amber/red, no "underused" word.
        # The reader judges "well or not" from the visible "N / M cores" fact.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=16, usage_ns=0, usage_percent=30.0, effective_cores=4.8
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig(cpu_underuse_threshold=0.5)  # ratio 0.3 < bar
        cpu_block = next(b for b in r.render().split("\n\n") if "CPU" in b)
        assert f"[{_CPU_COLOR}]●[/]" in cpu_block  # decorative identity marker
        assert _HEALTH_COLOR["warn"] not in cpu_block  # no amber health verdict
        assert _HEALTH_COLOR["crit"] not in cpu_block
        assert "underused" not in _render_markup(cpu_block).plain  # never a word
        assert "4.8 / 16 cores" in _render_markup(cpu_block).plain  # the fact IS shown

    def test_cpu_row_shows_peak_cores(self) -> None:
        # The CPU row surfaces the lifetime peak cores (for right-sizing
        # --cpus-per-task), alongside the current usage, like the memory peak.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=16,
            usage_ns=0,
            usage_percent=30.0,
            effective_cores=4.8,
            peak_effective_cores=11.2,
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        plain = _render_markup(next(b for b in r.render().split("\n\n") if "CPU" in b)).plain
        assert "4.8 / 16 cores" in plain  # current
        assert "peak 11.2" in plain  # lifetime high-water mark

    def test_cpu_row_hides_peak_when_none_yet(self) -> None:
        # No peak observed yet (0) → no misleading "peak 0"; the suffix is dropped,
        # exactly like the memory peak on a remote/blank snapshot.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.cpu = CpuMetrics(
            cores_allocated=16, usage_ns=0, usage_percent=0.0, effective_cores=0.0
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        plain = _render_markup(next(b for b in r.render().split("\n\n") if "CPU" in b)).plain
        assert "peak" not in plain


class TestMarkupValidity:
    """Every panel must emit valid Rich markup in every state Textual renders."""

    def test_resource_rows_all_mem_states(self) -> None:
        for warn, crit in [(False, False), (True, False), (True, True)]:
            snap = _make_snapshot()
            snap.memory.oom_guard_warning = warn
            snap.memory.oom_guard_critical = crit
            w = ResourceRows()
            w.snapshot = snap
            w.config = SlurmwatchConfig()
            _valid_markup(w.render())

    def test_throttling_is_not_surfaced_as_a_word(self) -> None:
        # Throttling is never shown as a word (jargon): a throttling but still-running
        # GPU reads as plain "active", in its decorative device hue, with no scary
        # recolour. Cool temp so the hot-temp colour can't stand in for a verdict one.
        snap = _make_snapshot()
        snap.gpus[0].utilization_percent = 90.0  # active
        snap.gpus[0].process_utilization_percent = 90.0
        snap.gpus[0].throttling = True
        snap.gpus[0].temperature_celsius = 60.0
        r = ResourceRows()
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        block = next(b for b in r.render().split("\n\n") if "compute" in b)  # the device block
        assert _HEALTH_COLOR["warn"] not in block  # not recoloured by throttle
        plain = _render_markup(block).plain
        assert "throttling" not in plain  # the word is gone entirely
        assert "active" in plain  # it reads as a plain, still-running device
        assert _gpu_health(snap.gpus[0], 5.0) == ("ok", "active")

    def test_hot_temp_marker_has_negative_control(self) -> None:
        # The '⚠' hot-temperature marker (matching the GPU table) appears at/above
        # the threshold and disappears below it, so a stray marker can't pass it.
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].throttling = False
        snap.gpus[0].temperature_celsius = 88.0  # hot -> '⚠' marker
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        # The marker trails the WHOLE temperature group (Celsius + its °F reading).
        assert "88°C (190°F) ⚠" in r.render()

        snap.gpus[0].temperature_celsius = 60.0
        r.snapshot = snap
        cool = r.render()
        assert "⚠" not in cool
        assert "60°C" in cool

    def test_hot_temp_marker_is_ascii_in_ascii_mode(self) -> None:
        r = ResourceRows()
        snap = _make_snapshot()
        snap.gpus[0].temperature_celsius = 88.0
        r.snapshot = snap
        cfg = SlurmwatchConfig()
        cfg.ascii_mode = True
        r.config = cfg
        out = r.render()
        assert "88C (190F) !" in out and "⚠" not in out and "·" not in out
        assert "°" not in out  # the degree sign goes too, for BOTH units
        # The GPU device block builds its OWN marker + bar glyphs (not via _head),
        # so assert ascii purity: no Unicode bullet or bar cells leak into --ascii.
        assert "●" not in out and "█" not in out and "░" not in out


# ---------------------------------------------------------------------------
# Integration (Textual Pilot)
# ---------------------------------------------------------------------------


class _StubCollector:
    def __init__(self, raise_once: bool = False) -> None:
        self.config = SlurmwatchConfig()
        # Mirror TelemetryCollector.is_mock: False so the dashboard poll loop takes
        # the real/remote branch for a non-local node (these tests stub open_stream),
        # not the demo synth branch. (Accessing a missing .is_mock here otherwise
        # AttributeError'd inside the loop's except -> a 99% CPU busy-loop hang.)
        self.is_mock = False
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


def _svg_text(svg: str) -> str:
    """Just the rendered glyphs from a Textual SVG screenshot."""
    import xml.etree.ElementTree as ET

    return "".join(
        "".join(el.itertext()) for el in ET.fromstring(svg).iter() if el.tag.endswith("text")
    )


class TestAsciiModeCoversTheFramesNotJustTheStrings:
    """`--ascii` promises ASCII-only characters, and the FRAMES are Textual CSS.

    Measured in a pty against a real job: **3954 box-drawing characters** still went
    to a terminal that had explicitly asked for none — `border: round` / `heavy` is
    not a string this module formats, so no amount of separator gating reached it.
    Plus one `awaiting telemetry…`, which is subtler: the placeholder reads
    `self.config or SlurmwatchConfig()`, that fallback's ascii_mode is False, and the
    widget's config was only injected when the FIRST SNAPSHOT arrived — so the one
    window the placeholder exists for was the one window --ascii did not cover.
    """

    @staticmethod
    def _glyphs(svg: str) -> str:
        """The rendered glyphs, with the export's own substitution undone.

        Textual's SVG writer emits U+00A0 where a space needs to survive SVG
        whitespace collapsing (see tui._NBSP), and it arrives as an entity — so the
        substitution has to be undone AFTER parsing, not on the raw markup. Getting
        that backwards is what made a pty capture the only trustworthy check.
        """
        return _svg_text(svg).replace("\u00a0", " ")

    @pytest.mark.asyncio
    async def test_no_unicode_reaches_the_screen_before_the_first_snapshot(self) -> None:
        """The placeholder window, which is where the ellipsis leaked."""
        collector = _StubCollector()
        collector.config = SlurmwatchConfig(ascii_mode=True)
        app = _dash_app(collector)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            rows = app.scr.query_one(ResourceRows)
            assert rows.snapshot is None, "this is the pre-telemetry state"
            assert rows.config is not None, "config must be injected at COMPOSE time"
            assert "..." in rows.render()
            assert "\u2026" not in rows.render()
            svg = app.export_screenshot()
        bad = sorted({c for c in self._glyphs(svg) if not c.isascii()})
        assert not bad, f"non-ascii on screen under --ascii: {bad}"

    @pytest.mark.asyncio
    async def test_no_unicode_reaches_the_screen_with_live_data(self) -> None:
        collector = _StubCollector()
        collector.config = SlurmwatchConfig(ascii_mode=True)
        app = _dash_app(collector)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            svg = app.export_screenshot()
        bad = sorted({c for c in self._glyphs(svg) if not c.isascii()})
        assert not bad, f"non-ascii on screen under --ascii: {bad}"

    @pytest.mark.asyncio
    async def test_the_default_mode_keeps_its_unicode_frame(self) -> None:
        """The complement: asciifying unconditionally would flatten the real UI."""
        app = _dash_app(_StubCollector())
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            svg = app.export_screenshot()
        assert any(c in svg for c in "\u256d\u2570\u2500"), "the rounded frame is gone"

    def test_asciify_keeps_every_colour_and_only_changes_the_glyphs(self) -> None:
        """A restyle would be a regression: only the border TYPE may move."""
        from textual.color import Color

        from slurmwatch.tui import _asciify_borders

        class _Styles:
            def __init__(self, edge: tuple[str, Color] | None) -> None:
                self.border_top = edge
                self.border = edge

        class _W:
            def __init__(self, edge: tuple[str, Color] | None) -> None:
                self.styles = _Styles(edge)

            def query(self, _sel: str) -> list[object]:
                return []

        red = Color.parse("red")
        for given, expected in [
            (("round", red), ("ascii", red)),
            (("heavy", red), ("ascii", red)),
            (("ascii", red), ("ascii", red)),  # idempotent
            (("none", red), ("none", red)),  # left alone
            (None, None),
        ]:
            w = _W(given)
            _asciify_borders(w)
            assert w.styles.border == expected, given

    @pytest.mark.asyncio
    async def test_every_screen_that_draws_a_frame_asciifies_it(self) -> None:
        """One call site per screen, so one test per screen — or four are half-fixed.

        The wiring is five separate `on_mount` calls reading five different config
        attributes (`self.config`, `self._config`, `self._dashboard.config`). A sweep
        that only reverts the dashboard's proves nothing about the other four, which
        is the shape of half-fix this exercise keeps finding.
        """
        from slurmwatch.pending import PendingJob
        from slurmwatch.tui import (
            ForeignJobScreen,
            JobSelectorScreen,
            PendingScreen,
            ResourceDetailScreen,
        )

        cfg = SlurmwatchConfig(ascii_mode=True)
        pending = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="p",
            qos="",
            account="",
            reason="Priority",
            submit_time=None,
            start_time_estimate=None,
            priority=1,
            req_cpus=4,
            req_nodes=1,
            req_mem_bytes=8 * 1024**3,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        foreign_ctx = JobContext(
            job_id="7",
            username="other",
            partition="p",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn001"],
            job_state="RUNNING",
            job_start_time=time.time() - 60,
            time_limit_seconds=3600,
            remote=True,
        )

        collector = _StubCollector()
        collector.config = cfg
        app = _dash_app(collector)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            for label, screen in (
                ("ResourceDetailScreen", ResourceDetailScreen(app.scr, "cpu")),
                ("PendingScreen", PendingScreen(pending, cfg)),
                ("ForeignJobScreen", ForeignJobScreen(foreign_ctx, cfg)),
                ("JobSelectorScreen", JobSelectorScreen([], config=cfg)),
            ):
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()
                bad = sorted({c for c in self._glyphs(app.export_screenshot()) if not c.isascii()})
                assert not bad, f"{label} leaks {bad} under --ascii"
                app.pop_screen()
                await pilot.pause()

    @pytest.mark.asyncio
    async def test_the_ascii_figure_still_carries_the_number(self) -> None:
        """Substituting the widget is only half of it — it has to be FILLED.

        `Digits.update()` and `Static.update()` are different call paths, so the
        ascii branch can render an empty hero and every purity assertion above still
        passes: an empty figure is impeccably ASCII.
        """
        from textual.widgets import Digits

        from slurmwatch.tui import ResourceDetailScreen

        collector = _StubCollector()
        collector.config = SlurmwatchConfig(ascii_mode=True)
        app = _dash_app(collector)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            detail = ResourceDetailScreen(app.scr, "cpu")
            await app.push_screen(detail)
            await pilot.pause()
            await pilot.pause()
            figure = detail.query_one("#detail-figure")
            assert not isinstance(figure, Digits), "ascii mode must not use Digits"
            shown = str(figure.render())
            assert "%" in shown and any(c.isdigit() for c in shown), shown

    def test_the_scrollbar_renderer_is_restored_in_default_mode(self) -> None:
        """It lives on a CLASS, so a one-way swap leaks into the next run.

        Whichever mode a screen mounts in has to leave the renderer describing THAT
        mode — otherwise the first --ascii screen in a process silently ascii-fies
        every later default-mode one (and every subsequent test in this session).
        """
        from textual.scrollbar import ScrollBar, ScrollBarRender

        from slurmwatch.tui import _apply_ascii_chrome, _AsciiScrollBarRender

        class _Styles:
            border_top = None
            border = None

        class _Bare:
            styles = _Styles()

            def query(self, _sel: str) -> list[object]:
                return []

        try:
            _apply_ascii_chrome(_Bare(), True)
            assert ScrollBar.renderer is _AsciiScrollBarRender
            _apply_ascii_chrome(_Bare(), False)
            assert ScrollBar.renderer is ScrollBarRender
        finally:
            ScrollBar.renderer = ScrollBarRender

    def test_the_ascii_scrollbar_glyphs_are_ascii(self) -> None:
        from slurmwatch.tui import _AsciiScrollBarRender

        for bars in (_AsciiScrollBarRender.VERTICAL_BARS, _AsciiScrollBarRender.HORIZONTAL_BARS):
            assert len(bars) == 8, "Textual indexes these by eighths"
            assert all(len(b) == 1 and b.isascii() for b in bars), bars

    def test_the_selectors_animated_border_uses_the_ascii_type(self) -> None:
        """It re-applies the border every frame, so a one-shot DOM pass is undone."""
        from slurmwatch.tui import JobSelectorScreen

        for ascii_mode, expected in ((True, "ascii"), (False, "heavy")):
            scr = JobSelectorScreen.__new__(JobSelectorScreen)
            scr._config = SlurmwatchConfig(ascii_mode=ascii_mode)
            assert scr._border("#ffffff")[0] == expected


class TestDashboardIntegration:
    @pytest.mark.asyncio
    async def test_renders_snapshot_and_header(self) -> None:
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            assert app.scr.query_one(ResourceRows).snapshot is not None
            assert app.scr.latest_snapshot is not None
            assert "12345" in str(app.scr.sub_title)

    @pytest.mark.asyncio
    async def test_header_leads_with_the_job_name(self) -> None:
        # The header is orientation, so it names the experiment. The id is NOT lost —
        # it's in the always-visible bottom bar and heads the JOB card — so a third
        # copy in the most prominent line would add nothing.
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr.job_ctx.job_name = "argonne35-pretrain"
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            sub = str(app.scr.sub_title)
            assert "argonne35-pretrain" in sub
            assert "12345" not in sub
            assert "ada" in sub  # who still rides along

    @pytest.mark.asyncio
    async def test_header_adds_the_task_index_for_an_array_task(self) -> None:
        # Every task of an array shares ONE name, so the name alone can't tell task 1
        # from task 7 — the index has to come along, from job_ctx (never the snapshot).
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr.job_ctx.job_name = "sweep"
            app.scr.job_ctx.job_id = "52330903_7"
            app.scr.job_ctx.array_task_id = "7"
            snap = _make_snapshot()
            snap.job_id = "52330910"  # the raw numeric id, which must not surface
            app.scr._update_widgets(snap)
            await pilot.pause()
            sub = str(app.scr.sub_title)
            assert "sweep" in sub and "task 7" in sub
            assert "52330910" not in sub

    @pytest.mark.asyncio
    async def test_header_renders_a_bracket_in_the_name_literally(self) -> None:
        # Textual assembles the header with Content(...), which is LITERAL text — so a
        # name from `sbatch -J 'exp[1]'` must appear as itself, with no MarkupError and
        # no stray backslash from over-escaping.
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr.job_ctx.job_name = "exp[1]-a35"
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            header_txt = ""
            for w in app.scr.walk_children():
                if type(w).__name__ == "HeaderTitle":
                    header_txt = str(w.render())  # type: ignore[attr-defined]
            assert "exp[1]-a35" in header_txt
            assert "\\[" not in header_txt

    @pytest.mark.asyncio
    async def test_header_uses_selected_job_id_not_raw_snapshot_id(self) -> None:
        # The no-name fallback (Slurm reported no JobName): the header reverts to the
        # id, and for an array task the user selects "52330903_1" (job_ctx.job_id)
        # while the collector's snapshot carries the raw numeric JobId ("52330910") —
        # it must show the id the user knows, matching the JOB card.
        app = _dash_app(_StubCollector())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr.job_ctx.job_id = "52330903_1"
            snap = _make_snapshot()
            snap.job_id = "52330910"
            app.scr._update_widgets(snap)
            await pilot.pause()
            assert "52330903_1" in str(app.scr.sub_title)
            assert "52330910" not in str(app.scr.sub_title)

    @pytest.mark.asyncio
    async def test_dashboard_header_is_ascii_under_ascii_mode(self) -> None:
        # Textual's HeaderTitle joins title/sub_title with a Unicode em-dash we
        # can't gate; under --ascii the brand is folded into the title (sub_title
        # empty) so the rendered header stays ASCII-clean.
        coll = _StubCollector()
        coll.config = SlurmwatchConfig(ascii_mode=True)
        app = _dash_app(coll)
        header_txt = ""
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            for w in app.scr.walk_children():
                if type(w).__name__ == "HeaderTitle":
                    header_txt = str(w.render())  # type: ignore[attr-defined]
        assert "slurmwatch" in header_txt
        assert header_txt.isascii(), f"non-ascii in dashboard header under --ascii: {header_txt!r}"

    @pytest.mark.asyncio
    async def test_card_border_titles_are_ascii_under_ascii_mode(self) -> None:
        # A border_title is rendered text like any other, but two of the three card
        # titles hard-coded the Unicode "·" while the third (ForeignJobScreen) gated it
        # via _sep. No test looked at border_title, which is how the leak survived.
        coll = _StubCollector()
        coll.config = SlurmwatchConfig(ascii_mode=True)
        app = _dash_app(coll)
        titles: list[str] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            app.scr._update_widgets(_make_snapshot())
            await pilot.pause()
            titles = [
                str(getattr(w, "border_title", None))
                for w in app.scr.walk_children()
                if getattr(w, "border_title", None) is not None
            ]
        assert titles, "expected at least one card border title"
        for t in titles:
            assert t.isascii(), f"non-ascii in a card border title under --ascii: {t!r}"

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
    async def test_three_or_more_gpus_render_inline_blocks(self) -> None:
        # 3+ GPUs render inline as spacious per-device blocks in ResourceRows (a
        # compute bar over a vram bar each), not a table — every device carries both
        # gauges, so the count of "compute"/"VRAM" labels equals the device count.
        app = _dash_app(_StubCollector(), gpus=4)
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
            snap.gpu_count_requested = 4
            app.scr._update_widgets(snap)
            await pilot.pause()
            out = _render_markup(app.scr.query_one(ResourceRows).render()).plain
            assert out.count("compute") == 4 and out.count("VRAM") == 4

    @pytest.mark.asyncio
    async def test_gpu_blocks_stack_compute_over_vram(self) -> None:
        # Each device is a two-line block: a compute bar immediately followed by a
        # vram bar, both labeled with their own %, their labels aligned in one
        # column — so VRAM reads as the same kind of gauge as compute, not bare text.
        app = _dash_app(_StubCollector(), gpus=4)
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
            snap.gpu_count_requested = 4
            app.scr._update_widgets(snap)
            await pilot.pause()
            lines = _render_markup(app.scr.query_one(ResourceRows).render()).plain.splitlines()
            pairs = 0
            for i, ln in enumerate(lines):
                if "compute" in ln:
                    nxt = lines[i + 1]
                    assert "VRAM" in nxt  # vram bar directly below its compute bar
                    assert ln.index("compute") == nxt.index("VRAM")  # aligned column
                    pairs += 1
            assert pairs == 4

    @pytest.mark.asyncio
    async def test_gpu_drillin_shows_interconnect_block(self) -> None:
        # Pressing g on a multi-GPU job opens the GPU drill-in, whose body leads
        # with the interconnect summary + topology grid before the per-device charts.
        app = _dash_app(_StubCollector(), gpus=4)
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(4)]
            snap.gpu_count_requested = 4
            snap.interconnect = _nvlink_ic(4)
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            from textual.widgets import Static

            detail = app.screen  # the drill-in is now the active screen
            assert isinstance(detail, ResourceDetailScreen)
            # The interconnect block is prepended to the chart area (above the
            # per-device history). str() of the Static's content is the plain text;
            # the plain substrings below appear whether or not markup wraps them.
            chart = str(detail.query_one("#detail-chart", Static).render())
            assert "NVLink 3" in chart  # the summary
            assert "CUDA0" in chart and "NV12" in chart  # the topology grid
            assert "GB/s" in chart  # live fabric transfer
            assert "per-device history" in chart  # the divider before the charts

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
    async def test_two_gpus_render_inline_blocks(self) -> None:
        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(2)]
            snap.gpu_count_requested = 2
            app.scr._update_widgets(snap)
            await pilot.pause()
            out = _render_markup(app.scr.query_one(ResourceRows).render()).plain
            assert out.count("VRAM") == 2  # both devices carry a vram bar (no table)

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
    async def test_gpu_detail_shows_job_share_line(self) -> None:
        # F6: drilling into GPU shows THIS job's per-device share (compute % + vram
        # GiB), which the dashboard's device blocks can't — so the keystroke pays
        # off. It's a one-line "this job" header above each device's charts.
        from slurmwatch.tui import _GPU_VRAM_BAR

        app = _dash_app(_StubCollector(), gpus=2)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            # _make_gpu(util, procmem→JOB VRAM, memused, memtot). Override the job's
            # compute share to a value DISTINCT from the device-wide util, so the
            # share line's compute figure is provably this job's share (40%), not the
            # device-wide 90% — the whole point of the line on a shared GPU.
            snap.gpus = [
                _make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, 80 * 1024**3, index=i) for i in range(2)
            ]
            for g in snap.gpus:
                g.process_utilization_percent = 40.0  # this job's share < device 90%
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            content: Any = scr.query_one("#detail-chart").render()  # a textual Content
            chart = content.plain
            assert chart.count("this job") == 2  # one share line per device
            assert "40% compute" in chart  # this job's share, NOT the device-wide 90%
            assert "90% compute" not in chart  # device-wide compute does not appear here
            assert "50.0 GiB VRAM" in chart  # this job's vram share, in GiB
            # The share line is coloured BY METRIC too: compute share violet, vram
            # share teal (a swap would be invisible to the .plain checks above).
            share_at = chart.find("this job")
            eol = chart.find("\n", share_at)
            share_styles = {
                str(sp.style) for sp in content.spans if sp.start < eol and sp.end > share_at
            }
            assert _GPU_COLOR in share_styles and _GPU_VRAM_BAR in share_styles

    @pytest.mark.asyncio
    async def test_gpu_detail_share_line_em_dash_when_no_per_process_vram(self) -> None:
        # When NVML has no per-process vram figure (process_memory_bytes == 0 — the
        # common MIG / MPS / non-attributable case), the share line shows an em-dash
        # for vram, not a misleading "0.0 GiB".
        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(80.0, 0, 30 * 1024**3, 40 * 1024**3, index=0)]  # procmem=0
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            chart = _render_markup(str(scr.query_one("#detail-chart").render())).plain
            assert "— VRAM" in chart  # em-dash, the "no per-process figure" fallback
            assert "0.0 GiB VRAM" not in chart  # never a misleading zero

    @pytest.mark.asyncio
    async def test_cpu_drill_in_shows_peak_cores(self) -> None:
        # The CPU drill-in headline includes the lifetime peak cores (right-sizing
        # --cpus-per-task) beside the current "cores busy".
        app = _dash_app(_StubCollector(), gpus=0)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.cpu = CpuMetrics(
                cores_allocated=16,
                usage_ns=0,
                usage_percent=40.0,
                effective_cores=6.4,
                peak_effective_cores=12.0,
            )
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            headline = _render_markup(str(scr.query_one("#detail-headline").render())).plain
            assert "peak 12 cores" in headline  # _fmt_cores drops the .0

    @pytest.mark.asyncio
    async def test_cpu_drill_in_hides_peak_when_none_yet(self) -> None:
        # No peak observed (0, e.g. a remote estimate) → the drill-in headline omits
        # the peak line rather than showing a misleading "peak 0 cores".
        app = _dash_app(_StubCollector(), gpus=0)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.cpu = CpuMetrics(
                cores_allocated=16, usage_ns=0, usage_percent=40.0, effective_cores=6.4
            )  # peak_effective_cores defaults to 0.0
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            headline = _render_markup(str(scr.query_one("#detail-headline").render())).plain
            assert "peak" not in headline

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
    async def test_gpu_detail_down_arrow_reaches_devices_below_the_fold(self) -> None:
        # Many GPUs on a short terminal overflow vertically; the scroll box is
        # focused (no inner table now), so ↑/↓ scroll it and every device stays
        # reachable by keyboard.
        from textual.containers import VerticalScroll

        app = _dash_app(_StubCollector(), gpus=8)
        async with app.run_test(size=(120, 16)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(8)]
            app.scr._update_widgets(snap)
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            box = scr.query_one("#detail-box", VerticalScroll)
            assert box.has_focus  # the box owns the arrows now (no inner table)
            assert box.max_scroll_y > 0  # content taller than the box
            before = box.scroll_y
            for _ in range(10):
                await pilot.press("down")
            await pilot.pause()
            assert box.scroll_y > before  # scrolled down to reach lower devices
            # Every device is charted — not just the ones initially in view. Guards
            # against a cap (e.g. gpus[:N]) that would chart only the first few: the
            # LAST device (GPU 7) must have both its compute and vram graphs drawn.
            scr._refresh()
            await pilot.pause()
            chart = _render_markup(str(scr.query_one("#detail-chart").render())).plain
            assert "CUDA 7 compute" in chart and "CUDA 7 VRAM" in chart

    @pytest.mark.asyncio
    async def test_gpu_detail_charts_every_device_with_stacked_graphs(self) -> None:
        # Multi-GPU drill-in: EVERY device gets its own big compute + vram area
        # charts (the same broad layout the single-GPU drill-in uses), because a
        # one-row inline sparkline is too small to read a trend from on a multi-GPU
        # job. No table (the dashboard already shows each device's current numbers).
        from collections import deque

        app = _dash_app(_StubCollector(), gpus=3)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(90.0, 50 * 1024**3, 55 * 1024**3, index=i) for i in range(3)]
            app.scr._update_widgets(snap)
            # A distinct, KNOWN compute AND vram history per device, so each chart's
            # stats are deterministic and provably its OWN series (not one shared /
            # averaged line, and no compute/vram swap). Device i: compute avg 20+10i,
            # vram avg 15+10i — every value distinct across devices and metrics.
            rows = app.scr.query_one(ResourceRows)
            for i in range(3):
                base = i * 10
                rows.gpu_history[i] = deque([float(base), float(base + 20), float(base + 40)])
                rows.gpu_vram_history[i] = deque(
                    [float(base + 5), float(base + 15), float(base + 25)]
                )
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            # There is no drill-in table any more (it duplicated the dashboard) —
            # only the per-device charts remain.
            with pytest.raises(NoMatches):
                scr.query_one("#detail-table")
            # The big chart is NOT empty for a multi-GPU job (the old bug): it draws
            # a compute AND a vram graph for each of the three devices, in order.
            chart = _render_markup(str(scr.query_one("#detail-chart").render())).plain
            assert chart.strip()
            positions = [(i, chart.find(f"CUDA {i} compute")) for i in range(3)]
            assert all(p >= 0 for _, p in positions)  # every device is charted
            # Devices appear in order (CUDA 0, CUDA 1, CUDA 2 top to bottom).
            assert [p for _, p in positions] == sorted(p for _, p in positions)
            for idx, (i, start) in enumerate(positions):
                end = positions[idx + 1][1] if idx + 1 < len(positions) else len(chart)
                section = chart[start:end]
                compute_part, sep, vram_part = section.partition(f"CUDA {i} VRAM")
                assert sep  # this device has BOTH a compute and a vram graph
                # The compute chart's stats are its own % series (avg 20+10i); the
                # vram chart's stats are GiB (a 40 GiB card: avg% × 0.4), so the two
                # series can't be confused even though both graphs read as a fill.
                c_avg = 20 + 10 * i
                v_avg_gib = (15 + 10 * i) * 40 / 100  # % → GiB on the 40 GiB card
                assert f"avg {c_avg:>3.0f}%" in compute_part
                assert f"avg {v_avg_gib:>3.0f}   " in vram_part  # GiB, not %
                assert "GiB" in vram_part and "GiB" not in compute_part
                # ...and neither series' stats leak into the other graph.
                assert f"avg {c_avg:>3.0f}%" not in vram_part
            # Colour BY METRIC: every device's compute graph is drawn in the GPU
            # violet and its vram graph in the distinct teal, never swapped. The
            # .plain checks above can't see this (swapping the two colour constants
            # would pass them silently), so inspect the rendered style spans — the
            # user has repeatedly caught colour regressions a text-only check missed.
            from slurmwatch.tui import _GPU_VRAM_BAR

            content: Any = scr.query_one("#detail-chart").render()  # a textual Content
            plain = content.plain

            def styles_between(lo: int, hi: int) -> set[str]:
                # Colours applied to any span overlapping the [lo, hi) char range.
                return {str(sp.style) for sp in content.spans if sp.start < hi and sp.end > lo}

            for i in range(3):
                # Analyse the CHART regions only: the compute graph runs from its
                # label to the vram label; the vram graph from its label to the START
                # of the NEXT device's share line (which carries both metric colours
                # — its "● GPU N" is violet — and would otherwise pollute this
                # device's vram region).
                c_at = plain.find(f"CUDA {i} compute")
                v_at = plain.find(f"CUDA {i} VRAM")
                nxt_share = plain.find("this job", v_at)
                vram_end = plain.rfind("\n", 0, nxt_share) + 1 if nxt_share >= 0 else len(plain)
                compute_styles = styles_between(c_at, v_at)
                vram_styles = styles_between(v_at, vram_end)
                assert _GPU_COLOR in compute_styles  # compute graph → violet
                assert _GPU_VRAM_BAR not in compute_styles  # never the teal
                assert _GPU_VRAM_BAR in vram_styles  # vram graph → teal
                assert _GPU_COLOR not in vram_styles  # never the violet

    @pytest.mark.asyncio
    async def test_history_skips_unreadable_compute_sample(self) -> None:
        # A2: when NVML can't read device util the collector emits 0.0 with
        # utilization_available=False. Recording that would drag the drill-in
        # chart's min/avg to a FALSE zero (permanently, for a MIG slice), so the
        # compute sample is skipped — while VRAM, still readable, keeps recording.
        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            rows = app.scr.query_one(ResourceRows)

            good = _make_snapshot()
            good.gpus = [_make_gpu(80.0, 18 * 1024**3, 20 * 1024**3, index=0)]
            app.scr._update_widgets(good)
            assert list(rows.gpu_history[0]) == [80.0]
            assert list(rows.gpu_vram_history[0]) == [50.0]

            blind = _make_snapshot()
            g = _make_gpu(0.0, 18 * 1024**3, 30 * 1024**3, index=0)
            g.utilization_available = False
            blind.gpus = [g]
            app.scr._update_widgets(blind)
            # No false 0.0 appended — the compute series is untouched...
            assert list(rows.gpu_history[0]) == [80.0]
            # ...while VRAM (readable) still advanced to 30/40 GiB = 75%.
            assert list(rows.gpu_vram_history[0]) == [50.0, 75.0]

            # A genuinely idle device (util readable, 0%) IS recorded — the skip
            # must not swallow a real zero.
            idle = _make_snapshot()
            idle.gpus = [_make_gpu(0.0, 18 * 1024**3, 20 * 1024**3, index=0)]
            app.scr._update_widgets(idle)
            assert list(rows.gpu_history[0]) == [80.0, 0.0]

    @pytest.mark.asyncio
    async def test_history_skips_unreadable_vram_sample(self) -> None:
        # The VRAM twin: a failed nvmlDeviceGetMemoryInfo must not drag the drill-in
        # chart's min/avg to a false 0 either — while compute, still readable, keeps
        # recording.
        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            rows = app.scr.query_one(ResourceRows)

            good = _make_snapshot()
            good.gpus = [_make_gpu(80.0, 18 * 1024**3, 20 * 1024**3, index=0)]
            app.scr._update_widgets(good)
            assert list(rows.gpu_history[0]) == [80.0]
            assert list(rows.gpu_vram_history[0]) == [50.0]

            blind = _make_snapshot()
            g = _make_gpu(30.0, 0, 0, index=0)
            g.memory_available = False
            blind.gpus = [g]
            app.scr._update_widgets(blind)
            # No false 0.0 appended — the vram series is untouched...
            assert list(rows.gpu_vram_history[0]) == [50.0]
            # ...while compute (readable) still advanced.
            assert list(rows.gpu_history[0]) == [80.0, 30.0]

            # A genuine 0% VRAM reading (memory readable, nothing used) IS recorded.
            empty = _make_snapshot()
            empty.gpus = [_make_gpu(30.0, 0, 0, index=0)]
            app.scr._update_widgets(empty)
            assert list(rows.gpu_vram_history[0]) == [50.0, 0.0]

    @pytest.mark.asyncio
    async def test_gpu_detail_single_device_shows_compute_and_vram_charts(self) -> None:
        # A single-GPU job gets TWO tall filled history graphs — compute AND vram —
        # where the lone inline sparkline would leave the panel mostly empty. Both
        # series are drawn, each labelled with its own summary stats, so a device
        # that's compute-idle yet holding memory (or the reverse) is visible.
        from collections import deque

        app = _dash_app(_StubCollector(), gpus=1)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [_make_gpu(72.5, 18 * 1024**3, 20 * 1024**3, index=0)]
            app.scr._update_widgets(snap)
            # _update_widgets itself must populate BOTH per-device histories from
            # the snapshot (not just compute) — assert that before overwriting, so a
            # regression that stops appending vram history is caught (otherwise the
            # drill-in vram chart would silently draw empty in production).
            rows = app.scr.query_one(ResourceRows)
            assert rows.gpu_history[0][-1] == 72.5  # compute util appended
            assert rows.gpu_vram_history[0][-1] == 50.0  # 20/40 GiB, vram fill appended
            # Now known, DISTINCT compute/vram histories so each chart's stats are
            # deterministic AND provably its own series (not the same one twice).
            rows.gpu_history[0] = deque([10.0, 50.0, 90.0])
            rows.gpu_vram_history[0] = deque([20.0, 40.0, 60.0])
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            chart = _render_markup(str(scr.query_one("#detail-chart").render())).plain
            assert chart.strip()  # the graph is drawn
            # A "this job" share line heads the device, then both series, each
            # labelled...
            assert "this job" in chart
            assert "18.0 GiB VRAM" in chart  # this job's vram share (procmem = 18 GiB)
            assert "CUDA 0 compute" in chart and "CUDA 0 VRAM" in chart
            # ...and — crucially — each label is PAIRED with its OWN series' stats,
            # not just "both stat-sets appear somewhere" (which a label/series swap
            # would still satisfy). Split at the vram label: compute's 10/50/90 read
            # as %, vram's 20/40/60% read as GiB (a 40 GiB card → 8/16/24 GiB).
            compute_part, _, vram_part = chart.partition("CUDA 0 VRAM")
            assert "min  10%" in compute_part and "avg  50%" in compute_part
            assert "max  90%" in compute_part
            assert "avg  16" in vram_part and "now  24 GiB" in vram_part  # GiB, not %
            # And the compute stats must NOT leak into the vram section (no swap).
            assert "min  10%" not in vram_part
            # No drill-in table any more — the dashboard already shows the numbers.
            with pytest.raises(NoMatches):
                scr.query_one("#detail-table")

    @pytest.mark.asyncio
    async def test_gpu_status_is_a_word_on_the_dashboard(self) -> None:
        # "nobody knows what the triangle means" — each per-device block spells the
        # status out as a plain word (idle / active), not a bare glyph.
        app = _dash_app(_StubCollector(), gpus=3)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            snap = _make_snapshot()
            snap.gpus = [
                _make_gpu(0.0, 0, 40 * 1024**3, index=0),  # idle (0% compute, no VRAM)
                _make_gpu(95.0, 30 * 1024**3, 40 * 1024**3, index=1),  # active
                _make_gpu(95.0, 30 * 1024**3, 40 * 1024**3, index=2),  # active
            ]
            app.scr._update_widgets(snap)
            await pilot.pause()
            out = _render_markup(app.scr.query_one(ResourceRows).render()).plain.lower()
            assert "idle" in out  # the 0% device's block status (a word, not a glyph)
            # "active" also appears once in the "N active" header, so require the two
            # busy device blocks to contribute it too (header 1 + 2 blocks = 3).
            assert out.count("active") >= 3

    @pytest.mark.asyncio
    async def test_stream_backoff_never_overflows(self) -> None:
        # audit-3 #8: after ~1000 failures, 2.0 ** (fails-1) overflowed before the
        # min() cap; capping the exponent keeps it at the 8s ceiling.
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, DashboardScreen)
            scr._stream_fails = 5000
            scr._selected_node = "cn002"  # != node arg -> the sleep loop exits at once
            await scr._stream_backoff("cn001")  # must not raise OverflowError

    @pytest.mark.asyncio
    async def test_job_end_clears_typed_node_input_and_keeps_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # audit-3 #3: if the job ends while a "go to node N" prefix is half-typed
        # (ambiguous, so a 0.9s pause-timer is armed), the JOB ENDED notice must
        # stand — the pending timer must be cancelled, not fire ~0.9s later and
        # hide the notice.
        nodes = [f"cn{i:03d}" for i in range(1, 13)]  # 12 nodes -> "1" is ambiguous
        app = self._multinode_app(nodes)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, DashboardScreen)
            scr._show(_make_snapshot(), scr._local_node)
            await pilot.pause()
            await pilot.press("1")  # ambiguous digit: arms the pause timer + prompt
            await pilot.pause()
            assert scr._node_input == "1" and scr._node_input_timer is not None
            scr._show_job_ended()
            await pilot.pause()
            assert scr._node_input == ""  # buffer cleared
            assert scr._node_input_timer is None  # pending timer cancelled
            banner = scr.query_one(SwitchBanner)
            assert banner.ended is True and banner.display is True  # notice stands

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
    async def test_switch_banner_arms_automatically_when_initial_node_is_remote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The job-selector's auto-discovery path (unlike `sw <job_id>`, whose
        # hop/ssh/sstat ladder resolves a remote job BEFORE any dashboard ever
        # mounts) can land here with the job's first node already selected and
        # remote. Without arming the same banner/spinner/watchdog `_set_node`
        # uses, a stream that can't launch sits on a silent "awaiting
        # telemetry…" forever with no feedback that anything is even trying.
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001", "cn002"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            assert scr._selected_node != scr._local_node  # the scenario in question
            assert scr._switch_target == scr._selected_node
            banner = scr.query_one(SwitchBanner)
            assert banner.display is True
            assert banner.node == scr._selected_node

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
            # Past the stuck threshold, which is derived from the transport's own
            # launch budget — so this must outlast it.
            scr._switch_started = time.monotonic() - (tuimod._SWITCH_STUCK_S + 1)
            scr._tick_switch()
            await pilot.pause()
            banner = scr.query_one(SwitchBanner)
            assert banner.stuck is True
            assert banner.display is True  # a warning is still shown
            assert "switching" not in scr.query_one("#body").classes  # no longer dimmed
            assert scr._switch_target == "cn002"  # still trying in the background

    def test_the_stuck_watchdog_outlasts_the_transport_it_watches(self) -> None:
        # At a flat 12s the watchdog fired before a first frame was even possible:
        # the GPU-attachability probe alone may take _GPU_PROBE_SECONDS + 3, the
        # stream step another --immediate=_STREAM_CONNECT_TIMEOUT, and
        # `_read_remote` keeps waiting up to _STREAM_LAUNCH_TIMEOUT. A healthy
        # attach therefore raised "it may be busy or unreachable" and then cleared
        # it — worst on the mount-armed path, where the slow case (the job's own
        # step holding the GPU) is the common one.
        from slurmwatch import remote as remotemod

        launch_floor = (remotemod._GPU_PROBE_SECONDS + 3) + remotemod._STREAM_CONNECT_TIMEOUT
        assert launch_floor <= tuimod._STREAM_LAUNCH_TIMEOUT
        assert tuimod._SWITCH_STUCK_S > tuimod._STREAM_LAUNCH_TIMEOUT
        # ...and the reassuring "this can take a moment" note still comes early.
        assert launch_floor > tuimod._SWITCH_SLOW_S

    @pytest.mark.asyncio
    async def test_a_first_attach_at_mount_reads_as_connecting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The job-selector path can mount the dashboard with the job's node already
        # selected and remote. The banner is armed there so a stream that can't
        # launch escalates instead of sitting on "awaiting telemetry…" — but
        # nothing was switched, and on a single-node job it read "switching to node
        # 1 of 1".
        async def _no_stream(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr("slurmwatch.tui.open_stream", _no_stream)
        app = self._multinode_app(["cn001"])
        app.scr._local_node = "login01"  # the job's only node is remote from here
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scr = app.scr
            assert scr._switch_target == "cn001"  # the watchdog is armed
            banner = scr.query_one(SwitchBanner)
            assert banner.connecting is True and banner.multi is False
            out = _render_markup(banner.render()).plain
            assert "connecting to the compute node" in out and "cn001" in out
            assert "of 1" not in out and "switching" not in out

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
    async def test_single_job_still_shows_the_picker(self) -> None:
        # A lone job must still open the selector (consistent `sw`), not jump
        # straight into the dashboard.
        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        one: list[dict[str, object]] = [
            {"job_id": "42", "state": "R", "partition": "gpu", "name": "solo", "nodes": "1"}
        ]
        app = SlurmwatchApp(jobs=one)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_cursor_moves_while_a_slow_poll_is_in_flight(self) -> None:
        # Regression (S1 "can't move the cursor"): the live-refresh poll must run on
        # a worker, not inline on the selector's message pump — otherwise a slow
        # squeue parks the pump and freezes arrow keys until it returns (up to
        # SLURM_CMD_TIMEOUT on a busy controller).
        import threading

        from textual.widgets import ListView

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        jobs: list[dict[str, object]] = [
            {"job_id": str(i), "state": "R", "partition": "gpu", "name": "j", "nodes": "1"}
            for i in range(5)
        ]
        release = threading.Event()

        def slow_refresh() -> list[dict[str, object]]:
            release.wait(timeout=10)  # block the executor thread until the test frees it
            return jobs

        app = SlurmwatchApp(jobs=jobs, config=SlurmwatchConfig(), refresh=slow_refresh)
        moved = -1
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, JobSelectorScreen)
                lv = screen.query_one(ListView)
                lv.index = 0
                lv.focus()
                screen._kick_poll()  # start the poll worker; it blocks inside slow_refresh
                await pilot.pause(0.2)  # let the worker reach the executor block
                await pilot.press("down")
                await pilot.press("down")
                await pilot.pause(0.05)
                moved = lv.index  # must have advanced even though the poll is blocked
                release.set()  # unblock so the app tears down cleanly
                await pilot.pause()
        finally:
            release.set()  # safety net so a blocked worker never hangs teardown
        assert moved == 2, f"cursor frozen while a slow poll was in flight (index={moved})"

    def test_column_widths_no_crash_when_job_list_empties(self) -> None:
        # Regression: `max(len(head), *())` crashed the whole TUI when every job
        # finished while the picker was open (empty live refresh -> empty self.jobs).
        from slurmwatch.tui import JobSelectorScreen

        screen = JobSelectorScreen(jobs=[])
        widths = screen._column_widths()
        assert widths == [len(h) for h in screen._headings()]
        assert all(w > 0 for w in widths)

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_selector_hint_is_ascii_under_ascii_mode(self) -> None:
        # --ascii must not leak Unicode: the selector used to hardcode the ↑/↓/·
        # glyphs (it never received config), so they showed as mojibake on a
        # non-UTF-8 terminal. It now threads config and swaps them.
        from textual.widgets import Static

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        app = SlurmwatchApp(jobs=self.JOBS, config=SlurmwatchConfig(ascii_mode=True))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            hint = str(app.screen.query_one("#selector-hint", Static).render())
        assert hint.isascii(), f"non-ascii in selector hint under --ascii: {hint!r}"

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_quit_returns_to_selector_then_escape_exits(self) -> None:
        # With multiple jobs, quitting a job's dashboard returns to the selector
        # (not a full exit) so the user can pick another job — and the cursor lands
        # back on the job they'd opened, not the top. Escaping the selector exits.
        from textual.widgets import ListView

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        app = SlurmwatchApp(jobs=self.JOBS)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            assert app.screen._flourish is True  # the launch flourish plays once
            await pilot.press("down")  # move to the SECOND job
            await pilot.press("enter")  # open it
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, DashboardScreen):
                    break
            assert isinstance(app.screen, DashboardScreen)
            assert app.screen.job_ctx.job_id == "12345"  # the second job
            await pilot.press("q")  # quit the dashboard
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, JobSelectorScreen):
                    break
            assert isinstance(app.screen, JobSelectorScreen)  # back to the list
            assert app.screen._flourish is False  # no re-animation on return
            assert app.return_code is None  # still running, not exited
            # The cursor is restored to the job that was opened (index 1), not reset
            # to the top.
            assert app.screen.query_one(ListView).index == 1
            await pilot.press("escape")  # now cancel the selector
            await pilot.pause()
        assert app.return_code == 0  # escaping the selector exits cleanly

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_cursor_restored_for_a_job_that_only_exists_via_live_refresh(self) -> None:
        # The picker's live-refresh (`refresh=`) can add a job the run loop's own
        # `jobs` snapshot never had (taken once, before the picker even opened).
        # Selecting that job and returning must still land the cursor on it —
        # not silently fall back to a stale index because the lookup searched the
        # snapshot instead of the screen's own live-refreshed list.
        from textual.widgets import ListView

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        initial: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
            {"job_id": "12345", "state": "R", "partition": "gpu", "name": "b", "nodes": "1"},
        ]
        refreshed: list[dict[str, object]] = [
            *initial,
            {"job_id": "999", "state": "R", "partition": "gpu", "name": "new", "nodes": "1"},
        ]
        # No `refresh=` at construction — on_mount only schedules the live-refresh
        # timer/interval when one is present, so leaving it unset here means no
        # background poll ever fires on its own. `_poll_jobs` is then driven
        # directly below, fully awaited in this coroutine with no concurrent timer
        # to race against.
        app = SlurmwatchApp(jobs=initial, config=SlurmwatchConfig())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            screen = app.screen
            screen._refresh = lambda: refreshed
            await screen._poll_jobs()
            await pilot.pause()
            assert len(screen.jobs) == 3  # the live refresh landed
            lv = screen.query_one(ListView)
            lv.focus()
            lv.index = 2  # the newly-appeared job's row
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, DashboardScreen):
                    break
            assert isinstance(app.screen, DashboardScreen)
            assert app.screen.job_ctx.job_id == "999"
            await pilot.press("q")  # quit the dashboard
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, JobSelectorScreen):
                    break
            assert isinstance(app.screen, JobSelectorScreen)  # back to the list
            assert app.screen.query_one(ListView).index == 2

    def test_the_time_column_is_read_against_its_own_sample(self) -> None:
        # The mechanism behind the drift, in isolation: a row's `wall_time` is an
        # elapsed time as of when squeue was read, so it is only meaningful paired
        # with that instant. Pair it with an older one and the column over-reports
        # by exactly the difference — silently, and in Slurm's own format, so it
        # looks like a measurement.
        from slurmwatch.tui import JobSelectorScreen

        job: dict[str, object] = {
            "job_id": "111",
            "state": "R",
            "partition": "gpu",
            "name": "a",
            "nodes": "1",
            "wall_time": "2:00:00",
        }
        now = time.time()
        assert JobSelectorScreen([job], reference=now)._cell(job, "_tail") == "2:00:00"
        stale = JobSelectorScreen([job], reference=now - 3600)
        assert stale._cell(job, "_tail") == "3:00:00"  # the bug, an hour of drift
        # No live clock at all is a static snapshot, which is honest.
        assert JobSelectorScreen([job], reference=None)._cell(job, "_tail") == "2:00:00"

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_a_live_refresh_reanchors_the_sample_time(self) -> None:
        # `_poll_jobs` writes the list and the instant it was sampled together,
        # because each row's elapsed time is relative to that instant. `sample`
        # hands both back so a caller cannot take one without the other.
        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        initial: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
        ]
        refreshed: list[dict[str, object]] = [
            *initial,
            {"job_id": "999", "state": "R", "partition": "gpu", "name": "new", "nodes": "1"},
        ]
        app = SlurmwatchApp(jobs=initial, config=SlurmwatchConfig())
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, JobSelectorScreen)
            before = screen._reference
            assert before is not None
            screen._refresh = lambda: refreshed
            await screen._poll_jobs()
            await pilot.pause()
            jobs, sampled_at = screen.sample
            assert len(jobs) == 2  # the rebuild landed...
            assert sampled_at is not None and sampled_at >= before  # ...and re-anchored
            # An unchanged key set early-returns, so nothing moves and the pair
            # stays intact — which is also why a stale reference never washes out.
            screen._refresh = lambda: refreshed
            await screen._poll_jobs()
            assert screen.sample[1] == sampled_at

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_returning_to_the_picker_carries_the_sample_time_with_the_list(self) -> None:
        # The run loop adopted `screen.jobs` but kept its own `reference` from
        # before the session started, so the next picker paired a list sampled at
        # T1 with a reference of T0 and double-counted T1-T0. It did not
        # self-correct: `_poll_jobs` early-returns on an unchanged key set, so on a
        # quiet cluster the TIME column stayed wrong for the whole picker session.
        from textual.widgets import ListView

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        jobs: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
            {"job_id": "12345", "state": "R", "partition": "gpu", "name": "b", "nodes": "1"},
        ]
        app = SlurmwatchApp(jobs=jobs, config=SlurmwatchConfig())
        async with app.run_test() as pilot:
            await pilot.pause()
            first = app.screen
            assert isinstance(first, JobSelectorScreen)
            # Stand in for a list-changing refresh an hour into the session: the
            # screen re-anchors, the run loop's own timestamp does not.
            resampled = first._reference
            assert resampled is not None
            resampled += 3600
            first._reference = resampled
            lv = first.query_one(ListView)
            lv.focus()
            lv.index = 1
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, DashboardScreen):
                    break
            assert isinstance(app.screen, DashboardScreen)
            await pilot.press("q")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, JobSelectorScreen):
                    break
            assert isinstance(app.screen, JobSelectorScreen)
            assert app.screen is not first  # a fresh screen, built by the run loop
            assert app.screen._reference == resampled

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_pending_pick_opens_pending_view(self) -> None:
        # A PENDING pick must route to the why/when/where view, not try to attach a
        # live collector (which can't work on a queued job).
        from slurmwatch.tui import JobSelectorScreen, PendingScreen, SlurmwatchApp

        jobs: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
            {
                "job_id": "999",
                "state": "PD",
                "partition": "gpu",
                "name": "queued",
                "nodes": "2",
                "reason": "(Priority)",
            },
        ]
        app = SlurmwatchApp(jobs=jobs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            await pilot.press("down")  # move to the pending job
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, PendingScreen):
                    break
            assert isinstance(app.screen, PendingScreen)

    async def test_foreign_job_pick_shows_readonly_view_not_a_live_dashboard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Picking another user's job from the bare-`sw` picker must not try to
        # attach a live collector: Slurm denies step creation and sstat to
        # everyone but the job's owner, so cli.py's `_run_interactive` already
        # detects this up front (`_job_owner_differs`) before it would attempt
        # the doomed hop — the picker's own path must do the same.
        import slurmwatch.tui as tui
        from slurmwatch.tui import ForeignJobScreen, JobSelectorScreen, SlurmwatchApp

        jobs: list[dict[str, object]] = [
            {"job_id": "111", "state": "R", "partition": "gpu", "name": "a", "nodes": "1"},
        ]
        foreign_ctx = JobContext(
            job_id="111",
            username="otheruser",
            partition="gpu",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=["cn001"],
            raw_job_id="111",
            job_state="RUNNING",
            job_start_time=1000.0,
            time_limit_seconds=7200,
            remote=True,
        )
        monkeypatch.setattr("getpass.getuser", lambda: "me")
        monkeypatch.setattr(tui, "resolve_job_context", lambda job_id: foreign_ctx)
        collector_started = False

        async def _boom_start(self: object) -> None:
            nonlocal collector_started
            collector_started = True

        monkeypatch.setattr("slurmwatch.collector.TelemetryCollector.start", _boom_start)
        app = SlurmwatchApp(jobs=jobs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobSelectorScreen)
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, ForeignJobScreen):
                    break
            assert isinstance(app.screen, ForeignJobScreen)
        assert collector_started is False  # never attempted a live collector

    def test_job_line_tags_running_and_pending(self) -> None:
        from slurmwatch.tui import JobSelectorScreen

        jobs: list[dict[str, object]] = [
            {"job_id": "1", "state": "R", "partition": "gpu", "name": "x", "nodes": "1"},
            {"job_id": "2", "state": "PD", "partition": "gpu", "name": "y", "reason": "(Priority)"},
        ]
        scr = JobSelectorScreen(jobs)
        widths = scr._column_widths()
        run = scr._job_line(jobs[0], widths)
        pend = scr._job_line(jobs[1], widths)
        assert "RUNNING" in run and "PENDING" not in run
        assert "PENDING" in pend and "Priority" in pend  # pending shows its reason, not time
        # A column header names each field so the reader knows what they're seeing.
        header = scr._header_line(widths)
        assert "JOB ID" in header and "STATE" in header and "PARTITION" in header
        # A pending job IS listed, so the reason column is headed "TIME / WHY".
        assert "TIME / WHY" in header

    def test_time_column_header_drops_why_when_nothing_is_pending(self) -> None:
        # The last column only shows a "why" (scheduler reason) for pending jobs;
        # with all jobs running it's just elapsed time, so the header is plain "TIME"
        # (no dangling "/ WHY").
        from slurmwatch.tui import JobSelectorScreen

        running: list[dict[str, object]] = [
            {
                "job_id": "1",
                "state": "R",
                "partition": "gpu",
                "name": "x",
                "nodes": "1",
                "wall_time": "1:00",
            },
            {
                "job_id": "2",
                "state": "R",
                "partition": "gpu",
                "name": "y",
                "nodes": "1",
                "wall_time": "2:00",
            },
        ]
        scr = JobSelectorScreen(running)
        header = scr._header_line(scr._column_widths())
        assert "TIME" in header and "WHY" not in header

    def test_time_column_ticks_live_from_reference(self) -> None:
        # A running job's TIME advances from the sample instant so the picker isn't a
        # frozen clock: with the reference 125s in the past, a job at 1:00 reads ~3:05.
        from slurmwatch.slurm import _parse_slurm_duration
        from slurmwatch.tui import JobSelectorScreen

        job: dict[str, object] = {
            "job_id": "1",
            "state": "R",
            "partition": "gpu",
            "name": "x",
            "nodes": "1",
            "wall_time": "1:00",
        }
        live = JobSelectorScreen([job], reference=time.time() - 125)._cell(job, "_tail")
        assert 184 <= _parse_slurm_duration(live) <= 190  # 60 + ~125s, ticked forward
        # No reference → a static snapshot (the raw squeue value, unticked).
        assert JobSelectorScreen([job])._cell(job, "_tail") == "1:00"

    def test_selector_name_column_is_capped_like_every_other_name_site(self) -> None:
        # _column_widths sizes NAME to the longest name and the box is max-width 96%, so
        # one un-capped sweep-style name pushed the header, rule and EVERY row past the
        # terminal edge — hiding STATE / PARTITION / NODES / TIME for all the other jobs.
        # This was the only name-render site that skipped _elide_job_name.
        from slurmwatch.tui import _JOB_NAME_MAX, JobSelectorScreen

        long_name = "sweep-lr3e4-wd0.01-warmup2000-cosine-bs512-seed7-h100x4-run17-resume"
        job: dict[str, object] = {
            "job_id": "1",
            "state": "R",
            "partition": "gpu",
            "name": long_name,
            "nodes": "1",
            "wall_time": "1:00",
        }
        scr = JobSelectorScreen([job])
        cell = scr._cell(job, "name")
        assert len(cell) <= _JOB_NAME_MAX
        assert len(long_name) > _JOB_NAME_MAX  # the input really was over the cap
        # And the column sized from it stays bounded, so the other columns survive.
        name_col = [k for _, k in scr._COLUMNS].index("name")
        assert scr._column_widths()[name_col] <= _JOB_NAME_MAX

    def test_format_slurm_elapsed_matches_squeue_style(self) -> None:
        from slurmwatch.tui import _format_slurm_elapsed

        assert _format_slurm_elapsed(28 * 60 + 35) == "28:35"  # M:SS under an hour
        assert _format_slurm_elapsed(3600 + 61) == "1:01:01"  # H:MM:SS under a day
        assert _format_slurm_elapsed(5 * 86400 + 20 * 3600 + 41 * 60 + 36) == "5-20:41:36"

    async def test_selector_refreshes_job_list_live(self) -> None:
        # The picker re-queries Slurm on a timer: a newly-submitted job appears and a
        # finished one drops out WITHOUT quitting/reopening, and the cursor stays on
        # the same job across the rebuild.
        from textual.app import App
        from textual.widgets import ListItem, ListView

        from slurmwatch.tui import JobSelectorScreen

        two: list[dict[str, object]] = [
            {
                "job_id": "1",
                "state": "R",
                "partition": "gpu",
                "name": "a",
                "nodes": "1",
                "wall_time": "1:00",
            },
            {
                "job_id": "2",
                "state": "R",
                "partition": "gpu",
                "name": "b",
                "nodes": "1",
                "wall_time": "2:00",
            },
        ]
        three = two + [
            {
                "job_id": "3",
                "state": "R",
                "partition": "gpu",
                "name": "c",
                "nodes": "1",
                "wall_time": "0:05",
            },
        ]
        box: dict[str, list[dict[str, object]]] = {"jobs": three}
        scr = JobSelectorScreen(two, refresh=lambda: box["jobs"])

        class Host(App[None]):
            def on_mount(self) -> None:
                self.push_screen(scr)

        async def settle(expected: int) -> None:
            """Wait for the ListView to reach ``expected`` rows.

            A single ``pilot.pause()`` yields one event-loop cycle, which is not
            always enough for Textual to mount/unmount the rows a rebuild changed:
            this test failed on CI as ``assert 2 == 1`` while a removed row was
            still in the DOM, on py3.10 only, and passed on re-run. Polling for the
            end state tests the same contract ("appears/drops out live") without
            depending on how many cycles the rebuild happens to take — the idiom
            already used by test_app_selector_refreshes_when_jobs_and_refresh_both_passed.
            """
            for _ in range(80):
                if len(scr.query(ListItem)) == expected:
                    return
                await pilot.pause(0.05)

        async with Host().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            lv = scr.query_one(ListView)
            # Poll for the INITIAL mount too, for the reason settle() documents: one
            # event-loop cycle is not always enough to mount the rows, and under load
            # (this suite alongside a busy box) this assert was the one that flaked.
            await settle(2)
            assert len(scr.query(ListItem)) == 2  # initial snapshot
            lv.index = 1  # cursor on job "2"
            await scr._poll_jobs()  # a new job (3) was submitted
            await settle(3)
            assert len(scr.query(ListItem)) == 3  # appeared live, no restart
            assert str(scr.jobs[lv.index]["job_id"]) == "2"  # cursor kept on job 2
            box["jobs"] = [two[1]]  # jobs 1 and 3 finished; only 2 remains
            await scr._poll_jobs()
            await settle(1)
            assert len(scr.query(ListItem)) == 1  # dropped out live
            assert str(scr.jobs[0]["job_id"]) == "2"

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_app_selector_refreshes_when_jobs_and_refresh_both_passed(self) -> None:
        # REGRESSION (the "#live" bug): the CLI pre-resolves `jobs` AND wants live
        # refresh, so passing `jobs` must NOT disable refresh. Previously refresh was
        # inferred from `self._jobs is None`, so the real CLI (which always passes
        # jobs) never refreshed. Drive the exact CLI shape: jobs + refresh together.
        from textual.widgets import ListItem

        from slurmwatch.tui import JobSelectorScreen, SlurmwatchApp

        def make(n: int) -> list[dict[str, object]]:
            return [
                {
                    "job_id": str(i),
                    "state": "R",
                    "partition": "test",
                    "name": f"j{i}",
                    "nodes": "1",
                    "wall_time": "1:00",
                }
                for i in range(n)
            ]

        box: dict[str, list[dict[str, object]]] = {"jobs": make(2)}
        app = SlurmwatchApp(jobs=make(2), refresh=lambda: box["jobs"])
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(40):
                await pilot.pause(0.1)
                if isinstance(app.screen, JobSelectorScreen):
                    break
            scr = app.screen
            assert isinstance(scr, JobSelectorScreen)
            assert scr._refresh is not None  # wired despite jobs being passed
            assert len(scr.query(ListItem)) == 2
            box["jobs"] = make(5)  # 3 jobs submitted while the picker is open
            for _ in range(80):
                await pilot.pause(0.1)
                if len(scr.query(ListItem)) == 5:
                    break
            assert len(scr.query(ListItem)) == 5  # appeared live via the CLI refresh

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
            peak_working_set_bytes=30 * 1024**3,
            # An ordinary on-node reading: the kernel handed us a real high-water
            # counter (v1 memory.max_usage_in_bytes / v2 memory.peak).
            peak_is_lifetime=True,
        ),
        gpus=[_make_gpu(72.5, 18 * 1024**3, 20 * 1024**3)],
        gpu_count_requested=1,
        gpu_active_count=1,
    )


def _many_device_snapshot(devices: int) -> TelemetrySnapshot:
    """A snapshot for a node exposing ``devices`` GPUs, interconnect grid included.

    The grid is what makes one frame O(devices^2): ``matrix[i][j]`` is a cell per
    ORDERED PAIR, so it dwarfs the per-device blocks well before the device count
    looks extreme. 56 is 8 GPUs x 7 MIG slices, an ordinary A100/H100 partitioning.
    """
    snap = _make_snapshot()
    snap.gpus = [_make_gpu(72.5, 18 * 1024**3, 20 * 1024**3, index=i) for i in range(devices)]
    for i, gpu in enumerate(snap.gpus):
        # A real NVML uuid and MIG product name, not the fixture's short ones: the
        # size claim this factory exists to test is about what a node really emits.
        gpu.uuid = f"MIG-{i:08x}-1234-5678-9abc-def012345678"
        gpu.name = "NVIDIA A100-SXM4-80GB MIG 1g.10gb"
        gpu.cuda_ordinal = i
        gpu.throttle_reasons = ["sw_power_cap"]
    snap.gpu_count_requested = devices
    snap.gpu_active_count = devices
    snap.interconnect = GpuInterconnect(
        fabric="nvlink",
        nvlink_version=4,
        links_per_gpu=18,
        link_speed_gbps=26.5,
        per_gpu_gbps=900.0,
        nvswitch=True,
        devices=list(range(devices)),
        matrix=[["self" if i == j else "NV18" for j in range(devices)] for i in range(devices)],
        nvlink_rx_gbps=[12.3456] * devices,
        nvlink_tx_gbps=[12.3456] * devices,
        pcie_rx_gbps=[1.2345] * devices,
        pcie_tx_gbps=[1.2345] * devices,
    )
    return snap


class TestForeignJobViewAlloc:
    """N8: the read-only foreign view's Allocation line must not misread per-node
    CPU/mem as whole-job totals."""

    def _ctx(self, nodes: list[str], cpus: int, mem_gib: int) -> JobContext:
        return JobContext(
            job_id="9_1",
            username="ada",
            partition="gpu",
            nodelist=",".join(nodes),
            hostname=nodes[0],
            cpus_allocated=cpus,
            mem_limit_bytes=mem_gib * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            nodelist_resolved=nodes,
        )

    def test_multinode_labels_cpu_and_mem_per_node(self) -> None:
        out = ForeignJobView()._alloc(self._ctx(["cn1", "cn2", "cn3", "cn4"], 16, 64), " · ")
        assert "16 CPU/node" in out and "64.0 GiB/node" in out

    def test_singlenode_omits_the_per_node_suffix(self) -> None:
        out = ForeignJobView()._alloc(self._ctx(["cn1"], 16, 64), " · ")
        assert "16 CPU" in out and "/node" not in out


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

    @pytest.mark.asyncio
    async def test_unparseable_line_does_not_reset_backoff_counter(self) -> None:
        # N5: a non-empty but unparseable line (version skew) must NOT reset the
        # stream-fail counter — only a genuinely parsed frame does.
        scr = self._screen(["cn001", "cn002"])

        class _Out:
            async def readline(self) -> bytes:
                return b"{oops: not our schema}\n"

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        scr._stream_proc = _Proc()  # type: ignore[assignment]
        scr._stream_node = "cn002"
        scr._stream_fails = 3
        assert await scr._read_remote("cn002") is None
        assert scr._stream_fails == 3  # untouched by an unparseable line
        await scr._stop_stream()

    @pytest.mark.asyncio
    async def test_read_timeout_returns_none_without_raising(self) -> None:
        # asyncio.wait_for's own timeout must be caught, not propagate. Pre-3.11,
        # asyncio.TimeoutError is a distinct class from the builtin TimeoutError (they
        # became the same object only in 3.11), so a bare `except TimeoutError:`
        # would miss it on this project's Python 3.10 floor.
        scr = self._screen(["cn001", "cn002"])

        class _Out:
            async def readline(self) -> bytes:
                await asyncio.sleep(3600)  # outlives the 0.5s read timeout
                return b""

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        scr._stream_proc = _Proc()  # type: ignore[assignment]
        scr._stream_node = "cn002"
        assert await scr._read_remote("cn002") is None
        await scr._stop_stream()

    @pytest.mark.asyncio
    async def test_persistent_version_skew_retires_the_stream(self) -> None:
        # N5: a node streaming only garbage (an incompatible slurmwatch build) must
        # be retired like a dead stream after the threshold — stop + back off —
        # instead of hanging the switch forever while `_show` is never called.
        from slurmwatch.tui import _STREAM_MAX_PARSE_FAILS

        scr = self._screen(["cn001", "cn002"])

        class _Out:
            async def readline(self) -> bytes:
                return b"garbage line\n"

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        proc = _Proc()
        scr._stream_proc = proc  # type: ignore[assignment]
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"  # so _stream_backoff returns without sleeping
        for _ in range(_STREAM_MAX_PARSE_FAILS):
            assert await scr._read_remote("cn002") is None
        assert scr._stream_proc is None  # retired, not streaming garbage forever
        assert proc.returncode == -9  # retired, not streaming garbage forever

    @staticmethod
    def _raising_stream_proc(exc: BaseException) -> object:
        """A stream whose ``readline`` raises ``exc`` on every frame."""

        class _Out:
            async def readline(self) -> bytes:
                raise exc

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        return _Proc()

    @pytest.mark.asyncio
    async def test_overlong_frame_retires_the_stream_instead_of_latching(self) -> None:
        """A frame past the pipe's line limit must degrade like an unparseable one.

        One JSON frame is O(devices^2) — GpuInterconnect.matrix is the
        device-by-device topology grid — so it crosses asyncio's default 64 KiB
        StreamReader line limit at 56 devices (measured 65,553 B, which is 8 GPUs x
        7 MIG slices). Past the limit ``readline()`` raises ValueError and DISCARDS
        the line, so the next frame from that node fails identically, forever.

        Only TimeoutError was caught here, so that ValueError escaped to
        ``_poll_loop``'s broad ``except Exception`` (B-C7): ``_stream_parse_fails``
        was never reached, N5's retirement could never fire, and the node latched on
        "still reaching…" at two iterations a second with the cause visible only in
        the Textual log. The same hang N5 fixed, reached through a path that never
        gets as far as parsing. It has to count as an unusable line so the switch
        resolves.
        """
        from slurmwatch.tui import _STREAM_MAX_PARSE_FAILS

        scr = self._screen(["cn001", "cn002"])
        proc = self._raising_stream_proc(
            ValueError("Separator is found, but chunk is longer than limit")
        )
        scr._stream_proc = proc  # type: ignore[assignment]
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"  # so _stream_backoff returns without sleeping
        for _ in range(_STREAM_MAX_PARSE_FAILS):
            # Not raised: the poll loop sees a dropped frame, as for any bad line.
            assert await scr._read_remote("cn002") is None
        assert scr._stream_proc is None  # retired, not latched on "still reaching…"
        assert proc.returncode == -9  # type: ignore[attr-defined]  # the stream srun was killed

    @pytest.mark.asyncio
    async def test_an_unexpected_stream_error_is_not_swallowed(self) -> None:
        # Control for the test above: the ValueError branch must not widen into a
        # blanket `except Exception`. A bug in our own read path is not an over-long
        # line, and counting it as one would hide it behind N5's retirement — the
        # frame is dropped either way, but the traceback saying WHY is the only
        # thing that ever reaches a developer.
        scr = self._screen(["cn001", "cn002"])
        scr._stream_proc = self._raising_stream_proc(  # type: ignore[assignment]
            RuntimeError("a bug in the read path, not an over-long line")
        )
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"
        with pytest.raises(RuntimeError):
            await scr._read_remote("cn002")

    @pytest.mark.asyncio
    async def test_stream_pipe_line_limit_fits_a_many_slice_mig_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``open_stream`` must size the pipe's line limit to a real large-node frame.

        The two tests above cover what happens when a frame is still too big; this
        is the one that keeps a legitimate node from getting there at all. asyncio's
        default is 64 KiB and a 56-device frame measures 65,553 B, so the default
        silently made the biggest nodes — the ones a right-sizing monitor is most
        wanted on — unreadable by the node switcher.
        """
        from slurmwatch import remote as remotemod

        captured: dict[str, Any] = {}

        class _Proc:
            returncode: int | None = None
            stdout = None
            stderr = None

            def kill(self) -> None:
                pass

            async def wait(self) -> int:
                return 0

        async def fake_exec(*_cmd: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _Proc()

        async def can_get_gpu(_job: str, _node: str) -> bool:
            return True

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(remotemod, "_stream_can_get_gpu", can_get_gpu)
        assert await remotemod.open_stream("12345", "cn002") is not None

        limit = captured["limit"]
        # Derived from the real payload, not a magic number, so a schema change
        # re-checks itself. The `> default` assertion is the control: asyncio's own
        # default is precisely what such a frame does not fit in, so the limit has
        # to be RAISED, not merely stated.
        frame = len(_many_device_snapshot(56).to_json())
        assert frame > 2**16, "the 56-device frame is the whole reason for the limit"
        assert limit >= frame, f"a {frame} B frame does not fit a {limit} B limit"

    @staticmethod
    def _dead_stream_proc(stderr_text: bytes) -> object:
        """A stream that has already died, with ``stderr_text`` waiting to be read."""

        class _Out:
            async def readline(self) -> bytes:
                return b""  # EOF: the stream died

        class _Err:
            async def read(self, _n: int) -> bytes:
                return stderr_text

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Out()
                self.stderr = _Err()
                self.returncode: int | None = None

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return -9

        return _Proc()

    @pytest.mark.asyncio
    async def test_ssh_being_refused_does_not_retire_the_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A site that blocks login->compute ssh must still get the step transport.

        Measured on a Booth cluster: `ssh mcn57` answers `Permission denied
        (publickey,gssapi-keyex,gssapi-with-mic,password).` — text a REFUSED SLURM
        STEP also produces, so `stream_error_is_permanent` matched and the switcher
        gave up on a node whose `--gres=none` step works. The transport is recorded
        at launch, so which rung spoke is a fact here, not an inference.
        """
        from slurmwatch import remote as _remote

        scr = self._screen(["cn001", "cn002"])
        _remote._STREAM_TRANSPORT["cn002"] = "ssh"
        scr._stream_proc = self._dead_stream_proc(  # type: ignore[assignment]
            b"youzhi@cn002: Permission denied (publickey,gssapi-keyex,password)."
        )
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"  # so _stream_backoff returns without sleeping
        assert await scr._read_remote("cn002") is None
        assert scr._stream_gave_up is False, "the step rung was never tried"
        assert _remote.stream_transport("cn002") == "", "ssh retired for this node"
        assert "ssh" in scr._stream_error and "Slurm refused" not in scr._stream_error

    @pytest.mark.asyncio
    async def test_the_step_failing_too_does_retire_the_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The complement: with no rung left, giving up is the right answer."""
        from slurmwatch import remote as _remote

        scr = self._screen(["cn001", "cn002"])
        _remote._STREAM_TRANSPORT["cn002"] = "step"
        scr._stream_proc = self._dead_stream_proc(  # type: ignore[assignment]
            b"srun: error: Access/permission denied for job 12345"
        )
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"
        assert await scr._read_remote("cn002") is None
        assert scr._stream_gave_up is True
        assert "Slurm refused a step" in scr._stream_error

    @pytest.mark.asyncio
    async def test_an_install_the_node_cannot_see_still_retires_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execve() over ssh is permanent on BOTH rungs, so one extra try, then stop.

        The fallback must not turn a genuinely hopeless failure into a retry loop:
        the second attempt runs as a step, fails the same way, and retires the node.
        """
        from slurmwatch import remote as _remote

        scr = self._screen(["cn001", "cn002"])
        scr._selected_node = "cn001"
        for transport, expected_gave_up in (("ssh", False), ("step", True)):
            _remote._STREAM_TRANSPORT["cn002"] = transport
            scr._stream_proc = self._dead_stream_proc(  # type: ignore[assignment]
                b"error: execve(): /tmp/v/bin/python: No such file or directory"
            )
            scr._stream_node = "cn002"
            assert await scr._read_remote("cn002") is None
            assert scr._stream_gave_up is expected_gave_up, transport

    @pytest.mark.asyncio
    async def test_a_transport_that_never_connected_hands_over_to_the_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refused case is handled by its WORDING; a site that drops the connection
        instead answers "Connection timed out", which matched no permanent token — so
        `retry_other_stream_transport` was never consulted and the switcher kept
        relaunching the rung that cannot work there, every backoff, for the session.

        The wording is openssh's own, captured from this login node against an
        unreachable target. What is asserted is the hand-over, not a duration: after
        this failure the node's recorded rung is cleared, so the NEXT launch is the
        `--gres=none` step, and the node is not given up on.
        """
        from slurmwatch import remote as _remote

        scr = self._screen(["cn001", "cn002"])
        _remote._STREAM_TRANSPORT["cn002"] = "ssh"
        scr._stream_proc = self._dead_stream_proc(  # type: ignore[assignment]
            b"connect to host cn002 port 22: Connection timed out"
        )
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"  # so _stream_backoff returns without sleeping
        assert await scr._read_remote("cn002") is None
        assert scr._stream_gave_up is False, "the step rung was never tried"
        assert _remote.stream_transport("cn002") == "", "the dead rung was not retired"

    @pytest.mark.asyncio
    async def test_a_working_stream_that_dies_later_keeps_its_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL — passes in both states. Only CONNECT-time failures demote a rung.
        The ssh rung is the only one that can read the job's GPUs, so a stream that got
        in and died an hour later must stay a retry on the same rung; demoting it would
        silently swap live GPU numbers for "GPU unreadable" for the rest of the session.
        """
        from slurmwatch import remote as _remote

        scr = self._screen(["cn001", "cn002"])
        _remote._STREAM_TRANSPORT["cn002"] = "ssh"
        scr._stream_proc = self._dead_stream_proc(  # type: ignore[assignment]
            b"Connection to cn002 closed by remote host."
        )
        scr._stream_node = "cn002"
        scr._selected_node = "cn001"
        assert await scr._read_remote("cn002") is None
        assert scr._stream_gave_up is False
        assert _remote.stream_transport("cn002") == "ssh", "demoted on a transient death"

    @pytest.mark.asyncio
    async def test_stop_stream_reaps_child_within_the_loop(self) -> None:
        # Regression (Event-loop-closed on quit): the killed stream child must be
        # REAPED (awaited) inside the live loop, so neither its stdout-pipe nor its
        # subprocess transport is left for the GC to finalize AFTER the loop closes
        # — that finalization calls loop.call_soon on the dead loop and prints a
        # spurious "RuntimeError: Event loop is closed". A pending readline on the
        # pipe at teardown is what keeps the transport open, so killing without
        # reaping is not enough.
        scr = self._screen(["cn001", "cn002"])

        class _ReapableProc:
            def __init__(self) -> None:
                self.returncode: int | None = None
                self.killed = False
                self.awaited = False

            def kill(self) -> None:
                self.killed = True
                self.returncode = -9

            async def wait(self) -> int:
                self.awaited = True
                return -9

        proc = _ReapableProc()
        scr._stream_proc = proc  # type: ignore[assignment]
        await scr._stop_stream()
        assert proc.killed
        assert proc.awaited  # reaped within the loop, so the transport closes now
        assert scr._stream_proc is None

    async def test_stop_stream_stays_bounded_on_unreapable_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B2: a stream child wedged in D-state can't be reaped even after SIGKILL,
        # so the reap is BOUNDED — _stop_stream must return promptly and never hang
        # on_unmount and trap the user in a TUI they can't quit. asyncio's child
        # watcher reaps the zombie once it finally dies.
        monkeypatch.setattr("slurmwatch.tui._STREAM_REAP_TIMEOUT", 0.05)
        scr = self._screen(["cn001", "cn002"])

        class _HangingProc:
            def __init__(self) -> None:
                self.returncode: int | None = None
                self.killed = False

            def kill(self) -> None:
                # A D-state child does NOT get reaped by kill: returncode stays None.
                self.killed = True

            async def wait(self) -> int:
                await asyncio.Event().wait()  # never returns
                return 0  # pragma: no cover

        proc = _HangingProc()
        scr._stream_proc = proc  # type: ignore[assignment]
        # Would hang forever on an UNBOUNDED await; the bounded reap completes fast.
        await asyncio.wait_for(scr._stop_stream(), timeout=2.0)
        assert proc.killed
        assert scr._stream_proc is None

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

    def test_the_first_attach_is_connecting_not_switching(self) -> None:
        # Armed at mount for an already-remote job: nothing was switched, so
        # "switching to node 1 of 1" described neither the action nor anything the
        # reader can act on.
        banner = SwitchBanner()
        banner.target_label = "the compute node"
        banner.node = "cn001"
        banner.connecting = True
        banner.multi = False
        out = _render_markup(banner.render()).plain
        assert "connecting to the compute node" in out and "cn001" in out
        assert "switching" not in out and "of 1" not in out

    def test_a_single_node_job_is_not_told_to_switch_nodes(self) -> None:
        banner = SwitchBanner()
        banner.target_label = "the compute node"
        banner.node = "cn001"
        banner.connecting = True
        banner.multi = False
        banner.stuck = True
        out = _render_markup(banner.render()).plain
        assert "still connecting to the compute node" in out
        assert "still retrying" in out
        assert "switch nodes" not in out  # there is no other node to switch to
        banner.multi = True
        assert "switch nodes" in _render_markup(banner.render()).plain


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


class TestForeignJobView:
    """Read-only facts view for another user's running job (no live telemetry).

    Renders unmounted (like TestJobDetailsPanel): ``render`` reads
    ``self.size.width or 100``, so an unmounted widget renders at width 100.
    """

    def _view(self, **overrides: object) -> ForeignJobView:
        v = ForeignJobView()
        v.job_ctx = _provenance_ctx(**overrides)
        v.config = SlurmwatchConfig()
        return v

    def test_shows_owner_state_and_node(self) -> None:
        out = self._view(username="yifchen", job_state="RUNNING", nodelist="midway3-0532").render()
        assert "yifchen" in out
        assert "RUNNING" in out
        assert "midway3-0532" in out

    def test_shows_the_job_name(self) -> None:
        # scontrol JobName is readable cross-user, and on someone else's job it's the
        # main clue to what the node is busy with — so it belongs here too.
        out = _plain(self._view(username="yifchen", job_name="bert-finetune").render())
        assert "name bert-finetune" in out
        # Absent name -> no chip, and the identity line still renders.
        bare = _plain(self._view(username="yifchen", job_name="").render())
        assert "name " not in bare and "yifchen" in bare

    def test_explains_no_live_telemetry(self) -> None:
        out = self._view(username="yifchen").render()
        assert "No Live View" in out
        assert "another user's job" in out
        # The tip points them at running slurmwatch themselves.
        assert "slurmwatch" in out

    def test_the_no_live_view_note_keeps_its_indent_on_every_line(self) -> None:
        """Rich has no hanging indent, so a paragraph that WRAPS loses its own.

        Measured at 125 columns against a real foreign job on a Booth cluster: the
        em-dash clause ran past the card and the continuation came back at the card's
        padding, two columns left of the text it belonged to. The lines are broken
        where the meaning breaks instead, so each one carries its own indent.
        """
        note = _plain(self._view(username="yifchen").render()).split("No Live View")[1]
        lines = [ln for ln in note.splitlines() if ln.strip()]
        assert lines, note
        assert all(ln.startswith("  ") and not ln.startswith("   ") for ln in lines), lines

    def test_the_note_lines_fit_a_narrow_terminal_without_wrapping(self) -> None:
        """80 columns is what a default ssh session gives; nothing may exceed it.

        The card renders at width 100 unmounted and adds its own padding, so the
        margin here is deliberately generous — the check that matters is that no
        single line is long enough to wrap on an ordinary terminal.
        """
        note = _plain(self._view(username="yifchen").render()).split("No Live View")[1]
        for line in note.splitlines():
            assert len(line) <= 76, (len(line), line)

    def test_the_note_still_says_which_tool_and_who_to_ask(self) -> None:
        """Splitting the sentence must not drop what the reader is meant to DO."""
        note = _plain(self._view(username="yifchen").render()).split("No Live View")[1]
        assert "sstat" in note and "job's owner" in note
        assert "Ask yifchen" in note and "slurmwatch" in note

    def test_time_budget_line_when_limited(self) -> None:
        out = self._view(time_limit_seconds=72 * 3600).render()
        assert "Time Budget" in out
        assert "limit" in out

    @pytest.mark.parametrize("limit", [None, 0])
    def test_time_budget_line_when_unbounded(self, limit: int | None) -> None:
        """`TimeLimit=UNLIMITED` (and a 0 from a site that spells it that way) feeds
        the "ran X% of limit" division. No job on the portability-test cluster had
        one, so this path stayed unexercised — a unit test is cheaper than another
        cluster hunt (round-2 note)."""
        out = _plain(self._view(time_limit_seconds=limit).render())
        assert "no wall-clock time limit" in out
        assert "% " not in out.split("Time Budget")[1].splitlines()[1]

    def test_allocation_line(self) -> None:
        out = self._view(
            cpus_allocated=4, mem_limit_bytes=100 * 1024**3, gpu_count_requested=1
        ).render()
        assert "Allocation" in out
        assert "4 CPU" in out
        assert "GiB" in out
        assert "GPU" in out

    def test_bracketed_values_are_escaped(self) -> None:
        # A '[' in any user-controlled field (name/command) must be backslash-escaped
        # so Textual's markup parser can't crash the view (see textual-markup lesson).
        out = self._view(username="ev[il", command="/x/[weird]/run.py").render()
        assert r"\[" in out

    def test_array_membership_line(self) -> None:
        # An array task shows its base id + task index as a fact.
        out = self._view(array_job_id="52353625", array_task_id="15").render()
        assert "Array" in out
        assert "52353625" in out
        assert "task" in out and "15" in out

    def test_array_counts_shown_when_present(self) -> None:
        v = self._view(array_job_id="52353625", array_task_id="15")
        v.array_counts = (20, 3)
        out = v.render()
        assert "20" in out and "running" in out
        assert "3" in out and "pending" in out

    def test_no_array_line_for_non_array_job(self) -> None:
        # _provenance_ctx has no array fields → no Array section.
        out = self._view(username="yifchen").render()
        assert "Array\n" not in out

    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_app_mounts_and_quits(self) -> None:
        app = ForeignJobApp(_provenance_ctx(username="yifchen"))
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, ForeignJobScreen):
                    break
            assert isinstance(app.screen, ForeignJobScreen)
            # Quitting dismisses the screen; the app's worker then exits cleanly.
            await pilot.press("q")


class TestCrossNodeGpuView:
    """A multi-node job's GPU layout, visible without hopping node by node.

    The switcher answers "what is node 2 doing" one node at a time; it never
    answers "what did my job get overall". On a job that holds every GPU it was
    allocated, utilization is unreadable from any monitor step, so the allocation
    is the only GPU fact there is — and it was reachable only by visiting each
    node in turn.
    """

    BY_NODE = {"beagle3-0015": [0, 2], "beagle3-0020": [1, 2]}

    def test_lists_every_node_with_its_indices_and_the_job_total(self) -> None:
        from slurmwatch.tui import _cross_node_gpu_block

        out = _plain(
            _cross_node_gpu_block(self.BY_NODE, "beagle3-0015", "NVIDIA A100-PCIE-40GB", False)
        )
        assert "beagle3-0015" in out and "idx 0,2" in out
        assert "beagle3-0020" in out and "idx 1,2" in out
        # The job-wide total, which no single node's view can state.
        assert "4" in out and "across 2 nodes" in out
        assert "A100-PCIE-40GB" in out

    def test_model_is_claimed_only_for_the_node_it_was_read_from(self) -> None:
        """procfs is LOCAL, so the model cannot be asserted job-wide.

        An allocation is not guaranteed homogeneous: on this cluster the `test`
        partition alone spans a100, H100, L40S, rtx6000 and v100, so a 2-node job
        with no --constraint can hold two different cards. "4 x H100 across 2
        nodes" would then be a measurement claim about hardware never looked at.
        """
        from slurmwatch.tui import _cross_node_gpu_block

        out = _plain(
            _cross_node_gpu_block(self.BY_NODE, "beagle3-0020", "NVIDIA A100-PCIE-40GB", False)
        )
        header = out.splitlines()[0]
        # The header counts GPUs and nodes; it must NOT name the hardware.
        assert "across 2 nodes" in header
        assert "A100" not in header, header
        # The model sits on the measured node's line, beside "this view".
        here = next(ln for ln in out.splitlines() if "beagle3-0020" in ln)
        there = next(ln for ln in out.splitlines() if "beagle3-0015" in ln)
        assert "A100-PCIE-40GB" in here and "this view" in here
        assert "A100" not in there, there

    def test_marks_which_node_is_on_screen(self) -> None:
        from slurmwatch.tui import _cross_node_gpu_block

        here = _plain(_cross_node_gpu_block(self.BY_NODE, "beagle3-0020", "A100", False))
        # The marker rides on the node actually being viewed, so the reader can
        # place themselves in the list rather than guessing.
        line = next(ln for ln in here.splitlines() if "beagle3-0020" in ln)
        assert "this view" in line
        other = next(ln for ln in here.splitlines() if "beagle3-0015" in ln)
        assert "this view" not in other

    def test_matches_the_local_node_through_a_domain_suffix(self) -> None:
        """gethostname() often returns an FQDN while Slurm names the short host."""
        from slurmwatch.tui import _cross_node_gpu_block

        out = _plain(_cross_node_gpu_block(self.BY_NODE, "beagle3-0020.rcc.local", "A100", False))
        line = next(ln for ln in out.splitlines() if "beagle3-0020" in ln)
        assert "this view" in line

    def test_single_node_job_renders_nothing(self) -> None:
        from slurmwatch.tui import _cross_node_gpu_block

        assert _cross_node_gpu_block({"cn001": [0, 1]}, "cn001", "A100", False) == ""
        assert _cross_node_gpu_block({}, "cn001", "A100", False) == ""

    def test_ascii_mode_is_pure(self) -> None:
        from slurmwatch.tui import _cross_node_gpu_block

        out = _cross_node_gpu_block(self.BY_NODE, "beagle3-0015", "A100", True)
        for glyph in ("×", "←", "●"):
            assert glyph not in out, glyph

    def test_unknown_model_still_counts_the_gpus(self) -> None:
        from slurmwatch.tui import _cross_node_gpu_block

        out = _plain(_cross_node_gpu_block(self.BY_NODE, "beagle3-0015", "", False))
        assert "4 GPUs" in out and "across 2 nodes" in out

    def test_drill_in_actually_renders_the_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The WIRE, not just the helper.

        Without this, the whole `_set_body(_cross_node_gpu_block(...))` call could be
        deleted and every test above would still pass — the same untested-plumbing
        gap that let a hardcoded "" through on the collector side.
        """
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.hostname = "beagle3-0015"
        snap.node_count = 2
        snap.gpu_monitoring_available = False
        snap.gpu_unavailable_reason = "devices_denied"
        snap.gpu_node_count = 4
        snap.gpu_node_model = "NVIDIA A100-PCIE-40GB"
        snap.gpu_allocated_indices = [0, 2]

        ctx = JobContext(
            job_id="54117243",
            username="youzhi",
            partition="beagle3",
            nodelist="beagle3-[0015,0020]",
            hostname="beagle3-0015",
            cpus_allocated=4,
            mem_limit_bytes=52 * 1024**3,
            gpu_count_requested=2,
            gpu_indices=[0, 2],
            gpu_indices_by_node=dict(self.BY_NODE),
        )

        class _Dash:
            job_ctx = ctx
            resource_rows = None

        screen._dashboard = _Dash()  # type: ignore[assignment]
        captured: dict[str, str] = {}
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_headline", lambda s, t: captured.__setitem__("head", t)
        )
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_body", lambda s, t: captured.__setitem__("body", t)
        )
        monkeypatch.setattr(ResourceDetailScreen, "_clear_chart", lambda s: None)
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_figure", lambda s, *a, **k: None, raising=False
        )

        screen._refresh_gpu(snap, SlurmwatchConfig())
        body = _plain(captured.get("body", ""))
        assert "beagle3-0020" in body and "idx 1,2" in body, body
        assert "across 2 nodes" in body
        _valid_markup(captured["body"])


class TestNodeFabricLines:
    """Rendering the inter-node fabric — the multi-node job's real bottleneck."""

    FAB = NodeFabric(
        ports=1,
        link_rate_gbps=100.0,
        kind="InfiniBand",
        rate_label="100 Gb/sec (2X HDR)",
        rx_gbps=50.435,
        tx_gbps=45.623,
        rates_known=True,
    )

    def test_names_the_link_the_rate_and_the_share(self) -> None:
        from slurmwatch.tui import _node_fabric_lines

        out = _plain("\n".join(_node_fabric_lines(self.FAB, 2, False)))
        assert "InfiniBand" in out and "100 Gb/sec (2X HDR)" in out
        assert "50.4" in out and "45.6" in out
        # Gigabits, matching how the link is specced, so rate vs ceiling compares.
        assert "Gb/s" in out
        # 50.4 of 100 -> half the link.
        assert "50% of link" in out

    def test_says_node_wide_because_the_counters_are_the_hosts(self) -> None:
        """Port counters belong to the HOST: on a shared node other jobs are in it.

        Presenting it as the job's own traffic would be a measurement claim sw
        cannot support.
        """
        from slurmwatch.tui import _node_fabric_lines

        out = _plain("\n".join(_node_fabric_lines(self.FAB, 2, False)))
        assert "node-wide" in out and "all jobs" in out

    def test_first_frame_says_measuring_not_zero(self) -> None:
        from slurmwatch.tui import _node_fabric_lines

        fab = NodeFabric(ports=1, link_rate_gbps=100.0, kind="InfiniBand")
        out = _plain("\n".join(_node_fabric_lines(fab, 2, False)))
        assert "measuring" in out
        # A hard 0.0 here would read as "the fabric is idle", which is not known yet.
        assert "0.0" not in out

    def test_hidden_without_an_hca(self) -> None:
        from slurmwatch.tui import _node_fabric_lines

        assert _node_fabric_lines(None, 2, False) == []
        assert _node_fabric_lines(NodeFabric(), 2, False) == []

    def test_single_node_job_omits_the_gradient_hint(self) -> None:
        """The fabric is still shown (it is real), but nothing crosses nodes."""
        from slurmwatch.tui import _node_fabric_lines

        out = _plain("\n".join(_node_fabric_lines(self.FAB, 1, False)))
        assert "InfiniBand" in out
        assert "gradients" not in out

    def test_ascii_mode_is_pure(self) -> None:
        from slurmwatch.tui import _node_fabric_lines

        out = "\n".join(_node_fabric_lines(self.FAB, 2, True))
        for glyph in ("↑", "↓", "·", "—", "…"):
            assert glyph not in out, glyph

    def test_drill_in_renders_the_fabric_when_gpus_are_unreadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even blind on GPUs, "is the job moving data between nodes" still answers.

        Guards the WIRE: the call could be deleted and every test above would pass.
        """
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        snap = _make_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.hostname = "beagle3-0020"
        snap.node_count = 2
        snap.gpu_monitoring_available = False
        snap.gpu_unavailable_reason = "devices_denied"
        snap.gpu_node_model = "NVIDIA A100-PCIE-40GB"
        snap.fabric = self.FAB

        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="beagle3-0020",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[1, 2],
            gpu_indices_by_node={"beagle3-0006": [1, 2], "beagle3-0020": [1, 2]},
        )

        class _Dash:
            job_ctx = ctx
            resource_rows = None

        screen._dashboard = _Dash()  # type: ignore[assignment]
        captured: dict[str, str] = {}
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_headline", lambda s, t: captured.__setitem__("head", t)
        )
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_body", lambda s, t: captured.__setitem__("body", t)
        )
        monkeypatch.setattr(ResourceDetailScreen, "_clear_chart", lambda s: None)
        screen._refresh_gpu(snap, SlurmwatchConfig())
        body = _plain(captured.get("body", ""))
        assert "InfiniBand" in body, body
        assert "50.4" in body
        # ...alongside the cross-node allocation, not instead of it.
        assert "beagle3-0006" in body
        _valid_markup(captured["body"])


class TestMemoryPeakForSizing:
    """The MEM row's "peak" must be the number you can size --mem from.

    Real case (2-node job, beagle3-0020, 2026-08-22): the cgroup's lifetime peak
    was 27.8 GiB while the working set was 4.6 GiB, because the dashboard attached
    hours into the run. The row showed "peak 5 GiB" — size --mem off that and the
    next run OOMs at 5x under-provision.
    """

    @staticmethod
    def _mem(**kw: int) -> MemoryMetrics:
        import inspect

        base = {
            p.name: (0 if p.default is inspect._empty else p.default)
            for p in inspect.signature(MemoryMetrics).parameters.values()
        }
        base.update(kw)
        return MemoryMetrics(**base)  # type: ignore[arg-type]

    def test_prefers_the_cgroup_lifetime_peak_over_the_session_max(self) -> None:
        from slurmwatch.tui import _mem_peak_for_sizing

        mem = self._mem(
            peak_bytes=28 * 1024**3,
            peak_working_set_bytes=5 * 1024**3,
            current_bytes=6 * 1024**3,
        )
        assert _mem_peak_for_sizing(mem) == 28 * 1024**3

    def test_falls_back_to_the_session_max_when_no_cgroup_counter(self) -> None:
        """A /proc-only node has no lifetime counter; the window max is all there is."""
        from slurmwatch.tui import _mem_peak_for_sizing

        mem = self._mem(peak_bytes=0, peak_working_set_bytes=7 * 1024**3)
        assert _mem_peak_for_sizing(mem) == 7 * 1024**3

    def test_never_reports_lower_than_either_input(self) -> None:
        """Erring high costs memory; erring low costs the run."""
        from slurmwatch.tui import _mem_peak_for_sizing

        mem = self._mem(peak_bytes=3 * 1024**3, peak_working_set_bytes=9 * 1024**3)
        assert _mem_peak_for_sizing(mem) == 9 * 1024**3

    def test_dashboard_row_shows_the_lifetime_peak(self) -> None:
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.remote = False
        snap.memory = self._mem(
            limit_bytes=51 * 1024**3,
            current_bytes=6 * 1024**3,
            working_set_bytes=5 * 1024**3,
            peak_bytes=28 * 1024**3,
            peak_working_set_bytes=5 * 1024**3,
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        mem_line = next(ln for ln in _plain(r.render()).splitlines() if "MEM" in ln)
        assert "peak 28 GiB" in mem_line, mem_line
        assert "peak 5 GiB" not in mem_line


class TestRenderedPeakIsNeverBelowTheLiveReading:
    """The one symptom the cross-cluster report calls "do not ship this": a
    half-applied fix rendered

        ● MEM  used  18%   35 / 196 MiB · peak 0 GiB
        peak working set 0 GiB   ·   total now 39.6 MiB

    — a peak BELOW the current reading, which is worse than the useless-but-
    consistent `0 / 0 GiB` it replaced. Asserted as an invariant over the RENDERED
    text at the sizes where it lived, not just as unit behaviour of one helper.
    """

    def _rendered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        limit: int,
        used: int,
        *,
        peak: int | None = None,
        ws_peak: int | None = None,
    ) -> str:
        from slurmwatch.tui import ResourceDetailScreen

        snap = _make_snapshot()
        snap.memory.limit_bytes = limit
        snap.memory.current_bytes = used
        snap.memory.working_set_bytes = used
        # Defaults are CONSISTENT (peak above current), which is why this class only
        # ever exercised the formatting. The peak_/ws_peak_ overrides are for the
        # inputs that violate the invariant — the case that actually shipped.
        snap.memory.peak_bytes = int(used * 1.05) if peak is None else peak
        snap.memory.peak_working_set_bytes = int(used * 1.02) if ws_peak is None else ws_peak
        snap.memory.cache_bytes = used // 10  # a plausible cache for THIS size
        snap.memory.working_set_percent = used / limit * 100
        snap.gpus = []
        rows = _SizedRows(150)
        rows.snapshot = snap
        rows.config = SlurmwatchConfig()
        row = next(ln for ln in _plain(rows.render()).splitlines() if "MEM" in ln)

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        for name in ("_set_headline", "_set_body"):
            key = name.rsplit("_", 1)[-1]
            monkeypatch.setattr(
                ResourceDetailScreen, name, lambda s, t, _k=key: cap.__setitem__(_k, t)
            )
        monkeypatch.setattr(ResourceDetailScreen, "_set_figure", lambda s, *a, **k: None)
        monkeypatch.setattr(ResourceDetailScreen, "_render_chart", lambda s, *a, **k: None)
        screen._refresh_mem(snap, SlurmwatchConfig())
        return row + "\n" + _plain("\n".join(cap.values()))

    @pytest.mark.parametrize(("peak", "ws_peak"), [(0, 0), (10 * 1024**2, 5 * 1024**2)])
    def test_an_inconsistent_payload_still_cannot_render_a_peak_below_the_reading(
        self, monkeypatch: pytest.MonkeyPatch, peak: int, ws_peak: int
    ) -> None:
        """The collector enforces peak >= current, but the RENDERERS trusted it — so a
        snapshot from anywhere else could contradict itself on screen. `from_dict` is
        such a path, and it is how the node switcher displays another node's frames.
        Rendered before this: `40 / 200 MiB · peak 0.0 B` beside `peak working set seen
        0.0 B · total now 39.6 MiB`, which the report called worse than the
        useless-but-consistent figure it replaced."""
        used = int(39.6 * 1024**2)
        out = self._rendered(monkeypatch, 200 * 1024**2, used, peak=peak, ws_peak=ws_peak)
        assert "peak 0.0 B" not in out and "peak 0 B" not in out, out
        assert "peak working set seen 0.0 B" not in out, out
        assert "peak this job (lifetime) 0.0 B" not in out, out
        # Every peak on screen reads at or above the live figure.
        assert "peak 40 MiB" in out, out
        assert "peak working set seen 39.6 MiB" in out, out

    def test_the_gap_note_does_not_invent_a_cache_figure_from_a_bad_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It used to subtract the raw fields, so an inconsistent pair produced a
        confident "incl. 5.0 MiB page cache" out of nonsense."""
        out = self._rendered(
            monkeypatch,
            200 * 1024**2,
            int(39.6 * 1024**2),
            peak=10 * 1024**2,
            ws_peak=5 * 1024**2,
        )
        assert "page cache" not in out, out

    def test_the_gap_is_measured_from_the_figures_on_screen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case where the raw and shown working-set peaks diverge AND the note
        still fires: an older payload with no ws-peak field, but a known working set.
        Subtracting the raw 0 would claim the entire 504 MiB peak is cache, next to a
        working-set line reading 326.8 MiB — the note must agree with what is beside
        it."""
        out = self._rendered(
            monkeypatch,
            800 * 1024**2,
            int(326.8 * 1024**2),
            peak=int(504 * 1024**2),
            ws_peak=0,
        )
        assert "peak working set seen 326.8 MiB" in out, out
        assert "incl. 177.2 MiB" in out, out
        assert "incl. 504" not in out, "the gap cannot exceed the peak's own excess"

    @pytest.mark.parametrize(
        ("limit_mib", "used_mib"),
        [(196, 35), (200, 36), (100, 18), (512, 92), (1024, 184), (65536, 11796)],
    )
    def test_no_surface_shows_a_zero_peak_beside_a_live_figure(
        self, monkeypatch: pytest.MonkeyPatch, limit_mib: int, used_mib: int
    ) -> None:
        import re as _re

        out = self._rendered(monkeypatch, limit_mib * 1024**2, used_mib * 1024**2)
        assert "0 / 0" not in out, out
        # A standalone zero figure, not the "4.0 GiB" of a legitimate one.
        zero = _re.search(r"(?<![\d.])0(?:\.0)? (?:GiB|MiB|KiB|B)\b", out)
        assert zero is None, f"{zero.group(0)!r} in:\n{out}"

    def test_the_peak_figures_read_above_the_current_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Numerically, not just textually: parse what the card prints."""
        import re as _re

        out = self._rendered(monkeypatch, 196 * 1024**2, 35 * 1024**2)
        units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}

        def _val(label: str) -> float:
            m = _re.search(rf"{label}\s+([\d.]+)\s+(B|KiB|MiB|GiB)", out)
            assert m, f"{label} not found in:\n{out}"
            return float(m.group(1)) * units[m.group(2)]

        assert _val("peak working set seen") >= _val("total now")
        assert _val("peak this job \\(lifetime\\)") >= _val("total now")


class TestOffNodeAlarmSpeaksInThePastTense:
    """SW-15: the OOM guard now fires off-node, where the figure is sstat's MaxRSS —
    a high-water mark. The alarm is honest only while every sentence about it says
    so; "working set IS 96% of the limit" would assert a present reading nothing
    measured."""

    def _card(self, monkeypatch: pytest.MonkeyPatch, *, remote: bool) -> str:
        from slurmwatch.tui import ResourceDetailScreen

        snap = _make_snapshot()
        snap.remote = remote
        snap.memory.limit_bytes = 200 * 1024**3
        snap.memory.current_bytes = 190 * 1024**3
        snap.memory.working_set_bytes = 190 * 1024**3
        snap.memory.working_set_percent = 95.0
        snap.memory.oom_guard_warning = True
        snap.memory.oom_guard_critical = True
        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        for name in ("_set_headline", "_set_body"):
            key = name.rsplit("_", 1)[-1]
            monkeypatch.setattr(
                ResourceDetailScreen, name, lambda s, t, _k=key: cap.__setitem__(_k, t)
            )
        monkeypatch.setattr(ResourceDetailScreen, "_set_figure", lambda s, *a, **k: None)
        monkeypatch.setattr(ResourceDetailScreen, "_render_chart", lambda s, *a, **k: None)
        screen._refresh_mem(snap, SlurmwatchConfig())
        return _plain("\n".join(cap.values()))

    def test_off_node_says_the_peak_reached_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = self._card(monkeypatch, remote=True)
        assert "peak working set reached 95% of the limit" in card, card
        assert "working set is 95%" not in card

    def test_on_node_still_speaks_in_the_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        card = self._card(monkeypatch, remote=False)
        assert "working set is 95% of the limit" in card, card

    def test_off_node_the_raise_mem_advice_is_not_the_last_word(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Off-node the guard fires on a per-process RSS SUM, which can overstate.

        Measured live: sstat MaxRSS read 1.88x the same job's real working set and
        13% above its cache-inclusive cgroup peak, firing the warning on a job at
        half its limit. "A higher --mem would cut the OOM-kill risk" is then advice
        to buy memory the job may not need, so the card has to say what to check.
        """
        card = self._card(monkeypatch, remote=True)
        assert "--mem" in card
        assert "Confirm on the node" in card, card
        assert "sums shared pages" in card, card

    def test_on_node_the_advice_carries_no_such_caveat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On-node the figure IS the cgroup's, so hedging it would be noise."""
        card = self._card(monkeypatch, remote=False)
        assert "--mem" in card
        assert "Confirm on the node" not in card


class TestUnmeasuredCacheIsNotZero:
    """SW-3: off-node there is no cache breakdown at all, so rendering the 0 as
    `0.0 B` claimed the job holds no page cache — a measurement nobody took."""

    def _detail_body(self, *, cache_measured: bool) -> str:
        snap = _make_snapshot()
        snap.memory.cache_bytes = 0
        snap.memory.cache_measured = cache_measured
        from slurmwatch.tui import _cache_reading

        return _cache_reading(snap.memory)

    def test_says_not_measured_when_nothing_measured_it(self) -> None:
        assert self._detail_body(cache_measured=False) == "not measured"

    def test_a_real_zero_still_reads_as_a_measurement(self) -> None:
        assert self._detail_body(cache_measured=True) == "0 B"


class TestSmallMemoryGauge:
    """SW-4: every `--mem` under 512 MiB rendered `0 / 0 GiB` — a zero-byte limit,
    zero used — beside a bar reading 18%. Array tasks, preprocessing steps and eval
    jobs all live down there, so the gauge was only usable for tens-of-GiB jobs."""

    def _mem_row(self, limit_bytes: int, pct: float, width: int = 150) -> str:
        r = _SizedRows(width)
        snap = _make_snapshot()
        used = int(limit_bytes * pct / 100)
        snap.memory.limit_bytes = limit_bytes
        snap.memory.current_bytes = used
        snap.memory.working_set_bytes = used
        snap.memory.peak_working_set_bytes = int(used * 1.1)
        snap.memory.peak_bytes = int(used * 1.1)
        snap.gpus = []
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        return next(ln for ln in _plain(r.render()).splitlines() if "MEM" in ln)

    @pytest.mark.parametrize("mib", [100, 200, 256, 512])
    def test_a_small_limit_is_never_rendered_as_zero(self, mib: int) -> None:
        row = self._mem_row(mib * 1024**2, 18.0)
        assert "0 / 0" not in row, row
        assert f"{mib} MiB" in row, row
        assert "18%" in row

    def test_the_used_figure_is_never_zero_while_the_bar_is_not(self) -> None:
        row = self._mem_row(200 * 1024**2, 18.0)
        assert "36 / 200 MiB" in row, row

    def test_the_peak_can_never_sit_below_the_live_reading(self) -> None:
        """`peak 0 GiB` next to a live MiB figure was the same rounding, and it
        contradicted itself rather than just being coarse."""
        row = self._mem_row(200 * 1024**2, 18.0)
        assert "peak 40 MiB" in row, row

    def test_the_familiar_gib_rendering_is_untouched(self) -> None:
        """The fix must not churn the tens-of-GiB case this row was designed for."""
        assert "26 / 51 GiB" in self._mem_row(51 * 1024**3, 51.0)
        assert "12 / 64 GiB" in self._mem_row(64 * 1024**3, 18.0)

    def test_a_tiny_reading_under_a_large_limit_keeps_its_own_unit(self) -> None:
        """A job minutes into a 64 GiB allocation: 20 MiB resident must not read 0."""
        row = self._mem_row(64 * 1024**3, 0.03)
        assert "19.7 MiB / 64 GiB" in row, row

    def test_pair_helper_shares_a_unit_only_when_both_fit_it(self) -> None:
        from slurmwatch.tui import _mem_pair

        assert _mem_pair(26 * 1024**3, 51 * 1024**3) == ("26", "51 GiB")
        assert _mem_pair(36 * 1024**2, 200 * 1024**2) == ("36", "200 MiB")
        assert _mem_pair(184 * 1024**2, 1024**3) == ("184.0 MiB", "1.0 GiB")
        assert _mem_pair(0, 200 * 1024**2) == ("0 B", "200 MiB")


class TestFabricRow:
    """The inter-node fabric as its own RESOURCES row.

    For distributed training this is usually the resource that explains a slow
    step, and it is readable from sysfs whether or not the GPUs are — so a tag on
    the GPU row would vanish in exactly the case (GPUs unreadable) where it is the
    only live number left. A row also always fits, which a suffix on an
    already-long line does not.
    """

    FAB = NodeFabric(
        ports=1,
        link_rate_gbps=100.0,
        kind="InfiniBand",
        rate_label="100 Gb/sec (2X HDR)",
        rx_gbps=93.6,
        tx_gbps=94.6,
        rates_known=True,
    )

    def _render(
        self, *, node_count: int, gpus: bool, fab: NodeFabric | None, width: int = 150
    ) -> str:
        r = _SizedRows(width)
        snap = _make_snapshot()
        snap.node_count = node_count
        snap.fabric = fab
        if gpus:
            snap.gpu_monitoring_available = True
            snap.gpus = [_make_gpu(100.0, 38 * 1024**3, 40 * 1024**3, index=1)]
        else:
            snap.gpus = []
            snap.gpu_count_requested = 2
            snap.gpu_monitoring_available = False
            snap.gpu_unavailable_reason = "devices_denied"
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        return _plain(r.render())

    def _net(self, **kw: object) -> str:
        out = self._render(**kw)  # type: ignore[arg-type]
        return next(ln for ln in out.splitlines() if "NET" in ln)

    def test_row_appears_even_when_the_gpus_are_unreadable(self) -> None:
        """Precisely the case a suffix on the GPU row would have hidden it in."""
        net = self._net(node_count=2, gpus=False, fab=self.FAB)
        assert "94.6" in net and "93.6" in net
        assert "Gb/s" in net and "IB" in net

    def test_row_appears_with_readable_gpus_too(self) -> None:
        assert "IB" in self._net(node_count=2, gpus=True, fab=self.FAB)

    def test_reports_share_of_the_link(self) -> None:
        """ "Is the all-reduce saturating the fabric" must answer at a glance."""
        net = self._net(node_count=2, gpus=True, fab=self.FAB)
        assert "95%" in net  # 94.6 of 100
        assert "100 Gb/s link" in net

    def test_names_which_network_it_measures(self) -> None:
        """``NET`` alone doesn't say which network — the GPU row carries an
        ``↑ ↓ GB/s`` pair too (its own in-node fabric), so the row states that this
        one is the inter-node link, and states it FIRST."""
        net = self._net(node_count=2, gpus=True, fab=self.FAB)
        assert "inter-node IB" in net
        assert net.index("inter-node") < net.index("94.6"), net

    def test_leaves_the_node_wide_caveat_to_the_drill_in(self) -> None:
        """The counters ARE the host's, but naming the fabric already tells the
        reader what the row measures, so the dashboard row does not spend a
        parenthetical on a caveat that only bites on a shared node — the drill-in
        states it in full (TestNodeFabricLines)."""
        net = self._net(node_count=2, gpus=True, fab=self.FAB)
        assert "node-wide" not in net, net
        assert "(" not in net, net

    def test_the_first_cut_is_the_links_spec_not_the_name(self) -> None:
        """The full row fits the 80 columns an SSH session gives; tighter than that,
        the first thing dropped is the link's SPEC ("100 Gb/s"), which the drill-in
        prints in full — never which network this is or how full it is."""
        assert "95% of 100 Gb/s link" in self._net(node_count=2, gpus=True, fab=self.FAB, width=80)
        net = self._net(node_count=2, gpus=True, fab=self.FAB, width=70)
        assert len(net) <= 70, f"{len(net)}: {net}"
        assert "inter-node IB" in net, net
        assert "95% of link" in net, net
        assert "100 Gb/s link" not in net, net

    WIDEST = NodeFabric(
        ports=8,
        link_rate_gbps=400.0,
        kind="RoCE",
        rate_label="400 Gb/sec (4X NDR)",
        rx_gbps=3198.7,
        tx_gbps=3199.4,
        rates_known=True,
    )

    def test_the_share_is_measured_against_the_whole_node(self) -> None:
        """Found by audit, not by the report: rx/tx are SUMMED across every active
        port, so dividing them by ONE port's rate reported "180% of 100 Gb/s link" on
        a busy 2-HCA node — a share above 100% of a stated ceiling cannot be true,
        and it is the multi-HCA sites this has never run on that see it."""
        fab = NodeFabric(
            ports=2,
            link_rate_gbps=100.0,
            link_rate_total_gbps=200.0,
            kind="InfiniBand",
            rx_gbps=180.0,
            tx_gbps=170.0,
            rates_known=True,
        )
        net = self._net(node_count=2, gpus=True, fab=fab)
        assert "90%" in net, net  # 180 of 200, not 180% of 100
        assert "180%" not in net
        # The ceiling is named, so the arithmetic is checkable from the row itself.
        assert "2 × 100 Gb/s link" in net, net

    def test_a_single_port_node_reads_exactly_as_before(self) -> None:
        net = self._net(node_count=2, gpus=True, fab=self.FAB)
        assert "95% of 100 Gb/s link" in net, net
        assert "1 ×" not in net, "a single HCA needs no multiplier"

    def test_an_unknown_aggregate_falls_back_to_the_port_rate(self) -> None:
        """A snapshot from an older slurmwatch (node switcher, or a --log replay)
        carries no aggregate; the per-port rate is still better than no share."""
        fab = NodeFabric(
            ports=1,
            link_rate_gbps=100.0,
            kind="InfiniBand",
            rx_gbps=50.0,
            tx_gbps=10.0,
            rates_known=True,
        )
        assert "50%" in self._net(node_count=2, gpus=True, fab=fab)

    def test_widest_row_still_fits_an_80_column_terminal(self) -> None:
        """A node with several HCAs sums their rates while the ceiling stays ONE
        port's, so the figures can reach four digits and the share can pass 100% —
        the widest this row ever gets, and it still names its fabric at 80 columns."""
        net = self._net(node_count=2, gpus=True, fab=self.WIDEST, width=80)
        assert len(net) <= 80, f"{len(net)}: {net}"
        assert "3199.4" in net and "3198.7" in net
        assert "800% of link" in net, net
        assert "inter-node RoCE" in net, net

    def test_a_terminal_too_narrow_for_that_drops_the_word_not_the_numbers(self) -> None:
        """The last cut: below the width an SSH session gives, "inter-node" goes and
        the bare fabric kind carries the identity — the live figures and the share
        stay. A wrapped row would push a whole GPU block out of a panel that clips."""
        net = self._net(node_count=2, gpus=True, fab=self.WIDEST, width=70)
        assert len(net) <= 70, f"{len(net)}: {net}"
        assert "inter-node" not in net, net
        assert "RoCE" in net and "3199.4" in net and "800% of link" in net, net

    def test_hidden_for_a_single_node_job(self) -> None:
        out = self._render(node_count=1, gpus=True, fab=self.FAB)
        assert not any("NET" in ln for ln in out.splitlines())

    def test_hidden_before_a_rate_is_known(self) -> None:
        """A first frame must not imply an idle network."""
        fab = NodeFabric(ports=1, link_rate_gbps=100.0, kind="InfiniBand")
        out = self._render(node_count=2, gpus=True, fab=fab)
        assert not any("NET" in ln for ln in out.splitlines())

    def test_hidden_without_an_hca(self) -> None:
        out = self._render(node_count=2, gpus=True, fab=None)
        assert not any("NET" in ln for ln in out.splitlines())

    def test_row_fits_every_terminal_width(self) -> None:
        """Asserting on WIDTH: a substring check passes on an overflowed line too."""
        for width in (80, 100, 119, 150):
            out = self._render(node_count=2, gpus=True, fab=self.FAB, width=width)
            for ln in out.splitlines():
                assert len(ln) <= width, f"{width}: {len(ln)} -> {ln}"

    def test_ascii_row_is_pure(self) -> None:
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.node_count = 2
        snap.fabric = self.FAB
        snap.gpu_monitoring_available = True
        snap.gpus = [_make_gpu(100.0, 38 * 1024**3, 40 * 1024**3, index=1)]
        r.snapshot = snap
        r.config = SlurmwatchConfig(ascii_mode=True)
        out = r.render()
        for glyph in ("\u2191", "\u2192", "\u00b7", "\u25cf"):
            assert glyph not in out, glyph

    def test_gpu_head_still_marks_its_own_traffic_in_node(self) -> None:
        """The PCIe figure keeps its scope label even though the row moved out."""
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.node_count = 2
        snap.gpu_monitoring_available = True
        snap.gpus = [
            _make_gpu(100.0, 38 * 1024**3, 40 * 1024**3, index=1),
            _make_gpu(100.0, 38 * 1024**3, 40 * 1024**3, index=2),
        ]
        snap.interconnect = GpuInterconnect(
            fabric="pcie",
            devices=[1, 2],
            matrix=[["self", "SYS"], ["SYS", "self"]],
            nvlink_rx_gbps=[],
            nvlink_tx_gbps=[],
            pcie_rx_gbps=[0.027, 0.03],
            pcie_tx_gbps=[0.007, 0.007],
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        head = next(ln for ln in _plain(r.render()).splitlines() if "device" in ln)
        assert "in-node" in head


class TestMemoryDrillInPeaks:
    """Both memory drill-in branches must report the LIFETIME peak, not just the
    window max — a job with no enforced limit still has to be sized for next time."""

    @staticmethod
    def _screen(monkeypatch: pytest.MonkeyPatch, limit: int) -> dict[str, str]:
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        snap = _make_snapshot()
        snap.memory = MemoryMetrics(
            **{
                **{
                    f.name: getattr(snap.memory, f.name)
                    for f in __import__("dataclasses").fields(MemoryMetrics)
                },
                "limit_bytes": limit,
                "current_bytes": 6 * 1024**3,
                "working_set_bytes": 5 * 1024**3,
                "peak_bytes": 28 * 1024**3,
                "peak_working_set_bytes": 5 * 1024**3,
                "cache_bytes": 1024**3,
            }
        )

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        for name in ("_set_headline", "_set_body"):
            key = name.rsplit("_", 1)[-1]
            monkeypatch.setattr(
                ResourceDetailScreen,
                name,
                lambda s, t, _k=key: cap.__setitem__(_k, t),
            )
        monkeypatch.setattr(ResourceDetailScreen, "_set_figure", lambda s, *a, **k: None)
        monkeypatch.setattr(ResourceDetailScreen, "_render_chart", lambda s, *a, **k: None)
        screen._refresh_mem(snap, SlurmwatchConfig())
        return cap

    def test_with_a_limit_reports_both_peaks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = _plain(self._screen(monkeypatch, 51 * 1024**3)["body"])
        assert "peak this job (lifetime)" in body and "28" in body
        assert "peak working set seen" in body

    def test_without_a_limit_also_reports_the_lifetime_peak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This branch showed ONLY the window max, so a late attach under-reported."""
        body = _plain(self._screen(monkeypatch, 0)["body"])
        assert "peak this job (lifetime)" in body
        assert "28" in body, body
        assert "peak working set seen" in body


class TestResourceRowOrder:
    """Glance-level rows must survive clipping; per-device detail is what gets cut.

    The RESOURCES panel does not scroll — it clips. The per-device GPU blocks run
    three rows each, so on an 8-GPU node they are ~24 rows and anything emitted
    after them is off-screen on any ordinary terminal.
    """

    def _lines(self, gpu_count: int) -> list[str]:
        r = _SizedRows(150)
        snap = _make_snapshot()
        snap.node_count = 2
        snap.gpu_monitoring_available = True
        snap.gpus = [
            _make_gpu(100.0, 38 * 1024**3, 40 * 1024**3, index=i) for i in range(gpu_count)
        ]
        snap.fabric = NodeFabric(
            ports=1,
            link_rate_gbps=100.0,
            kind="InfiniBand",
            rx_gbps=50.0,
            tx_gbps=40.0,
            rates_known=True,
        )
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        return [ln for ln in _plain(r.render()).splitlines() if ln.strip()]

    def test_net_row_precedes_the_per_device_blocks(self) -> None:
        lines = self._lines(2)
        net = next(i for i, ln in enumerate(lines) if "NET" in ln)
        first_device = next(i for i, ln in enumerate(lines) if "CUDA" in ln)
        assert net < first_device, lines

    def test_net_row_survives_an_eight_gpu_node(self) -> None:
        """The shape that pushed it off-screen: ~24 rows of device detail."""
        lines = self._lines(8)
        net = next(i for i, ln in enumerate(lines) if "NET" in ln)
        # Comfortably inside the first screen, not buried under the device blocks.
        assert net <= 3, f"NET at row {net}: {lines[:6]}"

    def test_single_line_rows_stay_contiguous(self) -> None:
        """CPU / MEM / NET / GPU-head read as one glance block, detail after."""
        lines = self._lines(4)
        order = [
            i for i, ln in enumerate(lines) if any(k in ln for k in ("CPU", "MEM", "NET", "GPU"))
        ]
        assert order == sorted(order)
        assert max(order) < next(i for i, ln in enumerate(lines) if "CUDA" in ln)


class TestThePeakSaysWhichPeakItIs:
    """Round 35: one job's headline peak read 307.9 MiB off-node and 504.0 MiB
    on-node — a 64% spread decided by where slurmwatch ran — because the cgroup's
    lifetime counter retains the page cache resident at the high-water mark while
    `cache_bytes` reports the cache resident NOW, which had been reclaimed. So the
    row could show `peak 504 MiB` beside `cache 0 MiB` and account for none of the
    177 MiB a reader was about to size --mem against."""

    MIB = 1024**2

    def _snap(
        self,
        peak_mib: float,
        ws_peak_mib: float,
        *,
        measured: bool = True,
        lifetime: bool = True,
    ) -> Any:
        snap = _make_snapshot()
        m = snap.memory
        m.limit_bytes = 800 * self.MIB
        m.current_bytes = m.working_set_bytes = 300 * self.MIB
        m.peak_bytes = int(peak_mib * self.MIB)
        m.peak_working_set_bytes = int(ws_peak_mib * self.MIB)
        m.cache_bytes = 0
        m.cache_measured = measured
        m.peak_is_lifetime = lifetime
        m.working_set_percent = 37.5
        snap.gpus = []
        return snap

    def _row(self, snap: Any) -> str:
        r = _SizedRows(150)
        r.snapshot = snap
        r.config = SlurmwatchConfig()
        return next(ln for ln in _plain(r.render()).splitlines() if "MEM" in ln)

    def test_the_row_marks_a_peak_that_is_not_the_sizing_figure(self) -> None:
        assert "peak 504 MiB (lifetime)" in self._row(self._snap(504, 326.8))

    def test_it_does_not_claim_the_gap_is_cache(self) -> None:
        """Measured on a long-lived job here, the gap was 31.4 GiB and almost all of
        it was growth from before the session started watching — not cache. Saying
        "cache-incl." would invite discounting the wrong thing."""
        row = self._row(self._snap(504, 326.8))
        assert "cache" not in row

    def test_no_label_when_the_two_peaks_agree(self) -> None:
        """No noise on a job where there is nothing to explain."""
        assert "(lifetime)" not in self._row(self._snap(330, 326.8))

    def test_no_label_off_node_where_the_peak_excludes_cache(self) -> None:
        """MaxRSS is cache-excluded, so the on-node caveat would be a lie there."""
        assert "(lifetime)" not in self._row(self._snap(307.9, 307.9, measured=False))

    def test_the_drill_in_quantifies_the_gap_and_names_the_causes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        for name in ("_set_headline", "_set_body"):
            key = name.rsplit("_", 1)[-1]
            monkeypatch.setattr(
                ResourceDetailScreen, name, lambda s, t, _k=key: cap.__setitem__(_k, t)
            )
        monkeypatch.setattr(ResourceDetailScreen, "_set_figure", lambda s, *a, **k: None)
        monkeypatch.setattr(ResourceDetailScreen, "_render_chart", lambda s, *a, **k: None)
        screen._refresh_mem(self._snap(504, 326.8), SlurmwatchConfig())
        body = _plain(cap["body"])
        assert "177.2 MiB" in body, body
        assert "page cache or growth from before this session" in body
        assert "size --mem from the working set" in body, "say which figure to use"
        assert "peak working set seen 326.8 MiB" in body

    def test_a_running_max_is_not_called_a_lifetime_peak(self) -> None:
        """cgroup v2 gained memory.peak in kernel 5.19; RHEL/Rocky 9 ships 5.14, so on
        a large share of clusters `peak_bytes` is a running max slurmwatch keeps since
        it attached. Round 35's label called that "lifetime", which is false: the
        figure has no pre-session history and a restart resets it."""
        row = self._row(self._snap(504, 326.8, lifetime=False))
        assert "peak 504 MiB (cache-incl.)" in row, row
        assert "lifetime" not in row

    def test_and_its_gap_is_not_blamed_on_pre_session_growth(self) -> None:
        """A running max cannot contain anything from before it started running."""
        from slurmwatch.tui import _peak_gap_note

        note = _plain(_peak_gap_note(self._snap(504, 326.8, lifetime=False).memory))
        assert "page cache" in note
        assert "before this session" not in note, note
        assert "size --mem from the working set" in note

    def test_the_drill_in_heading_says_which_peak_it_holds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch.tui import ResourceDetailScreen

        def _body(snap: Any) -> str:
            screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

            class _Dash:
                mem_history: list[float] = []

            screen._dashboard = _Dash()  # type: ignore[assignment]
            cap: dict[str, str] = {}
            monkeypatch.setattr(
                ResourceDetailScreen, "_set_body", lambda s, t: cap.__setitem__("b", t)
            )
            for name in ("_set_headline", "_set_figure", "_render_chart"):
                monkeypatch.setattr(ResourceDetailScreen, name, lambda s, *a, **k: None)
            screen._refresh_mem(snap, SlurmwatchConfig())
            return _plain(cap["b"])

        assert "peak this job (lifetime)" in _body(self._snap(504, 326.8))
        assert "peak this job (this session)" in _body(self._snap(504, 326.8, lifetime=False))

    def test_the_gap_note_is_absent_when_there_is_no_gap(self) -> None:
        from slurmwatch.tui import _peak_gap_note

        assert _peak_gap_note(self._snap(330, 326.8).memory) == ""

    def test_the_cache_figure_says_it_is_the_reading_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of round 35: `cache 0` beside a cache-inclusive peak read as
        "no cache to discount" when it means "none resident right now" — the cache at
        the high-water mark had already been reclaimed."""
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        monkeypatch.setattr(
            ResourceDetailScreen, "_set_body", lambda s, t: cap.__setitem__("body", t)
        )
        for name in ("_set_headline", "_set_figure", "_render_chart"):
            monkeypatch.setattr(ResourceDetailScreen, name, lambda s, *a, **k: None)
        screen._refresh_mem(self._snap(504, 326.8), SlurmwatchConfig())
        assert "reclaimable cache now" in _plain(cap["body"])


class TestAsciiModeCoversTheseViewsToo:
    """`--ascii` exists for terminals that cannot encode the glyphs, so ONE leaked
    character defeats it. The purity test covered a single widget (MonitorNote), which
    is why three strings leaked: the memory drill-in's gap note (added with a hard
    U+2014 and no ascii branch at all), the fabric line's "measuring…", and the job
    selector's empty-list title — in a class whose own comment records it "used to
    hardcode Unicode arrows/dots and leak them under --ascii"."""

    MIB = 1024**2

    def _mem_snap(self) -> Any:
        snap = _make_snapshot()
        m = snap.memory
        m.limit_bytes = 800 * self.MIB
        m.current_bytes = m.working_set_bytes = 300 * self.MIB
        m.peak_bytes = int(504 * self.MIB)
        m.peak_working_set_bytes = int(326.8 * self.MIB)
        m.cache_bytes = 0
        m.cache_measured = True
        m.working_set_percent = 37.5
        snap.gpus = []
        return snap

    def _drill_in_body(self, snap: Any, *, ascii_mode: bool) -> str:
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)

        class _Dash:
            mem_history: list[float] = []

        screen._dashboard = _Dash()  # type: ignore[assignment]
        cap: dict[str, str] = {}
        stubs = (
            ("_set_body", lambda s, t: cap.__setitem__("b", t)),
            ("_set_headline", lambda s, *a, **k: None),
            ("_set_figure", lambda s, *a, **k: None),
            ("_render_chart", lambda s, *a, **k: None),
        )
        # Restored, because these are set on the CLASS and this used to leak: every
        # `ResourceDetailScreen` built LATER in the process kept the no-op
        # `_set_headline`/`_set_body`, so any later test that drove the drill-in through
        # a real Pilot read an empty slot and could only assert absence. Measured: the
        # dashboard-vs-summary tests below pass alone and pass with the first 366 tests
        # of this file, and go red the moment this one runs before them.
        saved = {name: getattr(ResourceDetailScreen, name) for name, _ in stubs}
        try:
            for name, fn in stubs:
                setattr(ResourceDetailScreen, name, fn)
            screen._refresh_mem(snap, SlurmwatchConfig(ascii_mode=ascii_mode))
        finally:
            for name, original in saved.items():
                setattr(ResourceDetailScreen, name, original)
        return _plain(cap["b"])

    @pytest.mark.parametrize("lifetime", [True, False])
    def test_the_memory_drill_in_is_pure_ascii(self, lifetime: bool) -> None:
        snap = self._mem_snap()
        snap.memory.peak_is_lifetime = lifetime
        body = self._drill_in_body(snap, ascii_mode=True)
        assert "177.2 MiB" in body, "the note is still there, just ASCII"
        body.encode("ascii")  # raises if a glyph leaked

    def test_and_still_uses_the_em_dash_without_ascii(self) -> None:
        snap = self._mem_snap()
        snap.memory.peak_is_lifetime = True
        assert "\N{EM DASH}" in self._drill_in_body(snap, ascii_mode=False)

    def test_the_fabric_line_is_pure_ascii_before_any_traffic(self) -> None:
        """The first frame has no delta yet, so this is the line every --ascii user
        sees first."""
        from slurmwatch.model import NodeFabric
        from slurmwatch.tui import _node_fabric_lines

        fabric = NodeFabric(ports=1, kind="InfiniBand", link_rate_gbps=100.0)
        lines = _node_fabric_lines(fabric, 2, True)
        for line in lines:
            _plain(line).encode("ascii")
        joined = " ".join(_plain(x) for x in lines)
        assert "measuring..." in joined, joined
        assert "measuring\N{HORIZONTAL ELLIPSIS}" in " ".join(
            _plain(x) for x in _node_fabric_lines(fabric, 2, False)
        )

    def test_the_selectors_empty_title_is_pure_ascii(self) -> None:
        """Reachable while the picker is open and the last job finishes: the live
        refresh rebuilds this heading."""
        from slurmwatch.tui import _selector_title

        _selector_title(0, True).encode("ascii")
        assert "press q to quit" in _selector_title(0, True)
        assert "\N{EM DASH}" in _selector_title(0, False)
        assert _selector_title(3, True) == "Select a job (3 found):"


class TestCtrlCQuitsLikeQ:
    """Round 45's secondary finding: `grep -c "ctrl+c" tui.py` was 0, so the key was
    bound on no screen — Textual's usual priority binding was not reaching them. A user
    who reaches for Ctrl-C first got an app that ignored them, inside the alternate
    screen, with the footer possibly clipped on a narrow terminal. `q` worked and is
    advertised, but the trap cost one line per screen to remove."""

    def _bindings(self, screen_cls: Any) -> dict[str, str]:
        out: dict[str, str] = {}
        for b in screen_cls.BINDINGS:
            key = getattr(b, "key", None)
            if key:
                out[key] = getattr(b, "action", "")
        return out

    @pytest.mark.parametrize(
        "screen_name",
        [
            "DashboardScreen",
            "PendingScreen",
            "ForeignJobScreen",
            "JobSelectorScreen",
            "ResourceDetailScreen",
        ],
    )
    def test_every_screen_that_can_be_left_binds_ctrl_c(self, screen_name: str) -> None:
        import slurmwatch.tui as tui_mod

        screen_cls = getattr(tui_mod, screen_name)
        binds = self._bindings(screen_cls)
        assert "ctrl+c" in binds, f"{screen_name} ignores Ctrl-C"
        # Same action as the advertised key, so the two cannot diverge.
        advertised = binds.get("q")
        assert advertised, f"{screen_name} has no q binding to match"
        assert binds["ctrl+c"] == advertised, f"{screen_name}: {binds['ctrl+c']} != {advertised}"

    def test_it_is_a_priority_binding_and_not_shown_twice_in_the_footer(self) -> None:
        """priority so it beats a focused widget's own handling; show=False because the
        footer already advertises `q` and two rows for one action is noise."""
        import slurmwatch.tui as tui_mod

        for b in tui_mod.DashboardScreen.BINDINGS:
            if getattr(b, "key", None) == "ctrl+c":
                assert getattr(b, "priority", False) is True
                assert getattr(b, "show", True) is False
                return
        raise AssertionError("no ctrl+c binding found")


class TestThePollLoopAlwaysYields:
    """One suspension point per iteration, whatever branch runs.

    A branch that returned without awaiting starved the event loop, and a starved
    Textual app is not merely slow — it is UNKILLABLE: `q` and Ctrl-C are bytes it
    never reads, and SIGTERM/SIGHUP are queued callbacks that never run, because
    `loop.add_signal_handler` replaced the default disposition (verified directly:
    a starved loop survives SIGHUP). Only SIGKILL clears it. This asserts the
    unconditional yield that makes the mistake unrepeatable here.
    """

    def test_the_loop_body_opens_with_an_unconditional_sleep(self) -> None:
        import inspect

        from slurmwatch.tui import DashboardScreen

        src = inspect.getsource(DashboardScreen._poll_loop)
        body = src.split("while True:", 1)[1]
        lines = (ln.strip() for ln in body.splitlines())
        first = next(ln for ln in lines if ln and not ln.startswith("#"))
        assert first == "await asyncio.sleep(0)", first

    def test_a_starved_loop_really_does_survive_sighup(self) -> None:
        """The claim above, exercised — this is WHY the yield matters.

        If this ever fails, the reasoning in the comment is wrong and the guard can be
        argued about again.
        """
        import signal
        import subprocess
        import sys
        import time

        code = (
            "import asyncio, os, signal, time\n"
            "async def main():\n"
            "    loop = asyncio.get_running_loop()\n"
            "    loop.add_signal_handler(signal.SIGHUP, lambda: os._exit(129))\n"
            "    print('ready', flush=True)\n"
            "    t = time.time()\n"
            "    while time.time() - t < 8: pass\n"
            "asyncio.run(main())\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"
            time.sleep(0.3)
            proc.send_signal(signal.SIGHUP)
            time.sleep(1.5)
            assert proc.poll() is None, "a starved loop DID act on SIGHUP — revisit the guard"
        finally:
            proc.kill()
            proc.wait()


class TestGivingUpOnAStreamDoesNotSpinTheLoop:
    """Giving up must still PACE: the poll loop's remote branch has no sleep of its own.

    Measured live on the second cluster: after the permanent-failure check stopped
    relaunching, `_read_remote` returned immediately every tick, the event loop was
    starved, the spinner glyph froze mid-animation, and the watchdog that was meant to
    display the failure never fired — so giving up cost the very message it exists to
    show. The fix is a sleep, and this is the test that would have caught it.
    """

    @pytest.mark.asyncio
    async def test_the_gave_up_path_awaits_before_returning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio as aio

        from slurmwatch.tui import DashboardScreen

        scr = DashboardScreen.__new__(DashboardScreen)
        scr._stream_gave_up = True
        scr._stream_proc = None
        scr._stream_node = None
        scr.config = SlurmwatchConfig(poll_interval=0.5)
        slept: list[float] = []

        async def _record(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(aio, "sleep", _record)
        assert await scr._read_remote("cn001") is None
        assert slept, "returned without pacing — this is the hot loop"
        assert slept[0] >= 0.5, slept

    @pytest.mark.asyncio
    async def test_a_short_poll_interval_still_paces_at_half_a_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 0.05s interval must not become a 20-per-second relaunch-free spin."""
        import asyncio as aio

        from slurmwatch.tui import DashboardScreen

        scr = DashboardScreen.__new__(DashboardScreen)
        scr._stream_gave_up = True
        scr._stream_proc = None
        scr._stream_node = None
        scr.config = SlurmwatchConfig(poll_interval=0.05)
        slept: list[float] = []

        async def _record(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(aio, "sleep", _record)
        await scr._read_remote("cn001")
        assert slept and slept[0] >= 0.5, slept


class TestTheStuckBannerNamesTheCauseWhenItHasOne:
    """ "It may be busy or unreachable - still retrying" was all this could ever say.

    On a cluster whose `/tmp` is node-local the stream step reported
    `execve(): .../python: No such file or directory` and slurmwatch showed that guess
    instead, indefinitely. The guess is right when nothing was reported; it is wrong
    when the step said something.
    """

    @staticmethod
    def _banner(**over: object) -> str:
        from slurmwatch.tui import SwitchBanner

        b = SwitchBanner()
        b.target_label = "the compute node"
        b.node = "mcn05"
        b.stuck = True
        b.connecting = True
        for k, v in over.items():
            setattr(b, k, v)
        return _plain(b.render())

    def test_with_no_reason_it_keeps_the_honest_guess(self) -> None:
        """A slow controller really does look like this, so do not invent a cause."""
        text = self._banner()
        assert "busy or unreachable" in text
        assert "still retrying" in text

    def test_with_a_reason_it_says_that_instead(self) -> None:
        text = self._banner(
            reason="slurmwatch could not start on mcn05 - this install is not on a "
            "filesystem the compute node can see"
        )
        assert "compute node can see" in text, text
        assert "busy or unreachable" not in text, text
        assert "mcn05" in text

    def test_the_reason_is_escaped_like_every_other_free_text(self) -> None:
        """srun's message is not ours; a `[` in it must not reach the markup parser."""
        text = self._banner(reason="cannot exec /opt/[weird]/python")
        assert "[weird]" in text, text


class TestTheFabricRateOnAnEthernetLink:
    """A RoCE cluster's driver reports an INFINIBAND speed grade for its Ethernet port.

    Measured on a second cluster (Slurm 25.11, 25 GbE RoCE): the kernel's own
    `/sys/class/infiniband/mlx5_bond_0/ports/1/rate` reads `25 Gb/sec (1X EDR)` while
    `link_layer` reads `Ethernet`. slurmwatch quotes that file verbatim — correctly, it
    is the HCA's own words — but printing it beside "inter-node RoCE" tells the reader
    their Ethernet is EDR InfiniBand. The bandwidth means something on either fabric;
    the grade does not.
    """

    @pytest.mark.parametrize(
        ("label", "kind", "expected"),
        [
            ("25 Gb/sec (1X EDR)", "RoCE", "25 Gb/sec"),
            ("10 Gb/sec (1X QDR)", "RoCE", "10 Gb/sec"),
            # InfiniBand keeps its grade: there it is the real encoding, and it is how
            # people talk about the link ("4X HDR").
            ("100 Gb/sec (4X EDR)", "InfiniBand", "100 Gb/sec (4X EDR)"),
            ("200 Gb/sec (2X HDR)", "InfiniBand", "200 Gb/sec (2X HDR)"),
            # Nothing to strip, nothing to invent.
            ("25 Gb/sec", "RoCE", "25 Gb/sec"),
            ("", "RoCE", ""),
            # An unknown link layer is not InfiniBand, so the grade goes.
            ("10 Gb/sec (1X QDR)", "", "10 Gb/sec"),
        ],
    )
    def test_the_grade_is_dropped_only_where_it_does_not_apply(
        self, label: str, kind: str, expected: str
    ) -> None:
        from slurmwatch.units import fabric_rate_text

        assert fabric_rate_text(label, kind) == expected

    def test_the_row_shows_the_cleaned_rate(self) -> None:
        """Through the renderer, not just the helper."""
        from slurmwatch.model import NodeFabric
        from slurmwatch.tui import _node_fabric_lines

        fab = NodeFabric(
            ports=1,
            link_rate_gbps=25.0,
            link_rate_total_gbps=25.0,
            kind="RoCE",
            rate_label="25 Gb/sec (1X EDR)",
        )
        text = _plain("\n".join(_node_fabric_lines(fab, node_count=2, ascii_mode=False)))
        assert "25 Gb/sec" in text
        assert "EDR" not in text, text
        assert "RoCE" in text

    def test_the_payload_keeps_the_drivers_own_words(self) -> None:
        """Only the DISPLAY is cleaned: a machine consumer still gets the raw string."""
        from slurmwatch.model import NodeFabric

        fab = NodeFabric(
            ports=1,
            link_rate_gbps=25.0,
            link_rate_total_gbps=25.0,
            kind="RoCE",
            rate_label="25 Gb/sec (1X EDR)",
        )
        assert fab.rate_label == "25 Gb/sec (1X EDR)"


class TestCorrectingATypedNodeNumber:
    """`action_node_backspace` had ZERO coverage, on the one subsystem in this file
    with a history of stale-frame and banner bugs.

    It is reachable (bound to `backspace`, tui.py) and it is the only way to fix a
    mistyped node number before the 0.9s pause-commit fires — type `19` for `1` on a
    20-node job and without this you are switched to the wrong node.
    """

    @staticmethod
    def _screen(monkeypatch: pytest.MonkeyPatch) -> Any:
        from slurmwatch.tui import DashboardScreen

        scr = DashboardScreen.__new__(DashboardScreen)
        scr._node_input = ""
        scr._node_input_timer = None
        scr._node_list = ["cn001", "cn002", "cn003"]
        shown: list[str] = []
        cleared: list[bool] = []
        timers: list[object] = []

        class _Timer:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        monkeypatch.setattr(
            type(scr), "_show_node_prompt", lambda self: shown.append(self._node_input)
        )

        def _clear(self: Any) -> None:
            cleared.append(True)
            self._node_input = ""

        def _set_timer(self: Any, delay: float, cb: Any) -> _Timer:
            timer = _Timer()
            timers.append(timer)
            return timer

        monkeypatch.setattr(type(scr), "_clear_node_input", _clear)
        monkeypatch.setattr(type(scr), "set_timer", _set_timer)
        return scr, shown, cleared, timers

    def test_backspace_removes_one_digit_and_reprompts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scr, shown, cleared, _ = self._screen(monkeypatch)
        scr._node_input = "12"
        scr.action_node_backspace()
        assert scr._node_input == "1"
        assert shown == ["1"], "the prompt must show the corrected buffer"
        assert not cleared

    def test_the_pause_timer_is_rearmed_from_the_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise an auto-commit scheduled by the EARLIER keystroke fires mid-edit
        and jumps to the number you were in the middle of fixing."""
        scr, _, _, timers = self._screen(monkeypatch)
        scr._node_input = "12"
        first = scr.set_timer(0.9, lambda: None)
        scr._node_input_timer = first
        scr.action_node_backspace()
        assert first.stopped, "the previous pause timer must be cancelled"
        assert scr._node_input_timer is not first, "and a fresh one armed"

    def test_deleting_the_last_digit_clears_the_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scr, shown, cleared, _ = self._screen(monkeypatch)
        scr._node_input = "7"
        scr.action_node_backspace()
        assert scr._node_input == ""
        assert cleared == [True], "an empty buffer hides the prompt and stops the timer"
        assert shown == []

    def test_backspace_on_an_empty_buffer_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scr, shown, cleared, timers = self._screen(monkeypatch)
        scr.action_node_backspace()
        assert (scr._node_input, shown, cleared, timers) == ("", [], [], [])

    def test_the_key_is_actually_bound(self) -> None:
        """An unbound handler is unreachable, and this one is the only correction path."""
        from slurmwatch.tui import DashboardScreen

        bound = {
            b.key: b.action for b in DashboardScreen.BINDINGS if not isinstance(b, tuple) and b.key
        }
        assert bound.get("backspace") == "node_backspace", bound


class TestTheDashboardSurvivesEverySignal:
    """The app handled SIGTERM (slurmstepd sends it when a job is cancelled) so
    Textual could tear the screen down. SIGHUP had nothing, and nothing in Slurm
    sends it — but a tmux/screen pane being killed or an IDE terminal closing SIGHUPs
    the foreground group. Measured on-node in a real pty before the fix: no
    alt-screen exit, ECHO and ICANON left cleared, exit -1; SIGINT and SIGTERM clean
    in the same harness. SW-26's on-node sibling."""

    def _installed(self, monkeypatch: pytest.MonkeyPatch) -> dict[int, Any]:
        import asyncio as aio

        import slurmwatch.tui as tui_mod

        recorded: dict[int, Any] = {}

        class _Loop:
            def add_signal_handler(self, signum: int, cb: Any) -> None:
                recorded[signum] = cb

        monkeypatch.setattr(aio, "get_running_loop", lambda: _Loop())
        app = tui_mod.SlurmwatchApp.__new__(tui_mod.SlurmwatchApp)
        codes: list[int] = []
        monkeypatch.setattr(
            tui_mod.SlurmwatchApp,
            "exit",
            lambda self, return_code=0, **k: codes.append(return_code),
        )
        monkeypatch.setattr(tui_mod.SlurmwatchApp, "register_theme", lambda self, t: None)
        with contextlib.suppress(Exception):
            tui_mod.SlurmwatchApp.on_mount(app)
        self._codes = codes
        return recorded

    def test_sighup_exits_instead_of_dying_mid_draw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handlers = self._installed(monkeypatch)
        assert signal.SIGHUP in handlers, "a closing pane killed the app with the screen up"
        handlers[signal.SIGHUP]()
        assert self._codes == [129], "128+SIGHUP, so a caller still reads 'signalled'"

    def test_sigint_reports_130_rather_than_a_clean_quit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`kill -INT` was indistinguishable from pressing `q`.

        SIGINT was left out of the pair above because it already tore the terminal
        down cleanly — and it does — but it also exited **0**, so anything reading the
        status (`timeout --signal=INT`, a supervisor, the hop's own returncode check)
        read "the dashboard finished" from a run stopped from outside, while its
        siblings reported 143 and 129. Measured in a pty before this: SIGINT 0,
        SIGTERM 143, SIGHUP 129. A ctrl-c TYPED into the dashboard is unaffected — in
        raw mode that arrives as the byte 0x03 and is handled as a key, never as a
        signal, so it still exits 0 (see the ctrl+c binding tests above).
        """
        handlers = self._installed(monkeypatch)
        assert signal.SIGINT in handlers, "kill -INT looked like a clean quit"
        handlers[signal.SIGINT]()
        assert self._codes == [130], "128+SIGINT"

    def test_sigterm_still_reports_143(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The code the login-side hop keys off to say 'the job ended' rather than
        dumping a stale summary — must not change."""
        handlers = self._installed(monkeypatch)
        assert signal.SIGTERM in handlers
        handlers[signal.SIGTERM]()
        assert self._codes == [143]


class TestSimulatedDataSaysSoOnScreen:
    """SW-30's human half. Their argument for the machine payload was that nobody reads
    it; the inverse is just as live — SLURMWATCH_MOCK is a documented equivalent of
    --demo, so a leftover export in a .bashrc, a module file or a wrapper puts fabricated
    figures in front of someone who never typed --demo. And it was not merely an omission:
    the bottom bar claimed `source cgroup`, a FALSE provenance in the one chip that exists
    to say where the numbers came from."""

    @staticmethod
    def _bar(mock: bool) -> str:
        from slurmwatch.model import JobContext
        from slurmwatch.tui import JobInfoBar

        bar = JobInfoBar()
        snap = _make_snapshot()
        snap.mock = mock
        bar.snapshot = snap
        bar.job_ctx = JobContext(
            job_id="12345",
            username="u",
            partition="gpu",
            nodelist="cn1",
            hostname="cn1",
            cpus_allocated=16,
            mem_limit_bytes=64 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
        )
        bar.config = SlurmwatchConfig()
        return _plain(bar.render())

    def test_the_source_chip_does_not_claim_a_cgroup(self) -> None:
        out = self._bar(mock=True)
        assert "source simulated" in out, out
        assert "source cgroup" not in out, "it asserted a provenance it does not have"
        assert "--demo" in out, "name the flag, so the reader knows how to turn it off"

    def test_a_real_snapshot_still_says_cgroup(self) -> None:
        out = self._bar(mock=False)
        assert "source cgroup" in out
        assert "simulated" not in out

    def test_the_header_says_it_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bottom bar can be clipped on a short terminal, and EVERY number on
        screen is fabricated — so the always-visible element carries it as well."""
        import slurmwatch.tui as tui_mod

        captured: dict[str, str] = {}
        monkeypatch.setattr(
            tui_mod,
            "_apply_header",
            lambda screen, brand, body, ascii_mode: captured.update(body=body),
        )
        screen = tui_mod.DashboardScreen.__new__(tui_mod.DashboardScreen)
        screen.job_ctx = self._ctx()
        screen.config = SlurmwatchConfig()
        for mock, expected in ((True, True), (False, False)):
            snap = _make_snapshot()
            snap.mock = mock
            tui_mod.DashboardScreen._update_header(screen, snap)
            assert ("demo data" in captured["body"]) is expected, captured["body"]

    @staticmethod
    def _ctx() -> Any:
        from slurmwatch.model import JobContext

        return JobContext(
            job_id="12345",
            username="u",
            partition="gpu",
            nodelist="cn1",
            hostname="cn1",
            cpus_allocated=16,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )


class TestTheAreaChartGuardsANaNLikeItsSiblings:
    """`_area_chart` was the one of the three bar helpers without the NaN guard.

    `_bar_cells` and `_color_bar` both carry `if math.isnan(...)` and both state the
    reason in a comment -- *"a NaN slips past min/max and crashes round() (inf clamps
    fine)"*. `_area_chart` performs the same `min/max` clamp and feeds the result to
    the same `round()`, and had no guard, so it raised
    `ValueError: cannot convert float NaN to integer` where its two siblings drew an
    empty bar.
    """

    @staticmethod
    def _series(v: float) -> Any:  # a deque; imported in the body below
        from collections import deque

        return deque([10.0, v, 30.0])

    def test_a_nan_no_longer_crashes_the_chart(self) -> None:
        rows = _area_chart(self._series(float("nan")), width=6, height=3)
        assert len(rows) == 3
        assert all(len(r) == 6 for r in rows), rows

    def test_all_three_helpers_agree_that_a_nan_draws_nothing(self) -> None:
        """The family property, not three separate assertions.

        What makes this a drift rather than a missing feature is that the other two
        helpers already answered; a fix that only stopped the crash without matching
        them would leave the same inconsistency one step along.
        """
        nan = float("nan")
        # Stated as "a NaN renders exactly as a zero does", which is the property,
        # rather than as three hand-written expectations about fill characters.
        assert _bar_cells(nan, 10) == _bar_cells(0.0, 10)
        assert _color_bar(nan, 10, True, "green") == _color_bar(0.0, 10, True, "green")
        nan_column = [row[1] for row in _area_chart(self._series(nan), width=3, height=2)]
        zero_column = [row[1] for row in _area_chart(self._series(0.0), width=3, height=2)]
        assert nan_column == zero_column, (nan_column, zero_column)

    def test_infinities_still_clamp_rather_than_being_zeroed(self) -> None:
        """CONTROL -- passes before and after, and is not a mirror of the fix.

        `inf` was never the broken case: it clamps through `min`/`max` perfectly well,
        and the siblings' own comment says so. A guard written as "any non-finite
        becomes 0" would pass the test above and silently turn a pegged series into an
        empty one, so this pins the direction each infinity goes.
        """
        top = _area_chart(self._series(float("inf")), width=3, height=2)
        bottom = _area_chart(self._series(float("-inf")), width=3, height=2)
        assert "█" in [row[1] for row in top], top
        assert [row[1] for row in bottom] == [" ", " "], bottom

    def test_a_normal_series_is_unchanged(self) -> None:
        """CONTROL -- the ordinary path must render exactly as it always did."""
        from collections import deque

        assert _area_chart(deque([100.0] * 6), width=4, height=2) == ["█" * 4] * 2
        assert _area_chart(deque([0.0] * 6), width=4, height=2) == [" " * 4] * 2


# ---------------------------------------------------------------------------
# The live log viewer (LogViewScreen)
# ---------------------------------------------------------------------------


def _log_job(
    tmp_path: Any, out: str = "j.out", err: str | None = "j.err", **over: object
) -> JobContext:
    """A JobContext whose stdout/stderr point at real files under tmp_path.

    `err=None` means Slurm merged the two (its default), which is the common case on
    this cluster and the one where a second key would be noise.
    """
    out_path = str(tmp_path / out) if out else ""
    err_path = out_path if err is None else (str(tmp_path / err) if err else "")
    base: dict[str, object] = {
        "job_id": "56993530",
        "username": "ada",
        "partition": "amd",
        "nodelist": "cn001",
        "hostname": "cn001",
        "cpus_allocated": 8,
        "mem_limit_bytes": 16 * 1024**3,
        "gpu_count_requested": 0,
        "gpu_indices": [],
        "nodelist_resolved": ["cn001"],
        "job_name": "booth-2026-08",
        "raw_job_id": "56993530",
        "std_out": out_path,
        "std_err": err_path,
    }
    base.update(over)
    return JobContext(**base)  # type: ignore[arg-type]


def _log_app(job: JobContext, config: SlurmwatchConfig | None = None) -> _DashApp:
    collector = _StubCollector()
    if config is not None:
        collector.config = config
    return _DashApp(collector, job)


async def _open_log(app: _DashApp, pilot: Any, key: str = "o") -> Any:
    """Open the viewer from the dashboard and let its first read finish.

    The read runs in a worker (off the event loop, so a hung NFS mount cannot
    freeze the UI), so a test has to wait for the worker rather than just pause.
    """
    await pilot.pause()
    await pilot.press(key)
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
    return app.screen


async def _reread(app: _DashApp, pilot: Any) -> None:
    """Force the poll the 0.5 s timer would have done, deterministically."""
    await app.screen._poll_once()  # type: ignore[attr-defined]
    await pilot.pause()


def _log_body(screen: Any) -> str:
    from slurmwatch.tui import _LogPane

    return "\n".join(strip.text for strip in screen.query_one("#logview-log", _LogPane).lines)


def _log_status(screen: Any) -> str:
    return str(screen.query_one("#logview-card").border_subtitle or "")


def _live_line(screen: Any) -> str:
    from textual.widgets import Static

    text = screen.query_one("#logview-live", Static).render_line(0).text
    return str(text).rstrip()


class TestTheLogViewerOpensFromTheDashboard:
    """The JOB card printed the log paths and left the user to go read them elsewhere.

    o / e open the file here and follow it live. Both keys were free: the dashboard
    already owns q, escape, ctrl+c, c, m, g, p, 0-9, enter, backspace, the arrows and
    PgUp/PgDn.
    """

    @pytest.mark.asyncio
    async def test_o_opens_stdout_and_e_opens_stderr(self, tmp_path: Any) -> None:
        (tmp_path / "j.out").write_text("out side\n")
        (tmp_path / "j.err").write_text("err side\n")
        job = _log_job(tmp_path)
        app = _log_app(job)
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot, "o")
            assert isinstance(screen, LogViewScreen)
            assert "out side" in _log_body(screen)
        app = _log_app(job)
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot, "e")
            assert "err side" in _log_body(screen)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["q", "escape"])
    async def test_both_ways_out_return_to_the_dashboard(self, tmp_path: Any, key: str) -> None:
        (tmp_path / "j.out").write_text("hello\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            await _open_log(app, pilot)
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

    @pytest.mark.asyncio
    async def test_the_dashboard_comes_back_exactly_as_it_was(self, tmp_path: Any) -> None:
        # The viewer is PUSHED over the dashboard, never switched for it: the poll
        # loop keeps running, the p collapse state and the selected node survive.
        (tmp_path / "j.out").write_text("hello\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause()
            dash = app.scr
            await pilot.press("p")  # expand the paths first
            await pilot.pause()
            assert dash._paths_full is True
            node, task = dash._selected_node, dash._poll_task
            await _open_log(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is dash
            assert dash._paths_full is True, "the collapse state must survive"
            assert dash._selected_node == node
            assert task is not None
            assert dash._poll_task is task and not task.done(), "polling was interrupted"
            assert dash.query_one(JobDetailsPanel).full_paths is True

    @pytest.mark.asyncio
    async def test_the_keys_are_advertised_where_the_paths_are(self, tmp_path: Any) -> None:
        # Discoverability follows the p hint's established pattern: beside the paths
        # the keys act on, not in the footer.
        panel = JobDetailsPanel()
        panel.job_ctx = _log_job(tmp_path)
        panel.config = SlurmwatchConfig()
        assert "press o / e to follow stdout / stderr" in _plain(panel.render())
        panel.job_ctx = _log_job(tmp_path, err=None)
        assert "press o to follow the output" in _plain(panel.render())

    @pytest.mark.asyncio
    async def test_a_job_with_no_log_path_advertises_nothing_and_opens_nothing(
        self, tmp_path: Any
    ) -> None:
        job = _log_job(tmp_path, out="", err="")
        panel = JobDetailsPanel()
        panel.job_ctx = job
        panel.config = SlurmwatchConfig()
        assert "press o" not in _plain(panel.render())
        app = _log_app(job)
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen), "no path, no screen to push"

    @pytest.mark.asyncio
    async def test_leaving_while_a_read_is_in_flight_does_not_raise(self, tmp_path: Any) -> None:
        # The read runs in a worker; q can land before it returns, and the worker then
        # finds its widgets gone. A log viewer must never take the dashboard down.
        (tmp_path / "j.out").write_text("x\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause()
            await pilot.press("o")
            await pilot.press("q")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

    def test_the_scrollback_cap_stays_clear_of_the_readers_per_read_cap(self) -> None:
        # An invariant, not a preference: RichLog trims by RENDERED ROWS, so a single
        # read's worth of lines (plus wrapped rows, plus our note rows) has to fit
        # under the widget's cap or the burst trims away its own explanation.
        from slurmwatch.logtail import LogTail
        from slurmwatch.tui import _LOG_SCROLLBACK_LINES

        assert LogTail("/nonexistent").max_lines < _LOG_SCROLLBACK_LINES


class TestWhatTheViewerShows:
    @pytest.mark.asyncio
    async def test_a_progress_bar_is_one_line_on_screen_not_thousands(self, tmp_path: Any) -> None:
        # The single biggest difference between a pleasant viewer and an unusable one
        # here: 400 `\r` rewrites of one tqdm line must occupy one row.
        bar = "".join(f"\r{p:3d}%|{'#' * (p // 4):25s}| {p}/100" for p in range(101))
        (tmp_path / "j.out").write_text(f"epoch 1\n{bar}\nepoch 2\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            rows = _log_body(screen).split("\n")
            assert len(rows) == 3, rows[:6]
            assert rows[1].startswith("100%|"), rows[1]

    @pytest.mark.asyncio
    async def test_lines_appended_while_open_appear(self, tmp_path: Any) -> None:
        path = tmp_path / "j.out"
        path.write_text("first\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "second" not in _log_body(screen)
            with path.open("a") as fh:
                fh.write("second\n")
            await _reread(app, pilot)
            body = _log_body(screen)
            assert "second" in body
            assert body.count("first") == 1, "the earlier lines must not be redrawn"

    @pytest.mark.asyncio
    async def test_new_lines_arrive_on_their_own_with_nobody_poking_it(self, tmp_path: Any) -> None:
        """The whole point of the feature: LIVE, not a snapshot.

        Every other test here drives `_poll_once` directly for determinism, which
        would still pass if the interval were never armed. This one spends real
        wall-clock time instead and touches nothing but the file.
        """
        path = tmp_path / "j.out"
        path.write_text("before\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            with path.open("a") as fh:
                fh.write("appeared by itself\n")
            await pilot.pause(tuimod._LOG_POLL_SECONDS + 0.25)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "appeared by itself" in _log_body(screen)

    @pytest.mark.asyncio
    async def test_the_line_still_being_written_has_its_own_row(self, tmp_path: Any) -> None:
        # A half-line is not committed to the scrollback (it would print twice); it
        # sits on the live row until its newline lands, then joins the log.
        path = tmp_path / "j.out"
        path.write_text("committed\n 40%|## | 40/100")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert _log_body(screen) == "committed"
            assert "40/100" in _live_line(screen)
            with path.open("a") as fh:
                fh.write("\r100%|###| 100/100\ndone\n")
            await _reread(app, pilot)
            body = _log_body(screen)
            assert "100/100" in body and "done" in body
            assert body.count("40/100") == 0, "the superseded state must not be kept"
            assert _live_line(screen) == ""

    @pytest.mark.asyncio
    async def test_a_file_larger_than_the_window_shows_only_its_end(self, tmp_path: Any) -> None:
        path = tmp_path / "j.out"
        path.write_text("".join(f"line {i:06d}\n" for i in range(60000)))
        assert path.stat().st_size > 2 * 256 * 1024, "several times the 256 KiB window"
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            body = _log_body(screen)
            assert "line 059999" in body, "the END is what a reader wants"
            assert "line 000000" not in body, "the whole file must not be read in"
            assert "last 256.0 KiB only" in _log_status(screen), _log_status(screen)

    @pytest.mark.asyncio
    async def test_a_burst_bigger_than_the_line_cap_says_what_it_skipped(
        self, tmp_path: Any
    ) -> None:
        # 5000 short lines fit inside the 256 KiB byte window but exceed the 4000-line
        # cap, so the trim is the LINE cap's — and the status must not blame the byte
        # window for it, nor the view silently drop output.
        path = tmp_path / "j.out"
        path.write_text("".join(f"{i}\n" for i in range(5000)))
        assert path.stat().st_size < 256 * 1024, "inside the byte window"
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            body = _log_body(screen)
            assert "1000 earlier lines skipped" in body, body.split("\n")[0]
            assert "4999" in body, "the newest lines are the ones kept"
            assert "end of file only" in _log_status(screen), _log_status(screen)

    @pytest.mark.asyncio
    async def test_scrolling_back_pauses_the_follow_and_end_resumes_it(self, tmp_path: Any) -> None:
        path = tmp_path / "j.out"
        path.write_text("".join(f"line {i:04d}\n" for i in range(400)))
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            log = screen.query_one("#logview-log", tuimod._LogPane)
            assert log.has_focus, "the body owns the scroll keys"
            assert "FOLLOWING" in _log_status(screen)
            await pilot.press("pageup")
            await pilot.pause()
            assert "PAUSED" in _log_status(screen)
            assert "press End to follow" in _log_status(screen)
            parked = log.scroll_y
            with path.open("a") as fh:
                fh.write("APPENDED\n")
            await _reread(app, pilot)
            assert log.scroll_y == pytest.approx(parked), "a paused reader keeps their place"
            assert "APPENDED" in _log_body(screen), "...but the line still arrived"
            await pilot.press("end")
            await pilot.pause()
            assert "FOLLOWING" in _log_status(screen)


class TestEveryWayThereIsNothingToShow:
    """Each of these is a normal state of a Slurm log, and each must SAY so."""

    @pytest.mark.asyncio
    async def test_a_file_that_does_not_exist_yet(self, tmp_path: Any) -> None:
        app = _log_app(_log_job(tmp_path, err=None))  # nothing written to tmp_path
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            body = _log_body(screen)
            assert "no output file" in body
            assert "keeps watching" in body, "it is going to appear; say so"

    @pytest.mark.asyncio
    async def test_and_it_starts_showing_lines_the_moment_the_job_writes(
        self, tmp_path: Any
    ) -> None:
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "no output file" in _log_body(screen)
            (tmp_path / "j.out").write_text("job finally spoke\n")
            await _reread(app, pilot)
            body = _log_body(screen)
            assert body == "job finally spoke", "the notice must be cleared, not kept"

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 000 file")
    async def test_a_file_this_account_cannot_read(self, tmp_path: Any) -> None:
        path = tmp_path / "j.out"
        path.write_text("another user's output\n")
        path.chmod(0)
        app = _log_app(_log_job(tmp_path, err=None))
        try:
            async with app.run_test(size=(110, 30)) as pilot:
                screen = await _open_log(app, pilot)
                body = _log_body(screen)
                assert "permission denied" in body.lower(), body
                assert "another user's output" not in body
        finally:
            path.chmod(0o600)

    @pytest.mark.asyncio
    async def test_an_empty_file_is_not_a_blank_box(self, tmp_path: Any) -> None:
        (tmp_path / "j.out").touch()
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "empty" in _log_body(screen)

    @pytest.mark.asyncio
    async def test_an_unresolvable_slurm_pattern_is_explained_not_guessed(
        self, tmp_path: Any
    ) -> None:
        # `%t` names a file only one of the job's own tasks can identify. Guessing 0
        # would report "no such file" and blame the job for it.
        app = _log_app(_log_job(tmp_path, out="", err="", std_out="/scratch/run-%j-%t.out"))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            body = _log_body(screen)
            assert "%t" in body and "patterns" in body
            assert "keeps watching" not in body, "there is no file to watch"
            assert screen._path == "/scratch/run-56993530-%t.out", "%j still resolved"

    @pytest.mark.asyncio
    async def test_a_file_truncated_under_the_viewer_is_reset_with_a_note(
        self, tmp_path: Any
    ) -> None:
        path = tmp_path / "j.out"
        path.write_text("".join(f"old {i}\n" for i in range(20)))
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "old 19" in _log_body(screen)
            path.write_text("restarted\n")  # size goes backwards
            await _reread(app, pilot)
            body = _log_body(screen)
            assert "old 19" not in body, "another file's content must not linger"
            assert "restarted here" in body and "restarted" in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", ["missing", "empty"])
    async def test_a_notice_is_written_once_and_then_left_alone(
        self, tmp_path: Any, condition: str
    ) -> None:
        """The viewer polls twice a second; a notice must not be a line it appends.

        `_show_message` promises the message is "written once and left alone until it
        changes, so a file that stays missing does not accumulate the same line 120
        times a minute". Every other test here reads the body after ONE read, which
        cannot tell a notice rewritten in place from one that piles up. So drive the
        poll loop the way the timer does -- a call count, not wall clock, so a loaded
        node cannot change the answer -- and require the body to come out identical.
        """
        if condition == "empty":
            (tmp_path / "j.out").touch()
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            first = _log_body(screen)
            assert first.strip(), "the notice itself has to be there to be counted"
            for _ in range(12):  # six seconds' worth of the 0.5 s timer
                await _reread(app, pilot)
            assert _log_body(screen) == first, "the notice was re-appended, not left alone"

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 000 file")
    async def test_each_condition_says_its_own_thing_and_replaces_the_last(
        self, tmp_path: Any
    ) -> None:
        # The other half of the same promise: "nothing here", "the job has written
        # nothing yet" and "this is another user's file" must not read alike, and only
        # the first is a bug. Walking one file through three of them also pins that a
        # notice REPLACES its predecessor instead of stacking under it.
        path = tmp_path / "j.out"
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            seen = [_log_body(screen)]
            path.touch()
            await _reread(app, pilot)
            seen.append(_log_body(screen))
            path.chmod(0)
            try:
                await _reread(app, pilot)
                seen.append(_log_body(screen))
            finally:
                path.chmod(0o600)
        assert "no output file" in seen[0]
        assert "empty" in seen[1]
        assert "permission denied" in seen[2].lower()
        assert len({b.strip() for b in seen}) == 3, seen
        for body in seen[1:]:
            assert "no output file" not in body, "the superseded notice is still on screen"

    @pytest.mark.asyncio
    async def test_a_file_that_vanishes_mid_tail_stops_claiming_its_content(
        self, tmp_path: Any
    ) -> None:
        # A `.out` moved aside while it is being followed. What is on screen is no
        # longer anywhere on disk, so continuing to show it -- and to keep a half-line
        # on the live row as though it were still being written -- is the viewer
        # asserting something false about a file that is gone.
        path = tmp_path / "j.out"
        path.write_text("real output\nhalf-writ")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "real output" in _log_body(screen)
            assert "half-writ" in _live_line(screen)

            path.unlink()
            await _reread(app, pilot)
            body = _log_body(screen)
            assert "real output" not in body, "a deleted file's content must not linger"
            assert "no output file" in body
            assert "keeps watching" in body, "the job may yet write it again"
            assert _live_line(screen) == "", "nothing is being written to a file that is gone"
            assert "FOLLOWING" not in _log_status(screen), "there is nothing to follow"

            path.write_text("second attempt\n")
            await _reread(app, pilot)
            assert _log_body(screen) == "second attempt", "and it picks the new file up"
            assert "FOLLOWING" in _log_status(screen)

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 000 file")
    async def test_a_file_that_becomes_unreadable_mid_tail_says_so(self, tmp_path: Any) -> None:
        # Same shape, different cause: the file is still there and this account can no
        # longer open it. Untested at this level until now -- the permission case was
        # only ever exercised on the FIRST read, where there is no content to clear.
        path = tmp_path / "j.out"
        path.write_text("readable for now\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "readable for now" in _log_body(screen)
            path.chmod(0)
            try:
                await _reread(app, pilot)
                body = _log_body(screen)
            finally:
                path.chmod(0o600)
            assert "permission denied" in body.lower(), body
            assert "readable for now" not in body


class TestTheStatusLineGivesTheRIGHTReasonForATrimmedView:
    """Two independent caps trim the view, and they are not interchangeable claims.

    "last 256 KiB only" says the file was too big to read whole and the reader is
    parked at its end. "end of file only" says a burst outran the per-read line cap.
    A reader deciding whether the start of their log is missing from the FILE or
    merely from this VIEW acts on the difference, so naming the wrong cap is a wrong
    reason for a true fact -- which is what `_status_text`'s own comment says it is
    there to avoid.
    """

    @pytest.mark.asyncio
    async def test_a_line_cap_trim_is_not_re_blamed_on_the_window_once_the_file_grows(
        self, tmp_path: Any
    ) -> None:
        """The reason was re-derived from the CURRENT size, which keeps growing.

        5000 short lines fit inside the 256 KiB window, so the 4000-line cap is what
        trimmed the view and the caption is right at open. The byte window can only
        ever trim the FIRST read of a file, so no later append can make it the
        culprit -- but comparing the live size against the window said otherwise the
        moment the job crossed 256 KiB, and the caption flipped to "last 256.0 KiB
        only" for a view the window had never touched.
        """
        path = tmp_path / "j.out"
        path.write_text("".join(f"{i}\n" for i in range(5000)))
        assert path.stat().st_size < 256 * 1024, "inside the byte window"
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "end of file only" in _log_status(screen), _log_status(screen)

            # One long line is enough to push the FILE past the window; the view is
            # still the one the line cap trimmed.
            with path.open("a") as fh:
                fh.write("z" * 300 * 1024 + "\n")
            await _reread(app, pilot)
            assert path.stat().st_size > 256 * 1024, "the case this test is about"
            status = _log_status(screen)
            assert "end of file only" in status, status
            assert "256.0 KiB only" not in status, status

            # ...and still so when the line cap trims a SECOND time, now that the file
            # is larger than the window.
            with path.open("a") as fh:
                fh.write("".join(f"tail {i}\n" for i in range(5000)))
            await _reread(app, pilot)
            status = _log_status(screen)
            assert "end of file only" in status, status
            assert "256.0 KiB only" not in status, status

    @pytest.mark.asyncio
    async def test_a_windowed_view_keeps_naming_the_window_as_the_job_writes(
        self, tmp_path: Any
    ) -> None:
        """CONTROL -- the window genuinely trimmed, and says so for as long as it holds.

        Passes before and after. Two things at once: it stops the fix from being
        "never mention the window", and it pins the stickiness the field was
        introduced for -- only the first read of a file reports the byte window, so an
        indicator read off the latest chunk alone would announce itself once and then
        vanish while the view was still showing nothing but the end of the file.
        """
        path = tmp_path / "j.out"
        path.write_text("".join(f"line {i:06d}\n" for i in range(60000)))
        assert path.stat().st_size > 2 * 256 * 1024
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "last 256.0 KiB only" in _log_status(screen), _log_status(screen)
            for _ in range(3):
                with path.open("a") as fh:
                    fh.write("still going\n")
                await _reread(app, pilot)
            status = _log_status(screen)
            assert "last 256.0 KiB only" in status, status
            assert "line 000000" not in _log_body(screen)

    @pytest.mark.asyncio
    async def test_a_view_nothing_trimmed_claims_neither(self, tmp_path: Any) -> None:
        """CONTROL -- a whole small file on screen. Passes before and after.

        Neither caption belongs here: the reader is looking at the entire file, and
        either phrase would send them hunting for output already in front of them.
        """
        path = tmp_path / "j.out"
        path.write_text("all\nof\nit\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            with path.open("a") as fh:
                fh.write("and this\n")
            await _reread(app, pilot)
            status = _log_status(screen)
            assert "only" not in status, status
            assert "FOLLOWING" in status


class TestSwitchingFilesFromInsideTheViewer:
    @pytest.mark.asyncio
    async def test_o_and_e_swap_the_file_without_going_back_out(self, tmp_path: Any) -> None:
        (tmp_path / "j.out").write_text("stdout content\n")
        (tmp_path / "j.err").write_text("stderr content\n")
        app = _log_app(_log_job(tmp_path))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot, "o")
            assert "stdout content" in _log_body(screen)
            await pilot.press("e")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen is screen, "still the same screen, not a new one"
            body = _log_body(screen)
            assert "stderr content" in body
            assert "stdout content" not in body
            assert screen._label() == "stderr"  # type: ignore[attr-defined]
            await pilot.press("o")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "stdout content" in _log_body(screen)

    @pytest.mark.asyncio
    async def test_a_merged_log_names_one_file_and_one_key(self, tmp_path: Any) -> None:
        # Slurm merges the streams by default: offering two keys for one file, or
        # labelling it "stdout" when it is both, would be a distinction that isn't there.
        (tmp_path / "j.out").write_text("both streams\n")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert screen._label() == "output"
            caps = [key for key, _label, _color in screen.query_one(KeyFooter)._keys]
            assert caps == ["q", "o", "End"], caps
            await pilot.press("e")  # the other key resolves to the same file
            await pilot.pause()
            assert app.screen is screen
            assert "both streams" in _log_body(screen)


class TestTheViewerHonoursAsciiMode:
    @pytest.mark.asyncio
    async def test_nothing_non_ascii_reaches_an_ascii_terminal(self, tmp_path: Any) -> None:
        # --ascii covers the frames and the live-row marker too, not only the strings
        # this module formats (see TestAsciiModeCoversTheFramesNotJustTheStrings).
        (tmp_path / "j.out").write_text("plain line\n\r 50%|## | 5/10")
        app = _log_app(_log_job(tmp_path, err=None), SlurmwatchConfig(ascii_mode=True))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert "FOLLOWING" in _log_status(screen)
            svg = app.export_screenshot()
        glyphs = _svg_text(svg).replace(" ", " ")
        bad = sorted({c for c in glyphs if not c.isascii()})
        assert not bad, f"non-ascii on screen under --ascii: {bad}"

    @pytest.mark.asyncio
    async def test_the_default_mode_keeps_its_unicode_marker(self, tmp_path: Any) -> None:
        (tmp_path / "j.out").write_text("plain line\n 50%|## | 5/10")
        app = _log_app(_log_job(tmp_path, err=None))
        async with app.run_test(size=(110, 30)) as pilot:
            screen = await _open_log(app, pilot)
            assert _live_line(screen).startswith("\N{BLACK RIGHT-POINTING SMALL TRIANGLE}")


class TestALogIsUntrustedInput:
    def test_ansi_escapes_are_parsed_into_styles_never_emitted(self) -> None:
        # Rich passes a bare ESC straight through to the terminal, so a log carrying
        # \x1b[2J would clear the dashboard around itself. from_ansi turns the
        # sequence into a style instead — the colour survives, the escape does not.
        from slurmwatch.tui import _log_line

        text = _log_line("\x1b[31mFAILED\x1b[0m tests/test_x.py")
        assert "\x1b" not in text.plain
        assert text.plain == "FAILED tests/test_x.py"
        # ...and the colour the author meant survives, on exactly the word it covered.
        assert len(text.spans) == 1, text.spans
        span = text.spans[0]
        assert (span.start, span.end) == (0, len("FAILED"))
        assert getattr(span.style, "color", None) is not None, span.style

    def test_a_clear_screen_sequence_cannot_reach_the_terminal(self) -> None:
        from slurmwatch.tui import _log_line

        assert "\x1b" not in _log_line("before\x1b[2Jafter").plain

    def test_other_control_bytes_are_shown_rather_than_executed_or_dropped(self) -> None:
        from slurmwatch.tui import _log_line

        plain = _log_line("bell\x07null\x00").plain
        assert "\x07" not in plain and "\x00" not in plain
        assert "bell" in plain and "null" in plain

    def test_an_ordinary_line_is_untouched(self) -> None:
        from slurmwatch.tui import _log_line

        assert _log_line("epoch 1 loss 3.21").plain == "epoch 1 loss 3.21"


# ---------------------------------------------------------------------------
# The dashboard and the plain-text summary describe ONE job: do they agree?
# ---------------------------------------------------------------------------
#
# `units.py`'s module docstring records why this pair is the danger: the gauge, the
# memory drill-in and "the plain-text summary the degraded (sstat) path prints" were
# formatted independently, "so a fix to one left the others rounding a 400 MiB limit
# to `0.4 GiB` or `0 / 0 GiB`" (SW-4). That was ONE function; the rest of the two
# surfaces had never been driven from the same bytes and compared.
#
# Both surfaces are reachable on the SAME transport, which is what makes a
# disagreement a user-visible contradiction rather than a curiosity: from a login
# node, `slurmwatch <jobid>` climbs srun -> ssh -> `_print_remote_summary`, while
# `slurmwatch` with no id goes through the job selector, and `SlurmwatchApp._open_job`
# builds a `TelemetryCollector` on the remote JobContext and pushes `DashboardScreen`
# straight onto it -- no hop. So one user sees the dashboard rendering an sstat
# snapshot where the next sees the prose summary rendering the identical one.
#
# Fabricated snapshots, not a live job: a real job's numbers move between the two
# renderings, which would make any difference unattributable.


def _sstat_ctx(**kw: Any) -> JobContext:
    """A job resolved from a LOGIN node — `remote`, i.e. sstat is the transport."""
    base: dict[str, Any] = {
        "job_id": "4711",
        "username": "ada",
        "partition": "gpu",
        "nodelist": "cn042",
        "hostname": "login1",
        "cpus_allocated": 8,
        "mem_limit_bytes": 64 * 1024**3,
        "gpu_count_requested": 0,
        "gpu_indices": [],
        "step_id": "0",
        "uid": 1001,
        "job_start_time": time.time() - 3600,
        "nodelist_resolved": ["cn042"],
        "cgroup_v2_path": "/x",
        "job_state": "RUNNING",
        "job_name": "train",
        "time_limit_seconds": 7200,
        "remote": True,
        "raw_job_id": "4711",
    }
    base.update(kw)
    return JobContext(**base)


def _sstat_snapshot(
    *,
    rss: int,
    limit: int,
    cpu_seconds: float,
    cores: int = 8,
    elapsed: int = 3600,
    sampled: bool = True,
) -> TelemetrySnapshot:
    """The exact shape ``TelemetryCollector._collect_remote`` builds off-node.

    Derived rather than hand-set — mem percent clamped to 100, `current_bytes ==
    peak_bytes == working_set_bytes == peak_working_set_bytes` (all one MaxRSS
    high-water), `cache_measured=False`, `peak_is_lifetime=True`, `source="sstat"` —
    so a divergence found here is a divergence in the RENDERERS and not in a fixture
    that neither renderer would ever be handed.
    """
    mem_pct = min(100.0, rss / limit * 100.0) if limit > 0 else 0.0
    effective = cpu_seconds / elapsed if elapsed > 0 else 0.0
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="4711",
        step_id="0",
        hostname="cn042",
        elapsed_seconds=elapsed,
        cpu=CpuMetrics(
            cores_allocated=cores,
            usage_ns=int(cpu_seconds * 1_000_000_000),
            usage_percent=round(max(0.0, min(100.0, effective / cores * 100.0)), 1),
            effective_cores=round(effective, 1),
            peak_effective_cores=round(effective, 1),
            source="sstat",
        ),
        memory=MemoryMetrics(
            current_bytes=rss,
            limit_bytes=limit,
            peak_bytes=rss,
            usage_percent=round(mem_pct, 1),
            oom_guard_warning=mem_pct >= 80.0,
            oom_guard_critical=mem_pct >= 90.0,
            working_set_bytes=rss,
            cache_bytes=0,
            peak_working_set_bytes=rss,
            working_set_percent=round(mem_pct, 1),
            peak_is_lifetime=True,
            source="sstat",
            cache_measured=False,
        ),
        gpus=[],
        gpu_count_requested=0,
        gpu_active_count=0,
        gpu_monitoring_available=False,
        remote=True,
        usage_sampled=sampled,
        node_count=1,
        partition="gpu",
        job_name="train",
    )


def _plain_summary(
    ctx: JobContext, snap: TelemetrySnapshot, config: SlurmwatchConfig | None = None
) -> str:
    """What the degraded (sstat) path prints for this snapshot."""
    import io
    from contextlib import redirect_stdout

    from slurmwatch import cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_remote_summary(ctx, snap, config or SlurmwatchConfig())
    return buf.getvalue()


async def _dashboard_surfaces(
    ctx: JobContext,
    snap: TelemetrySnapshot,
    config: SlurmwatchConfig,
    drill: str | None = None,
) -> dict[str, str]:
    """The dashboard's rendering of one snapshot: rows, info bar, optional drill-in."""
    collector = _StubCollector()
    collector.config = config
    app = _DashApp(collector, ctx)
    out: dict[str, str] = {}
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        app.scr._update_widgets(snap)
        await pilot.pause()
        out["rows"] = _render_markup(app.scr.query_one(ResourceRows).render()).plain
        out["bar"] = _render_markup(app.scr.query_one(JobInfoBar).render()).plain
        if drill is not None:
            await pilot.press(drill)
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, ResourceDetailScreen)
            scr._refresh()
            await pilot.pause()
            for slot in ("headline", "body"):
                out[slot] = _render_markup(str(scr.query_one(f"#detail-{slot}").render())).plain
    return out


class TestTheDashboardAndTheSummaryAgreeAboutOneJob:
    """Drive both renderings from one fabricated sstat snapshot, figure by figure.

    The figures that matter are the ones carrying a QUALIFIER rather than a number:
    where one surface says "this reading was never taken" and the other prints the
    zero bare, the two are describing different jobs.
    """

    @pytest.fixture(autouse=True)
    def _accounting_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin `acct_gather_disabled()` False for both surfaces.

        It shells out to `scontrol show config` and caches the answer PROCESS-wide,
        so left alone it makes these renders depend on the machine the suite runs on
        — and it is the very branch two of these cases are about.
        """
        from slurmwatch import cli

        monkeypatch.setattr(tuimod, "acct_gather_disabled", lambda: False)
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: False)

    # -- the CONTROL, which must hold in BOTH states --------------------------

    def test_a_fully_measured_job_renders_the_same_figures_on_both_surfaces(self) -> None:
        """26 GiB of a 64 GiB limit, 6.1 of 8 cores — byte-identical, both surfaces.

        This is the case a fix must not touch. The pair, the percent and the core
        figures are asserted as exact strings so a renderer that "improves" one of
        them silently can't leave the other behind (SW-4's whole lesson), and the
        two prose blocks are pinned verbatim.
        """
        ctx, cfg = _sstat_ctx(), SlurmwatchConfig()
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=6.1 * 3600)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="c"))
        summary = _plain_summary(ctx, snap, cfg)

        # The same figures, spelled the same way on both surfaces.
        assert "26 / 64 GiB" in got["rows"] and "26 / 64 GiB (41%)" in summary
        assert "6.1 / 8 cores" in got["rows"]
        assert "~6.1 of 8 cores (avg, running steps)" in summary
        assert "41%" in got["rows"]

        # And the rows themselves, verbatim.
        assert got["rows"] == (
            "  ● CPU     used    █████████████▊░░░░  76%   6.1 / 8 cores   · steady\n"
            "\n"
            "  ● MEM     peak    ███████▎░░░░░░░░░░  41%    26 / 64 GiB   · steady\n"
            "\n"
            "  ● GPU     none requested"
        )
        # A job that is using its cores gets no advisory on either surface, and the
        # transport chip says which reading this is.
        assert got["body"] == ""
        assert "source sstat (peaks, no cache)" in got["bar"]
        assert "Advice" not in summary
        assert summary == (
            "Job 4711  gpu  RUNNING  on cn042  name `train`\n"
            "  Memory   peak 26 / 64 GiB (41%)\n"
            "  CPU      6:06:00 CPU-time  ~6.1 of 8 cores (avg, running steps)\n"
            "  source: sstat — covers only the job's tracked process tree, so a job whose\n"
            "          work runs in detached workers (R multisession/PSOCK, nohup, setsid)\n"
            "          can read far lower than reality on BOTH cpu and memory. Memory can\n"
            "          also read HIGHER than reality: MaxRSS sums each process's RSS, so a\n"
            "          shared page counts once per process. Run on the node (or let --once\n"
            "          hop there) for the true figures; live GPU utilization is on-node "
            "only.\n"
        )

    # -- finding 1: the off-node right-sizing advisory ------------------------

    def test_the_cpu_card_does_not_hand_out_the_off_node_advisory_bare(self) -> None:
        """0.4 of 8 cores, off-node. The summary REFUSES the advice; the card gave it.

        `cpu_is_underused` lives in model.py so "the degraded plain-text summary …
        should reach the same verdict from the same numbers" (SW-18) — and it does.
        But the summary is then forbidden from ACTING on that verdict off-node,
        because sstat sees only the job's tracked process tree: a job whose work runs
        in detached workers (R multisession/PSOCK, nohup, setsid) "reads ~0.1 of 8
        cores while saturating all 8 — measured 79-100x low", so the advisory there
        "would turn a silently wrong number into actively wrong advice, on the one
        path least able to support it" (SW-23, cli.py's own words).

        The drill-in reached the identical verdict from the identical numbers and
        printed the instruction unconditionally. Same treatment the MEM card already
        gives its off-node advice ("Confirm on the node first: this peak sums shared
        pages"): the flag stays, the instruction stops being unqualified.
        """
        ctx, cfg = _sstat_ctx(), SlurmwatchConfig()
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=0.4 * 3600)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="c"))
        summary = _plain_summary(ctx, snap, cfg)

        # Both surfaces reach the verdict from the same numbers.
        assert "0.4 of 8" in got["headline"] and "underused" in got["headline"]
        assert "~0.4 of 8 cores" in summary
        # The summary says nothing about shrinking the request.
        assert "--cpus-per-task" not in summary
        # So the card must not present the instruction as if it were sound here.
        assert "--cpus-per-task" in got["body"], "the flag itself is still worth raising"
        assert "Confirm on the node first" in got["body"], (
            "off-node the advisory rests on a figure measured 79-100x low; the summary "
            "refuses it outright (SW-23) and this card gave it unqualified"
        )
        assert "tracked process tree" in got["body"]

    def test_on_the_node_the_advisory_is_unqualified_on_both_surfaces(self) -> None:
        """The complement: gating on the TRANSPORT, not on the verdict.

        On-node the figure is a cgroup reading and the advice is sound — both surfaces
        give it, and neither hedges. Without this, "never say it" would pass the test
        above while deleting the advisory this tool exists to give.
        """
        ctx = _sstat_ctx(remote=False)
        cfg = SlurmwatchConfig()
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=0.4 * 3600)
        snap.remote = False
        snap.cpu.source = snap.memory.source = "cgroup"
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="c"))
        summary = _plain_summary(ctx, snap, cfg)

        assert "--cpus-per-task" in got["body"] and "--cpus-per-task" in summary
        assert "Confirm on the node first" not in got["body"]
        assert "only ~0.4 of 8 cores are doing work" in got["body"]
        assert "only ~0.4 of 8 cores are doing work" in summary

    # -- finding 2: a frame with no measurement in it -------------------------

    def test_the_source_chip_says_when_slurm_has_sampled_nothing_yet(self) -> None:
        """`usage_sampled=False`: every counter is 0 because nothing was measured.

        The summary refuses to print a single figure and says why — "usage not yet
        sampled by Slurm (samples ~every 30s)". The dashboard renders the same
        snapshot as `0 / 8 cores` and `0 B / 64 GiB` at 0%, and `usage_sampled` was
        referenced NOWHERE in tui.py. The one chip whose job is to say where the
        numbers came from called them "peaks, no cache" — a description of a reading
        that was never taken. SW-20 put that chip there precisely so two materially
        different views would stop looking identical.
        """
        ctx, cfg = _sstat_ctx(), SlurmwatchConfig()
        snap = _sstat_snapshot(rss=0, limit=64 * 1024**3, cpu_seconds=0.0, sampled=False)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))
        summary = _plain_summary(ctx, snap, cfg)

        # The summary prints no figure at all for this frame.
        assert "usage not yet sampled by Slurm" in summary
        assert "Memory   peak" not in summary and "CPU      " not in summary
        # The dashboard does print the zeros, so the chip has to name them.
        assert "0 / 8 cores" in got["rows"] and "0 B / 64 GiB" in got["rows"]
        assert "source sstat (no sample yet)" in got["bar"], (
            "a frame with no measurement in it was labelled 'peaks, no cache'"
        )
        assert "peaks, no cache" not in got["bar"]

    def test_a_cluster_that_gathers_nothing_keeps_its_own_wording(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordering, which is where this could still drift.

        With `JobAcctGatherType=none` the job is ALSO unsampled — permanently — and
        the summary prefers the cluster explanation over "try again shortly", because
        that would be a false promise (`acct_gather_disabled`'s docstring: "the reader
        retries forever for a figure that cannot exist"). The chip has to break the
        tie the same way round.
        """
        from slurmwatch import cli

        ctx, cfg = _sstat_ctx(), SlurmwatchConfig()
        snap = _sstat_snapshot(rss=0, limit=64 * 1024**3, cpu_seconds=0.0, sampled=False)
        monkeypatch.setattr(tuimod, "acct_gather_disabled", lambda: True)
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: True)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))
        summary = _plain_summary(ctx, snap, cfg)

        assert "JobAcctGatherType=none" in summary
        assert "try again shortly" not in summary
        assert "source sstat (gathers nothing on this cluster)" in got["bar"]
        assert "no sample yet" not in got["bar"]

    def test_a_sampled_frame_is_still_called_peaks(self) -> None:
        """The complement for the chip: the ordinary off-node frame is unchanged."""
        ctx, cfg = _sstat_ctx(), SlurmwatchConfig()
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=6.1 * 3600)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))
        assert "source sstat (peaks, no cache)" in got["bar"]
        assert "no sample yet" not in got["bar"]

    # -- finding 3: the memory figure of a job with no --mem at all ----------
    #
    # Two disagreements in one case, pulling opposite ways. The WORD for the figure:
    # `used` on the dashboard, `peak` in the summary. The CAVEAT: `no limit set` on
    # the dashboard, nothing in the summary.
    #
    # What the figure IS settles the word, and it differs by transport, which is
    # exactly what the `limit > 0` branches encode and the no-limit ones did not:
    #   * OFF-node it is `sstat` MaxRSS -- a job-LIFETIME high-water that sums each
    #     process's RSS (a shared page counted once per process), so it can only
    #     climb and is not a reading of "now". `_collect_remote` sets
    #     `current_bytes == peak_bytes == working_set_bytes == peak_working_set_bytes`
    #     from that one number and `peak_is_lifetime=True`.
    #   * ON-node it is the cgroup's CURRENT anonymous working set (`current - cache`),
    #     a live reading; the lifetime figure is carried separately in `peak_bytes`.
    #
    # So the caveat is a fact about the JOB (true on both transports) while the word
    # is a fact about the TRANSPORT (different on each) -- and the fixes go opposite
    # ways: the dashboard was wrong about the word, the summary was silent about the
    # caveat.

    @staticmethod
    def _no_limit(*, remote: bool) -> tuple[JobContext, TelemetrySnapshot]:
        """A job submitted with no ``--mem``: ``limit_bytes == 0`` on every field.

        Shaped as the collector builds it for that job rather than hand-set -- the
        percent is 0.0 because both collect paths guard their division on
        ``limit > 0``, so there is no percentage for either surface to show.
        """
        ctx = _sstat_ctx(mem_limit_bytes=0, remote=remote)
        snap = _sstat_snapshot(rss=int(1.2 * 1024**3), limit=0, cpu_seconds=6.1 * 3600)
        if not remote:
            snap.remote = False
            snap.cpu.source = snap.memory.source = "cgroup"
        return ctx, snap

    def test_off_the_node_the_no_limit_figure_is_called_a_peak_on_both_surfaces(self) -> None:
        """`--mem=0` over sstat: the row called a job-lifetime high-water `used`.

        The rule is not in dispute -- it is written down twice, on both sides of this
        number. The neighbouring `limit > 0` branch: "Off-node (sstat) the figure is a
        lifetime peak, not a live 'used', so label the bar 'peak' -- matching the text
        summary ... #34". And `_collect_remote`, from the producing end: the snapshot
        is tagged `remote=True` "so the UI labels this bar 'peak' (not 'used')", and
        "this is why the reading is LABELLED a peak everywhere it is shown (the row's
        bar says 'peak', the snapshot says source='sstat')".

        *Everywhere it is shown* -- and the no-limit branch was a place it is shown,
        hardcoding `used` for the same bytes the summary one line over already calls
        `peak`. The memory drill-in had the identical split: its `limit > 0` branch
        derives `"peak working set" if snap.remote else "working set"` because "every
        sentence about it has to be in the past tense or the card asserts a 'now' it
        never measured", and its no-limit branch hardcoded the present tense in both
        the label AND the unit caption ("GiB in use").
        """
        ctx, snap = self._no_limit(remote=True)
        cfg = SlurmwatchConfig()
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))
        summary = _plain_summary(ctx, snap, cfg)

        # The summary has said `peak` for this figure all along.
        assert "  Memory   peak 1.2 GiB" in summary
        # The row now agrees, and keeps the amount-only shape (no misleading 0% bar).
        assert "● MEM     peak    1.2 GiB · no limit set" in got["rows"], got["rows"]
        assert "MEM     used" not in got["rows"], (
            "off-node this is sstat MaxRSS, a job-lifetime high-water that only "
            "climbs -- the word the neighbouring branch and the collector both "
            "reserve for it is 'peak'"
        )
        # And the drill-in, which is one keypress away from the same bytes.
        assert "peak working set 1.2 GiB" in got["headline"], got["headline"]
        assert "GiB at peak" in got["headline"], got["headline"]
        assert "GiB in use" not in got["headline"], (
            "the big figure is a lifetime peak; 'in use' asserts a present reading"
        )

    def test_on_the_node_the_no_limit_figure_is_still_called_used(self) -> None:
        """The complement: gating on the TRANSPORT, not on the missing limit.

        On-node the same field is the cgroup's live working set, so `used` / `working
        set` / `in use` are the honest words and nothing here changes. Without this,
        "say peak" would satisfy the test above while relabelling a live reading as a
        high-water mark -- the mirror of the defect, and the reason the `limit > 0`
        branch spells the condition out rather than picking one word.
        """
        ctx, snap = self._no_limit(remote=False)
        cfg = SlurmwatchConfig()
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))

        assert "● MEM     used    1.2 GiB · no limit set" in got["rows"], got["rows"]
        assert "MEM     peak" not in got["rows"]
        assert "working set 1.2 GiB" in got["headline"], got["headline"]
        assert "peak working set" not in got["headline"]
        assert "GiB in use" in got["headline"], got["headline"]

    def test_the_summary_names_the_absent_limit_the_dashboard_names_twice(self) -> None:
        """`Memory   peak 1.2 GiB` -- and the reader had to infer "no limit" from a gap.

        The dashboard says it outright, in two places (the row's `· no limit set` and
        the drill-in headline's). The summary said it nowhere: the fact was carried
        only by the ABSENCE of the `/ 64 GiB (41%)` the limited job prints, which
        needs the other format already in mind to read. It is the surface with the
        least context to spare -- what a reader gets redirected to a file from a login
        node when they cannot have the dashboard at all.

        And `0` is overloaded on the way here, so the gap is genuinely ambiguous:
        `_parse_mem_to_bytes` returns None for a spelling it cannot read precisely
        because "downstream a limit of 0 means 'no limit is enforced'", while callers
        "with only a number to show still fall back to their own 0" (SW-12). Named,
        the line says which. Unnamed, "no --mem was asked for" and "we could not read
        the --mem that was" print identically.

        It goes in the parenthetical the limited line puts the percent in, because it
        answers that slot's question -- how this figure compares to the limit -- with
        the reason there is no comparison.
        """
        ctx, snap = self._no_limit(remote=True)
        cfg = SlurmwatchConfig()
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))
        summary = _plain_summary(ctx, snap, cfg)

        assert "  Memory   peak 1.2 GiB (no limit set)\n" in summary, summary
        # Both surfaces, same words.
        assert "no limit set" in got["rows"] and "no limit set" in got["headline"]
        # Still no limit figure and no percentage invented for a job that has none.
        assert "0 B" not in summary and "(0%)" not in summary
        # The absent limit is not an unsampled frame; the summary's own prose for
        # "nothing was measured" must not appear beside a real figure.
        assert "usage not yet sampled" not in summary

    def test_a_job_with_a_limit_renders_exactly_as_it_does_today_on_either_transport(
        self,
    ) -> None:
        """CONTROL. The limited job is byte-identical on both surfaces, both ways.

        Every string above is reached through `mem.limit_bytes <= 0`, and the words
        being changed there are spelled by the same rule the `limit > 0` branch has
        applied since #34 -- so the risk is a hoisted expression or a reused label
        leaking into the limited job, which is nearly every job. Pinned verbatim:
        the resource rows, the memory drill-in headline and the whole plain-text
        summary, off-node AND on-node. This must read the same before and after.
        """
        cfg = SlurmwatchConfig()
        expect_summary = (
            "Job 4711  gpu  RUNNING  on cn042  name `train`\n"
            "  Memory   peak 26 / 64 GiB (41%)\n"
            "  CPU      6:06:00 CPU-time  ~6.1 of 8 cores (avg, running steps)\n"
            "  source: sstat — covers only the job's tracked process tree, so a job whose\n"
            "          work runs in detached workers (R multisession/PSOCK, nohup, setsid)\n"
            "          can read far lower than reality on BOTH cpu and memory. Memory can\n"
            "          also read HIGHER than reality: MaxRSS sums each process's RSS, so a\n"
            "          shared page counts once per process. Run on the node (or let --once\n"
            "          hop there) for the true figures; live GPU utilization is on-node "
            "only.\n"
        )

        # Off-node: "peak", because the figure is MaxRSS.
        ctx = _sstat_ctx()
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=6.1 * 3600)
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))
        assert got["rows"] == (
            "  ● CPU     used    █████████████▊░░░░  76%   6.1 / 8 cores   · steady\n"
            "\n"
            "  ● MEM     peak    ███████▎░░░░░░░░░░  41%    26 / 64 GiB   · steady\n"
            "\n"
            "  ● GPU     none requested"
        )
        assert got["headline"] == "● healthy\npeak working set 26 / 64 GiB\nheadroom 38.0 GiB"
        assert _plain_summary(ctx, snap, cfg) == expect_summary

        # On-node: "used", and the row's "· peak" suffix appears (a separate figure).
        ctx = _sstat_ctx(remote=False)
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=64 * 1024**3, cpu_seconds=6.1 * 3600)
        snap.remote = False
        snap.cpu.source = snap.memory.source = "cgroup"
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))
        assert got["rows"] == (
            "  ● CPU     used    █████████████▊░░░░  76%   6.1 / 8 cores · peak 6.1   "
            "· steady\n"
            "\n"
            "  ● MEM     used    ███████▎░░░░░░░░░░  41%    26 / 64 GiB · peak 26 GiB   "
            "· steady\n"
            "\n"
            "  ● GPU     none requested"
        )
        assert got["headline"] == "● healthy\nworking set 26 / 64 GiB\nheadroom 38.0 GiB"
        assert _plain_summary(ctx, snap, cfg) == expect_summary


# ---------------------------------------------------------------------------
# The `--log` record and the screen it was recorded from: do they agree?
# ---------------------------------------------------------------------------
#
# The pair above compares two things a human reads NOW. A `--log` record is what
# somebody replays, or hands to a colleague, after the job is over -- so a figure the
# screen qualifies and the record publishes bare misleads exactly the reader who no
# longer has the screen to qualify it. Same method: one fabricated snapshot, shaped as
# `collector._collect_remote` builds it, driven through the real writer AND the real
# dashboard, then compared field by field.


def _log_records(
    ctx: JobContext,
    snap: TelemetrySnapshot,
    config: SlurmwatchConfig,
    path: Any,
    fmt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """The lines ``--log FILE`` writes for ONE snapshot, through the REAL writer.

    ``_headless_loop`` builds its own collector, so the fabricated frame goes in by
    swapping the class it constructs. That keeps the record on the shipping path --
    the same dialect resolution, header sizing, conform map and one-``write()``-
    per-record call a real run uses -- instead of re-deriving it from
    ``to_csv_row``/``to_json`` here, which is precisely the kind of second
    formatting of one number SW-4 is about.

    ``is_job_active`` is stubbed because it shells out to ``squeue``: a render in
    this file must fork nothing.
    """
    from slurmwatch import cli

    class _OneFrame(_StubCollector):
        def __init__(self, job_ctx: JobContext, cfg: SlurmwatchConfig) -> None:
            super().__init__()
            self.config = cfg
            self._left = [snap]

        async def next_snapshot(self) -> TelemetrySnapshot:
            if self._left:
                return self._left.pop()
            self.job_ended = True
            await asyncio.sleep(3600)
            raise RuntimeError

    monkeypatch.setattr(cli, "TelemetryCollector", _OneFrame)
    monkeypatch.setattr(cli, "is_job_active", lambda *a, **k: False)

    async def _drive() -> None:
        task = asyncio.create_task(cli._headless_loop(ctx, config, str(path), fmt))
        # A record is one atomic write, so a non-empty file already holds a whole
        # one: no wall-clock assertion, just "the writer got there" then stop it.
        for _ in range(400):
            await asyncio.sleep(0.005)
            if path.exists() and path.stat().st_size:
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10.0)

    asyncio.run(_drive())
    # A regular .csv gets RFC 4180 CRLF; strip the \r so the fields can be pinned.
    return [ln.rstrip("\r") for ln in path.read_text().split("\n") if ln.strip()]


class TestTheTimeBudgetDoesNotClaimTheClockIsGone:
    """`100%` was printed beside `4m left of 24:00:00 limit`, in one line.

    The info bar renders the elapsed-of-limit figure and the time remaining as two
    fields of the same sentence, and ``:.0f`` reaches ``100`` from 99.5 up. So the
    two halves contradicted each other: the percentage said the wall clock was
    spent, the field beside it said there were four minutes. On a 24-hour limit
    every job passes through that band in its last ~7 minutes -- which is exactly
    when someone is watching this line, and exactly when "am I about to be killed"
    is the question they are asking it.

    `_time_frac_text` keys the bound on ``remaining`` rather than on ``frac``,
    because `frac` is already capped with ``min(100.0, ...)``: a job PAST its limit
    also arrives reading 100, and that one has genuinely spent its budget. So
    ``100%`` survives for the case where it is true.
    """

    LIMIT = 86_400

    def _bar(self, elapsed: int, mp: Any) -> str:
        ctx = _sstat_ctx(time_limit_seconds=self.LIMIT)
        snap = _sstat_snapshot(rss=1024, limit=64 * 1024**3, cpu_seconds=1.0)
        snap.elapsed_seconds = elapsed
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        return asyncio.run(_dashboard_surfaces(ctx, snap, cfg))["bar"]

    def test_the_last_four_minutes_do_not_read_as_no_time_left(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bar = self._bar(self.LIMIT - 260, monkeypatch)
        assert "left of" in bar, bar
        assert ">99%" in bar, bar
        assert "100%" not in bar, bar

    @pytest.mark.parametrize("remaining", [1, 60, 260, 432])
    def test_every_point_inside_the_band_is_bounded(
        self, remaining: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bar = self._bar(self.LIMIT - remaining, monkeypatch)
        assert ">99%" in bar and "100%" not in bar, (remaining, bar)

    # -- CONTROLS, which must hold in BOTH states ---------------------------

    def test_control_a_job_at_its_limit_still_reads_one_hundred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing remaining is the case where the claim is TRUE. If this reddened
        the fix would have traded a false 100% for a missing one."""
        bar = self._bar(self.LIMIT, monkeypatch)
        assert "100%" in bar, bar
        assert ">99%" not in bar, bar

    def test_control_a_job_past_its_limit_still_reads_one_hundred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bar = self._bar(self.LIMIT + 600, monkeypatch)
        assert "100%" in bar, bar

    @pytest.mark.parametrize(
        ("elapsed", "shown"),
        [(0, "0%"), (86, "0%"), (43_200, "50%"), (85_880, "99%")],
    )
    def test_control_every_figure_outside_the_band_is_unchanged(
        self, elapsed: int, shown: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """85880 of 86400 is 99.398%, the last value that rounds to a real 99%."""
        bar = self._bar(elapsed, monkeypatch)
        assert shown in bar, (elapsed, bar)

    # -- the rule, at BOTH sites --------------------------------------------

    def test_the_two_render_blocks_cannot_disagree_again(self) -> None:
        """`JobInfoBar` and `ForeignJobView` compute this figure separately, and
        the note above the first says they have already drifted once ("only one
        guarded its input"). Both now go through `_time_frac_text`, pinned as the
        absence of the raw format anywhere in the module -- a second copy is what
        the drift was.
        """
        from pathlib import Path

        source = Path(tuimod.__file__).read_text()
        # Exactly once: inside `_time_frac_text`, which is the one place allowed
        # to spell it. A second occurrence is a render site formatting the figure
        # itself, which is what the drift was.
        assert source.count("frac:.0f") == 1, "a render site spells the figure itself"
        assert source.count("_time_frac_text(frac, remaining)") == 2

    @pytest.mark.parametrize(
        ("frac", "remaining", "shown"),
        [
            (0.0, 86_400, "0%"),
            (0.4, 86_000, "0%"),
            (50.0, 43_200, "50%"),
            (99.4, 520, "99%"),
            (99.5, 432, ">99%"),
            (99.7, 260, ">99%"),
            (100.0, 4, ">99%"),
            (100.0, 0, "100%"),
            (-3.0, 3_700, "-3%"),
        ],
    )
    def test_the_helper_itself(self, frac: float, remaining: int, shown: str) -> None:
        """Including the negative `frac` a clock-skewed job produces, which
        `test_remote.py::TestANegativeElapsedFromAnOlderNode` pins the pair of --
        `remaining > 0` is true there, so the guard must not swallow it."""
        assert tuimod._time_frac_text(frac, remaining) == shown


class TestTheLogRecordSaysWhatTheScreenSaidAboutOneJob:
    """One snapshot, written by ``--log`` and rendered by the dashboard, compared.

    The stamp is pinned so a record can be asserted as BYTES rather than as parsed
    fields -- the control below is "an ordinary job's record is byte-identical to
    today", and a live ``time.time()`` in column one makes that unwritable.
    """

    STAMP = 1_700_000_000.0

    @pytest.fixture(autouse=True)
    def _accounting_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same pin as the sibling class: `acct_gather_disabled()` forks `scontrol`
        and caches the answer process-wide, so left alone it makes both surfaces
        depend on the machine the suite runs on."""
        from slurmwatch import cli

        monkeypatch.setattr(tuimod, "acct_gather_disabled", lambda: False)
        monkeypatch.setattr(cli, "acct_gather_disabled", lambda: False)

    def _snap(self, *, limit: int) -> TelemetrySnapshot:
        snap = _sstat_snapshot(rss=26 * 1024**3, limit=limit, cpu_seconds=6.1 * 3600)
        snap.timestamp = self.STAMP
        return snap

    def _row(
        self, ctx: JobContext, snap: TelemetrySnapshot, tmp_path: Any, mp: Any, gpus: int = 0
    ) -> dict[str, str]:
        """One record as name -> cell.

        ``gpus`` mirrors the writer's own sizing (``max(len(snap.gpus),
        gpu_count_requested)``, #38), asserted rather than assumed so a row is only
        ever read against the header it was written under.
        """
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        lines = _log_records(ctx, snap, cfg, tmp_path / "run.csv", "csv", mp)
        assert len(lines) == 2, lines
        header, row = lines
        assert header.split(",") == TelemetrySnapshot.csv_header(gpus), "header layout moved"
        return dict(zip(header.split(","), row.split(","), strict=True))

    # -- the finding: a percentage of a limit that does not exist ------------

    def test_the_record_publishes_no_memory_percent_when_the_screen_shows_none(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job with no ``--mem``, off-node: 26 GiB against no limit at all.

        Both human surfaces refuse the percentage and name the reason in its slot --
        the row prints ``26.0 GiB · no limit set`` with no bar at all, because "a
        'used 0%' bar would contradict the GiB in use", and the summary prints
        ``peak 26.0 GiB (no limit set)``.

        The record printed the bar's missing number anyway::

            mem_current_bytes=27917287424 ... mem_limit_bytes=0 ... mem_percent=0.00

        i.e. "this job used 0% of its memory" about a job holding 26 GiB -- and
        `mem_percent` is the figure a right-sizing consumer acts on, so the advice it
        yields is "shrink --mem". Off-node is where this is the NORMAL spelling, not
        an edge case: `_collect_remote` copies `ctx.mem_limit_bytes` through with no
        node-RAM fallback, so a job submitted with no `--mem` on a cluster with no
        DefMemPerCPU has `limit_bytes == 0` in every row of the file, and `sw
        <jobid>` from a login node is the primary documented workflow.

        Empty is this file's own convention for the case, established one column
        earlier for the wall-clock denominator (`time_limit_seconds`: "empty when the
        job has no limit") and again for the fabric rates.
        """
        ctx = _sstat_ctx(mem_limit_bytes=0)
        snap = self._snap(limit=0)
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))
        summary = _plain_summary(ctx, snap, cfg)
        # The screen half of the pair: no percentage anywhere on the MEM row.
        mem_row = next(ln for ln in got["rows"].splitlines() if "MEM" in ln)
        assert mem_row == "  ● MEM     peak    26.0 GiB · no limit set", mem_row
        assert "%" not in mem_row
        assert "  Memory   peak 26.0 GiB (no limit set)\n" in summary

        row = self._row(ctx, snap, tmp_path, monkeypatch)
        # The bytes are still reported, and still say which reading they are.
        assert row["mem_current_bytes"] == str(26 * 1024**3)
        assert row["mem_limit_bytes"] == "0"
        assert row["mem_source"] == "sstat" and row["remote"] == "1"
        # ...and the percentages are withheld rather than invented.
        assert row["mem_percent"] == "", row["mem_percent"]
        assert row["mem_working_set_percent"] == "", row["mem_working_set_percent"]

    def test_the_json_record_withholds_it_too_and_still_replays(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same withholding in the format ``--log`` DEFAULTS to.

        Only a ``.csv`` extension picks CSV (`_infer_use_json`), so a bare ``--log
        run.jsonl`` -- or any name at all -- is JSON, and fixing one encoding would
        leave the common one publishing the bare zero. It would also break the rule
        `--help` documents and `TestTheDocumentedFormatMappingIsTrue` asserts:
        `memory.usage_percent` and `mem_percent` are one quantity under two names.

        `null` is safe to hand back to `from_dict`, which is not incidental -- the
        node switcher parses exactly these lines. `_only` already coerces a null in a
        numeric field to 0.0 for the NaN contract, and the rebuilt snapshot still
        carries `limit_bytes == 0`, so every renderer takes the "no limit set" branch
        that shows no percentage anyway.
        """
        import json

        ctx = _sstat_ctx(mem_limit_bytes=0)
        snap = self._snap(limit=0)
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
        lines = _log_records(ctx, snap, cfg, tmp_path / "run.jsonl", "json", monkeypatch)
        assert len(lines) == 1, lines
        doc = json.loads(lines[0])

        assert doc["memory"]["current_bytes"] == 26 * 1024**3
        assert doc["memory"]["limit_bytes"] == 0
        assert doc["memory"]["usage_percent"] is None
        assert doc["memory"]["working_set_percent"] is None

        back = TelemetrySnapshot.from_json(lines[0])
        assert back.memory.limit_bytes == 0
        assert back.memory.usage_percent == 0.0
        assert back.memory.working_set_percent == 0.0
        # And the replayed frame renders the same words the recorded one did.
        replayed = asyncio.run(_dashboard_surfaces(ctx, back, cfg))
        assert "26.0 GiB · no limit set" in replayed["rows"], replayed["rows"]

    # -- the CONTROL, which must hold in BOTH states -------------------------

    def test_an_ordinary_jobs_record_is_byte_identical_to_today(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL. 26 GiB of a 64 GiB limit -- the record, verbatim, both formats.

        Nearly every job has a limit, so the risk of withholding a figure on
        `limit_bytes <= 0` is that the condition leaks into the branch that was
        already right. The CSV data row is pinned as one string (that IS the record,
        byte for byte) and the JSON payload's whole `memory` object as another -- the
        two places the change can reach. This must read the same before and after.
        """
        import json

        ctx = _sstat_ctx()
        snap = self._snap(limit=64 * 1024**3)
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        lines = _log_records(ctx, snap, cfg, tmp_path / "run.csv", "csv", monkeypatch)
        assert lines[0].split(",") == TelemetrySnapshot.csv_header(0)
        assert lines[1] == (
            "1700000000.000,4711,train,cn042,3600,,gpu,,,,,,8,21960000000000,sstat,"
            "76.20,6.10,6.10,27917287424,68719476736,27917287424,0,sstat,0,40.60,"
            "40.60,27917287424,1,27917287424,0,0,0,0,0.00,1,0,1,0,1,0,0,,0,,,,,,,,"
            ",,,,,,"
        ), lines[1]

        jlines = _log_records(ctx, snap, cfg, tmp_path / "run.jsonl", "json", monkeypatch)
        assert json.dumps(json.loads(jlines[0])["memory"]) == (
            '{"current_bytes": 27917287424, "limit_bytes": 68719476736, '
            '"peak_bytes": 27917287424, "usage_percent": 40.6, '
            '"oom_guard_warning": false, "oom_guard_critical": false, '
            '"working_set_bytes": 27917287424, "cache_bytes": 0, '
            '"peak_working_set_bytes": 27917287424, "working_set_percent": 40.6, '
            '"source": "sstat", "cache_measured": false, "peak_is_lifetime": true}'
        ), jlines[0]

        # The screen it was recorded from, unchanged too: the percentage is shown
        # here, so withholding it must be keyed on the missing limit and nothing else.
        got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))
        assert "41%    26 / 64 GiB" in got["rows"], got["rows"]

    # -- cases checked and found already qualified ---------------------------

    def test_a_frame_with_no_reading_in_it_says_so_in_the_record(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`usage_sampled=False`: every counter 0 because nothing was measured.

        Not a finding -- pinned so it stays that way. The chip says "no sample yet"
        and the summary prints no figure at all; the record's zeros travel with
        `usage_sampled=0`, which is the field that distinction exists on.
        """
        ctx = _sstat_ctx()
        snap = _sstat_snapshot(rss=0, limit=64 * 1024**3, cpu_seconds=0.0, sampled=False)
        snap.timestamp = self.STAMP
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        assert "no sample yet" in asyncio.run(_dashboard_surfaces(ctx, snap, cfg))["bar"]
        row = self._row(ctx, snap, tmp_path, monkeypatch)
        assert row["usage_sampled"] == "0"
        assert row["mem_current_bytes"] == "0" and row["cpu_effective_cores"] == "0.00"
        # A limit exists here, so the percent is a real measured 0 -- not withheld.
        assert row["mem_percent"] == "0.00"

    def test_the_unreadable_gpu_set_is_unknown_in_the_record_not_idle(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four GPUs allocated, none openable: the screen says "telemetry
        unavailable here", and the record says it too.

        Not a finding -- pinned. `gpu_active_count` is empty rather than a summed 0,
        and the cause travels beside it, so the row cannot be read as "read all four,
        all four idle".
        """
        ctx = _sstat_ctx(gpu_count_requested=4, gpu_indices=[0, 1, 2, 3])
        snap = self._snap(limit=64 * 1024**3)
        snap.gpu_count_requested = 4
        snap.gpu_active_count = None
        snap.gpu_unavailable_reason = "devices_denied"
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        rows = asyncio.run(_dashboard_surfaces(ctx, snap, cfg))["rows"]
        assert "telemetry unavailable here" in rows, rows
        # Four requested and none readable still sizes the header for four groups
        # (`max(len(snap.gpus), gpu_count_requested)`), so the row has to be read
        # against `csv_header(4)`; against `csv_header(0)` every name would be
        # paired with the wrong cell.
        row = self._row(ctx, snap, tmp_path, monkeypatch, gpus=4)
        assert row["gpu_active_count"] == ""
        assert row["gpu_count_requested"] == "4"
        assert row["gpu_monitoring_available"] == "0"
        assert row["gpu_unavailable_reason"] == "devices_denied"

    def test_the_off_node_memory_readings_provenance_survives_into_the_record(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The drill-in says "peak this job (lifetime)" and "cache now not
        measured"; the record carries the flags both of those are spelled from.

        Not a finding -- pinned, because these are the fields the sibling class's
        findings were about and a record that dropped them would be the same defect
        one surface further out.
        """
        ctx = _sstat_ctx()
        snap = self._snap(limit=64 * 1024**3)
        cfg = SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)

        body = asyncio.run(_dashboard_surfaces(ctx, snap, cfg, drill="m"))["body"]
        assert "(lifetime)" in body and "not measured" in body, body
        row = self._row(ctx, snap, tmp_path, monkeypatch)
        assert row["mem_peak_is_lifetime"] == "1"
        assert row["mem_cache_measured"] == "0" and row["mem_cache_bytes"] == "0"
        assert row["mem_source"] == "sstat" and row["remote"] == "1"
        # One MaxRSS high-water behind all four figures, as the record must show.
        assert (
            row["mem_current_bytes"]
            == row["mem_peak_bytes"]
            == row["mem_working_set_bytes"]
            == row["mem_peak_working_set_bytes"]
        )
