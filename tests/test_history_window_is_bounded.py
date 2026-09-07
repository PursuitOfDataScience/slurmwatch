"""D5: the history deque was sized from two independently clamped knobs.

`history_seconds` has a ceiling (`MAX_HISTORY_SECONDS`, one day) and
`poll_interval` has a floor (`MIN_INTERVAL`, 0.1 s). Their **quotient** — which
is what `deque(maxlen=…)` gets — had none. Re-measured before touching anything,
because nine items in this family have been found already fixed while their row
still called them open. This one is **live**, and the row's mechanism holds; its
arithmetic was off by 2x and its verification line was measured at a non-default
`history_seconds`.

What was measured, on this machine::

    asked (s)  interval | clamped int   maxlen per series
        86400      0.05 |         0.1            864,000     <- the blessed maximum
        86400       0.1 |         0.1            864,000
        86400       1.0 |         1.0             86,400
         1e19      0.05 |         0.1            864,000
           60      0.05 |         0.1                600     <- the row said 1200;
           120     0.05 |         0.1              1,200        that is history=120

(`--interval 0.05` is floored to 0.1, so the row's 1,728,000 is 864,000. The
"maxlen 1200, not 120" line is true at `history_seconds=120`, not at the
default 60.)

The dashboard keeps `2 + 2 x N_GPU` of these — 18 on an 8-GPU node. Filled the
way the app fills them, one float per poll, and read from `/proc/self/statm`::

    18 x     120 slots ->     0 KiB
    18 x   3,600 slots ->  2,112 KiB =   2.06 MiB   (33.4 B/slot)
    18 x  86,400 slots -> 57,940 KiB =  56.58 MiB
    18 x 864,000 slots -> 616,550 KiB = 602.1 MiB   (40.6 B/slot)

**602 MiB, on the compute node being monitored** — memory that belongs to the
job slurmwatch is there to help right-size. And `_trend_tag` walks the whole
window twice a frame::

    maxlen   list+min+max   min/max no copy
       120         3.4 us            2.8 us
     3,600        92.7 us           75.2 us
   864,000     24840.4 us        18298.2 us

24.5 ms a call x 2 calls a frame x 10 frames a second (the 0.1 s interval floor)
= **49% of the event loop**, spent aggregating a window nothing can read.

Two changes, both pinned below:

1. **`MAX_HISTORY_SAMPLES = 3600` caps the slot count** (`config.py`), applied in
   `DashboardScreen._history_maxlen`. The cap is on slots, not seconds, so no
   reasonable configuration retains less than it asked for — 30 minutes at 2 s is
   900 slots, an hour at 1 s is 3,600, the 60 s default at any interval down to
   the floor is at most 600. Where it does bite, `_history_window_seconds` is
   what the row trend tag, the drill-in chart caption and the chart's time axis
   report, so the UI never advertises a depth it is not holding: `HISTORY_SECONDS
   =86400` at the 0.1 s floor now reads `over 360s`, not `over 86400s`.
2. **`_trend_tag` takes min/max off the deque** instead of `list(hist)` first.
   The copy was 28,928 bytes allocated and freed per call at a 3,600-slot window,
   twice a frame. Now 72 bytes, and it does not grow with the window.

`slurmwatch --once --json` is untouched: the same real job (53834744, on
midway3-0200) emits the same 58-key payload before and after.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from collections import deque
from typing import cast

import pytest
from rich.markup import render as _render_markup

from slurmwatch.config import MAX_HISTORY_SAMPLES, MAX_HISTORY_SECONDS, SlurmwatchConfig
from slurmwatch.tui import DashboardScreen, ResourceDetailScreen, ResourceRows

from .test_tui import _make_snapshot

#: The number of history series a real 8-GPU node carries: cpu, mem, and a
#: compute + vram pair per device. What the 602 MiB was measured on.
SERIES_ON_AN_EIGHT_GPU_NODE = 2 + 2 * 8


def _screen(history_seconds: int, poll_interval: float, *, remote: bool = False) -> DashboardScreen:
    """A `DashboardScreen` carrying a real, clamped config and nothing else.

    `_history_maxlen` and `_effective_interval` read exactly three attributes, so
    this drives the production code without mounting a Textual app — the same
    trick `test_tui.py`'s cadence test uses, minus the app.
    """
    cfg = SlurmwatchConfig()
    cfg.history_seconds = history_seconds
    cfg.poll_interval = poll_interval
    cfg.clamp()  # what `from_env` and the CLI both do
    screen = object.__new__(DashboardScreen)
    screen.config = cfg
    screen._local_node = "cn001"
    screen._selected_node = "cn002" if remote else "cn001"
    return screen


def _rss_kib() -> int:
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") // 1024


def _fill(series: int, maxlen: int) -> list[deque[float]]:
    """`series` history deques, full, appended one sample at a time."""
    out: list[deque[float]] = []
    for _ in range(series):
        window: deque[float] = deque(maxlen=maxlen)
        for i in range(maxlen):
            window.append(i * 1.000001)  # a distinct float per slot, as telemetry is
        out.append(window)
    return out


def _rows(cpu_history: deque[float], window_seconds: int | None) -> ResourceRows:
    rows = ResourceRows()
    rows.snapshot = _make_snapshot()
    rows.config = SlurmwatchConfig()
    rows.cpu_history = cpu_history
    if window_seconds is not None:
        rows.window_seconds = window_seconds
    return rows


def _timed(window: deque[float]) -> float:
    """One `_trend_tag` call, wall clock. Best-of, so a login-node stall is not it."""
    start = time.perf_counter()
    ResourceRows._trend_tag(window, 60, False)
    return time.perf_counter() - start


def _cpu_line(rows: ResourceRows) -> str:
    return next(line for line in _render_markup(rows.render()).plain.splitlines() if "CPU" in line)


class _StubDashboard:
    """The one fact `_chart_lines` asks its dashboard for."""

    def __init__(self, window_seconds: int) -> None:
        self._window_seconds = window_seconds

    def _history_window_seconds(self) -> int:
        return self._window_seconds


def _chart(window_seconds: int, history_seconds: int) -> list[str]:
    """The drill-in chart's lines, with the dashboard's retained window stubbed.

    `_chart_lines` is otherwise self-contained (it takes the series, the width and
    the height), so this is the real caption code with the one fact it asks the
    dashboard for supplied.
    """
    cfg = SlurmwatchConfig()
    cfg.history_seconds = history_seconds
    screen = object.__new__(ResourceDetailScreen)
    screen._dashboard = cast(DashboardScreen, _StubDashboard(window_seconds))
    series: deque[float] = deque([10.0, 40.0, 55.0] * 8, maxlen=120)
    return screen._chart_lines(series, cfg, "green", "CPU", area_w=40, height=6)


# --------------------------------------------------------------------------- #
# The cap.
# --------------------------------------------------------------------------- #


class TestTheSlotCountIsBounded:
    """The quotient of two independently clamped knobs now has a ceiling."""

    @pytest.mark.parametrize(
        "history,interval",
        [
            (86_400, 0.05),  # the blessed maximum at the interval floor: 864,000
            (86_400, 0.1),
            (86_400, 0.5),  # the TUI default interval:                   172,800
            (86_400, 1.0),  # the headless default:                        86,400
            (10**19, 0.05),  # a finite nonsense value, clamped to the day
            (43_200, 0.1),
        ],
    )
    def test_no_configuration_can_ask_for_more_slots_than_the_cap(
        self, history: int, interval: float
    ) -> None:
        assert _screen(history, interval)._history_maxlen() <= MAX_HISTORY_SAMPLES

    def test_the_worst_case_is_the_cap_exactly(self) -> None:
        # 864,000 before; the number the memory and timing figures below are for.
        assert _screen(86_400, 0.05)._history_maxlen() == MAX_HISTORY_SAMPLES

    def test_the_cap_also_bounds_a_remote_stream(self) -> None:
        # The remote cadence is floored at 1.0 s, which divides the slot count by
        # 10 but does not bound it: a day at 1 s is still 86,400 slots.
        assert _screen(86_400, 0.5, remote=True)._history_maxlen() == MAX_HISTORY_SAMPLES

    @pytest.mark.skipif(not os.path.exists("/proc/self/statm"), reason="needs Linux /proc for RSS")
    def test_the_deques_cost_megabytes_not_hundreds_of_megabytes(self) -> None:
        """The figure the whole item is about, measured rather than projected.

        18 series at the worst-case maxlen, filled. 602 MiB before (864,000
        slots), 2.06 MiB after (3,600). The bound is deliberately loose — this
        reads a real RSS on a shared login node — but 32 MiB still sits 18x below
        the pre-fix figure at the 86,400-slot rung and 300x below the 864,000 one.
        """
        maxlen = _screen(86_400, 0.05)._history_maxlen()
        baseline = _rss_kib()
        held = _fill(SERIES_ON_AN_EIGHT_GPU_NODE, maxlen)
        grew_kib = _rss_kib() - baseline
        assert len(held[0]) == maxlen
        assert grew_kib < 32 * 1024, f"{grew_kib} KiB for {len(held)} x {maxlen:,} slots"

    def test_the_trend_tag_stays_cheap_at_the_worst_case_window(self) -> None:
        # 24.5 ms a call before, 75 us after -- two calls a frame, ten frames a
        # second. A generous bound (a shared login node stalls), still 12x under
        # the pre-fix figure.
        maxlen = _screen(86_400, 0.05)._history_maxlen()
        window: deque[float] = deque((i % 97 * 1.01 for i in range(maxlen)), maxlen=maxlen)
        best = min(_timed(window) for _ in range(5))
        assert best < 2e-3, f"{best * 1e3:.2f} ms per _trend_tag call at {maxlen:,} slots"


class TestTheTrendTagDoesNotCopyTheWindow:
    """`list(hist)` allocated the whole window on every frame, to aggregate it."""

    def test_one_call_allocates_bytes_not_kilobytes(self) -> None:
        maxlen = MAX_HISTORY_SAMPLES
        window: deque[float] = deque((i % 97 * 1.01 for i in range(maxlen)), maxlen=maxlen)
        ResourceRows._trend_tag(window, 60, False)  # warm any lazy import
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            ResourceRows._trend_tag(window, 60, False)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # 28,928 bytes before at this length; 72 after.
        assert peak - before < 4096, f"{peak - before} bytes for a {maxlen:,}-slot window"

    def test_the_allocation_does_not_grow_with_the_window(self) -> None:
        """The scaling, not just the constant -- and it is deterministic.

        Before: 872 bytes at 100 slots, 28,928 at 3,600. After: 72 at both.
        """
        peaks = []
        for maxlen in (100, MAX_HISTORY_SAMPLES):
            window: deque[float] = deque((i % 97 * 1.01 for i in range(maxlen)), maxlen=maxlen)
            ResourceRows._trend_tag(window, 60, False)
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                ResourceRows._trend_tag(window, 60, False)
                peaks.append(tracemalloc.get_traced_memory()[1] - before)
            finally:
                tracemalloc.stop()
        small, large = peaks
        assert large <= small + 512, f"grew from {small} to {large} bytes with the window"


# --------------------------------------------------------------------------- #
# Telling the truth about what is retained.
# --------------------------------------------------------------------------- #


class TestTheReportedWindowIsTheRetainedWindow:
    """A cap that still advertises the requested depth would be worse than none."""

    def test_the_retained_window_is_the_slots_times_the_cadence(self) -> None:
        screen = _screen(86_400, 0.05)
        # 3,600 slots x the 0.1 s floor = 360 s, not the 86,400 s asked for.
        assert screen._history_window_seconds() == 360

    @pytest.mark.parametrize(
        "history,interval,expected",
        [
            (86_400, 0.1, 360),
            (86_400, 0.5, 1_800),
            (86_400, 1.0, 3_600),
            (3_600, 0.5, 1_800),
        ],
    )
    def test_every_capped_configuration_reports_what_it_holds(
        self, history: int, interval: float, expected: int
    ) -> None:
        screen = _screen(history, interval)
        assert screen._history_window_seconds() == expected
        assert screen._history_window_seconds() < history

    def test_the_row_trend_tag_names_the_retained_window(self) -> None:
        # The surface a user actually reads: "9-15% over 360s". It said 86400s.
        screen = _screen(86_400, 0.05)
        rows = _rows(
            deque([10.0, 40.0, 20.0, 55.0] * 4, maxlen=screen._history_maxlen()),
            screen._history_window_seconds(),
        )
        line = _cpu_line(rows)
        assert "over 360s" in line, line
        assert "86400" not in line, line

    def test_the_chart_caption_and_axis_name_the_retained_window(self) -> None:
        lines = _chart(window_seconds=360, history_seconds=86_400)
        joined = "\n".join(_render_markup(line).plain for line in lines)
        assert "last 360s" in joined, joined
        assert "← 360s" in joined, joined
        assert "86400" not in joined, joined


# --------------------------------------------------------------------------- #
# Controls. Each of these passes with both changes reverted as well.
# --------------------------------------------------------------------------- #


class TestControls:
    """What must not change. Verified green against the pre-fix code too."""

    @pytest.mark.parametrize(
        "history,interval,slots",
        [
            (60, 0.5, 120),  # the TUI default -- and test_tui's cadence test
            (60, 0.1, 600),  # the default window at the interval floor
            (60, 1.0, 60),  # the headless default
            (1_800, 2.0, 900),  # "30 minutes at 2s must still get 30 minutes"
            (3_600, 1.0, 3_600),  # an hour at 1s: exactly at the cap, untouched
            (7_200, 2.0, 3_600),  # two hours at 2s: also exactly at the cap
        ],
    )
    def test_control_a_reasonable_configuration_keeps_every_slot_it_asked_for(
        self, history: int, interval: float, slots: int
    ) -> None:
        assert _screen(history, interval)._history_maxlen() == slots

    @pytest.mark.parametrize(
        "history,interval",
        [(60, 0.5), (60, 0.1), (60, 1.0), (1_800, 2.0), (3_600, 1.0), (7_200, 2.0)],
    )
    def test_control_a_reasonable_configuration_still_spans_what_it_asked(
        self, history: int, interval: float
    ) -> None:
        # Expressed as slots x cadence rather than through the new helper, so it
        # is a control on the retained span and not on the new method's existence.
        screen = _screen(history, interval)
        span = screen._history_maxlen() * screen._effective_interval()
        assert round(span) == history, (history, interval, span)

    def test_control_the_remote_cadence_still_sizes_the_window(self) -> None:
        # #55, the reason `_effective_interval` exists: 60 s of a 1 s remote
        # stream is 60 slots, not the 120 the local 0.5 s cadence would give.
        assert _screen(60, 0.5, remote=True)._history_maxlen() == 60
        assert _screen(60, 0.5)._history_maxlen() == 120

    def test_control_the_floor_of_ten_slots_still_holds(self) -> None:
        # A 1 s window at the 1 hour interval ceiling would otherwise be 0 slots.
        assert _screen(1, 3_600.0)._history_maxlen() == 10

    def test_control_history_seconds_is_still_clamped_to_a_day(self) -> None:
        cfg = SlurmwatchConfig()
        cfg.history_seconds = 10**19
        cfg.clamp()
        assert cfg.history_seconds == MAX_HISTORY_SECONDS

    def test_control_the_trend_tag_still_reports_the_observed_range(self) -> None:
        rows = _rows(deque([10.0, 40.0, 20.0, 55.0, 30.0] * 4, maxlen=120), None)
        line = _cpu_line(rows)
        assert "10–55%" in line, line
        assert "over 60s" in line, line

    def test_control_a_steady_series_still_reads_steady(self) -> None:
        line = _cpu_line(_rows(deque([50.0] * 20, maxlen=120), None))
        assert "steady" in line, line
        assert "–" not in line, line

    def test_control_an_empty_series_still_prints_no_tag(self) -> None:
        assert ResourceRows._trend_tag(deque(), 60, False) == ""
        assert ResourceRows._trend_tag(deque(), 60, True) == ""

    def test_control_the_trend_tag_is_exact_not_sampled(self) -> None:
        # A single outlier at the far end of a long window must still show up --
        # the cheap way to make min/max fast would be to look at a subset.
        maxlen = MAX_HISTORY_SAMPLES
        window: deque[float] = deque([50.0] * maxlen, maxlen=maxlen)
        window[0] = 3.0
        window[maxlen // 2] = 97.0
        assert "3–97%" in ResourceRows._trend_tag(window, 60, False)

    def test_control_the_ascii_trend_tag_is_unchanged(self) -> None:
        window: deque[float] = deque([10.0, 55.0], maxlen=120)
        tag = ResourceRows._trend_tag(window, 60, True)
        assert "10-55% over 60s" in tag, tag
        assert "–" not in tag and "·" not in tag, tag

    def test_control_an_uncapped_window_reports_the_requested_depth(self) -> None:
        # The other side of the truth-telling: when the cap does not bite, all
        # three surfaces still say exactly what the user configured.
        rows = _rows(deque([10.0, 40.0, 20.0, 55.0] * 4, maxlen=120), 60)
        assert "over 60s" in _cpu_line(rows)
        joined = "\n".join(
            _render_markup(line).plain for line in _chart(window_seconds=60, history_seconds=60)
        )
        assert "last 60s" in joined, joined
        assert "← 60s" in joined, joined

    def test_control_the_chart_still_reports_its_own_statistics(self) -> None:
        joined = "\n".join(
            _render_markup(line).plain
            for line in _chart(window_seconds=360, history_seconds=86_400)
        )
        assert "min " in joined and "avg " in joined and "max " in joined, joined
        assert "now " in joined, joined
