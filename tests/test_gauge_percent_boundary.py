"""Every gauge claimed 100% from 99.5 up, on both surfaces.

`:.0f` reaches `100` at 99.5, so the CPU, memory, GPU-compute and VRAM gauges --
and the plain report's `Memory peak x / y (n%)` -- all said a job was at its
limit while it still had headroom. Measured on the dashboard renderer before the
fix::

    99.40 -> 'used    ███████▉  99%'
    99.60 -> 'used    ████████ 100%'      <- solid bar AND a boundary claim
    99.99 -> 'used    ████████ 100%'

**This file's own module already fixed it one gauge over.**
`tui._time_frac_text` bounds the elapsed figure to `>99%` and its docstring gives
the reason and the family's spelling: "`:.0f` reaches 100 from 99.5 up, so the two
halves of that one line contradicted each other ... An inequality is the family's
spelling for 'past the resolution but not at the boundary' -- `rapidu.fmt.ratio_x`
returns `<0.01x`, `nodetop.core.duration` returns `<1m`". The other gauges never
got it. The memory figure is the one the off-node OOM guard fires on and the one a
reader consults before raising `--mem`, so "at your limit" and "a fraction under
it" must not print the same.

`units.pct_text` is the shared spelling, and it lives in `units` rather than in
either renderer because BOTH draw this figure -- which is the reason `mem_pair`
lives there too, recorded at its own call site: "not a local `:.1f` GiB -- a
`--mem=400M` job read `0.0 GiB / 0.4 GiB` here long after the dashboard gauge was
fixed, because this renderer had its own copy of the arithmetic".

**The bar had to move with the label**, in both draw paths. Each carried a guard
whose comment says a solid bar must agree with a "100%" label, and each keyed on
`round(percent) < 100` -- so in the 99.5-99.99 band the bar and the label claimed
100% *together*, consistently and wrongly. Both now key on the unrounded value, so
a `>99%` label is drawn with the last cell (unicode) or the last character (ASCII)
held back.

Only the top boundary moves: `0%` for a sub-0.5% value is kept deliberately (the
bar beside it is empty, and that is the reading its own underuse advisory gives),
values at or above 100 are untouched, and a negative from a clock-skewed reading
still prints as it did.
"""

from __future__ import annotations

import re

import pytest

from slurmwatch.tui import _bar_cells, _color_bar, _labeled_bar
from slurmwatch.units import pct_text


def _plain(markup: str) -> str:
    return re.sub(r"\[/?[^\]]*\]", "", markup)


def _gauge(percent: float, ascii_mode: bool = False, width: int = 8) -> str:
    return _plain(_labeled_bar("used", percent, width, ascii_mode, "white")).strip()


#: The band `:.0f` rounds up to 100 while the truth is below it.
JUST_UNDER = [99.5, 99.6, 99.9, 99.99]


class TestTheLabelDoesNotClaimAnUnreachedBoundary:
    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_band_reads_as_a_bound(self, percent: float) -> None:
        assert pct_text(percent) == ">99%"

    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_gauge_shows_the_bound(self, percent: float) -> None:
        assert ">99%" in _gauge(percent), _gauge(percent)

    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_gauge_does_not_say_one_hundred(self, percent: float) -> None:
        assert "100%" not in _gauge(percent), _gauge(percent)

    def test_the_bound_fits_the_slot(self) -> None:
        # Four characters either way, which is what the gauge's `:>4` allows.
        assert len(pct_text(99.6)) == len(pct_text(100.0)) == 4


class TestTheBarAgreesWithTheLabel:
    """Both draw paths carry a guard saying a solid bar means a 100% label."""

    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_unicode_bar_is_not_solid_in_the_band(self, percent: float) -> None:
        bar = _plain(_color_bar(percent, 8, False, "white"))
        assert bar.count("█") < 8, (percent, bar)

    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_ascii_bar_reserves_its_last_cell(self, percent: float) -> None:
        assert _bar_cells(percent, 8) == 7, percent

    @pytest.mark.parametrize("percent", JUST_UNDER)
    def test_the_two_halves_of_the_gauge_do_not_contradict(self, percent: float) -> None:
        for ascii_mode in (False, True):
            cell = _gauge(percent, ascii_mode)
            solid = "########" in cell or "████████" in cell
            assert not solid, (percent, ascii_mode, cell)

    def test_a_real_hundred_is_still_solid_and_says_so(self) -> None:
        for ascii_mode in (False, True):
            cell = _gauge(100.0, ascii_mode)
            assert "100%" in cell, cell
            assert "########" in cell or "████████" in cell, cell


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize(
        ("percent", "expected"),
        [
            (0.0, "0%"),
            (0.4, "0%"),
            (1.0, "1%"),
            (42.0, "42%"),
            (99.4, "99%"),
            (100.0, "100%"),
            (100.4, "100%"),
            (250.0, "250%"),
        ],
    )
    def test_every_other_value_prints_as_it_did(self, percent: float, expected: str) -> None:
        assert pct_text(percent) == expected

    def test_a_negative_reading_is_untouched(self) -> None:
        # A clock-skewed job reads -3%, and `test_remote.py` pins that pair.
        assert pct_text(-3.0) == "-3%"
        assert "-3%" in _gauge(-3.0)

    def test_a_sub_half_percent_still_draws_empty(self) -> None:
        # The bar and the "0%" beside it are deliberately kept in step.
        assert _bar_cells(0.4, 8) == 0
        assert "0%" in _gauge(0.4)
        assert "█" not in _gauge(0.4)

    def test_a_one_percent_value_still_keeps_its_sliver(self) -> None:
        assert _bar_cells(1.0, 8) >= 1
        assert "1%" in _gauge(1.0)

    def test_the_elapsed_figure_keeps_its_own_stronger_rule(self) -> None:
        # `_time_frac_text` keys on the time LEFT, not on the value: a job whose
        # budget really is spent still reads 100%.
        from slurmwatch.tui import _time_frac_text

        assert _time_frac_text(100.0, remaining=0) == "100%"
        assert _time_frac_text(99.6, remaining=240) == ">99%"

    def test_a_nan_still_does_not_crash_either_path(self) -> None:
        assert _bar_cells(float("nan"), 8) == 0
        assert _plain(_color_bar(float("nan"), 8, False, "white")).count("█") == 0

    def test_a_zero_width_gauge_still_draws_nothing(self) -> None:
        assert _color_bar(50.0, 0, False, "white") == ""
        assert _bar_cells(50.0, 0) == 0
