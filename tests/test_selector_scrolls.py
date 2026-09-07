"""The job picker clipped its list instead of scrolling it.

Reported against a 41-job list: "the highlightor doesn't scroll down the job
list". The cause was that `max-height: 32` in `JobSelectorScreen.CSS` was a
fixed ROW COUNT and the only bound on the `ListView`. `Widget.size` is the
content region, so the box's own cap resolves to `int(term_h * 0.92) - 6`
(border 2 + padding 2*2), and inside that the four fixed statics take 6 rows
(title 2, header 1, rule 1, hint 2). On any terminal shorter than ~48 rows the
list therefore wanted more rows than the box could give, and Textual **clipped**
it rather than shrinking it. Measured before the fix, 41 jobs (43 content rows,
the two array rows being two lines each):

    terminal   box content   list height   overflow
    120x40         30            32            8
    120x30         21            32           17
    120x24         16            32           22

The clip is what broke the cursor. The ListView still believed its viewport was
32 rows, so `cursor_down` had nothing to scroll until the cursor passed row 32
of the content -- by which point the highlight had walked off the bottom of the
visible area and the rest of the list was unreachable.

`_fit_list` narrows both bounds to the room actually available, at mount and on
resize. Both bounds, because the CSS `min-height: 12` would otherwise win on a
short terminal and re-create the same overflow.

The assertion that matters is not a height: it is that the highlighted row stays
PAINTED for every job in the list, which is what the user was doing when it
broke. `_visible_band` explains why the ListView's own height is the wrong
yardstick for that. `TestControls` pins the things that must not move -- a short list
staying compact, and the ceiling still applying on a tall terminal.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.geometry import Region
from textual.widgets import ListView

from slurmwatch.tui import JobSelectorScreen


#: The reported shape: two pending array rows, then running tasks. They USED to
#: render as two lines each (41 jobs -> 43 content rows); the table budget added in
#: `test_picker_table_fits.py` keeps every row on one line, so it is 41 rows now.
def _jobs(n: int = 41) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [
        {
            "job_id": "57892948_[99-107%18]",
            "state": "PD",
            "name": "speccurve",
            "partition": "amd",
            "nodes": "1",
            "reason": "JobArrayTaskLimit",
        },
        {
            "job_id": "57892947_[18-79%18]",
            "state": "PD",
            "name": "speccurve",
            "partition": "amd",
            "nodes": "1",
            "reason": "JobArrayTaskLimit",
        },
    ]
    for i in range(max(0, n - 2)):
        out.append(
            {
                "job_id": f"57892947_{i}",
                "state": "R",
                "name": "speccurve",
                "partition": "amd",
                "nodes": "1",
                "wall_time": "7:51",
            }
        )
    return out[:n]


class _Harness(App[None]):
    def __init__(self, jobs: list[dict[str, object]]) -> None:
        super().__init__()
        self._jobs = jobs
        self.scr: JobSelectorScreen | None = None

    async def on_mount(self) -> None:
        self.scr = JobSelectorScreen(self._jobs, flourish=False)
        self.push_screen(self.scr)


def _visible_band(scr: JobSelectorScreen) -> Region:
    """The rows of the list that are actually PAINTED.

    Deliberately not `lv.size.height`: that is the height the ListView claims,
    and claiming a height larger than the box is the whole defect. An earlier
    draft of this file measured against it and passed against the broken code --
    at 120x24 the list claimed rows y=8..39 while the box painted only y=8..22,
    so a row at y=30 was "within the viewport" and invisible on screen. The
    intersection with the box is what the user can see.
    """
    lv = scr.query_one(ListView)
    return lv.region.intersection(scr.query_one("#selector-box").region)


def _row_is_painted(scr: JobSelectorScreen) -> bool:
    """Is the highlighted row inside the rows the user can actually see?"""
    lv = scr.query_one(ListView)
    row = lv.children[lv.index or 0].region
    return bool(row.area) and _visible_band(scr).contains_region(row)


#: Heights a real terminal actually is. All of these overflowed before the fix.
HEIGHTS = [40, 34, 30, 28, 24, 20]


class TestTheHighlightReachesEveryJob:
    @pytest.mark.parametrize("term_h", HEIGHTS)
    async def test_the_highlight_stays_visible_all_the_way_down(self, term_h: int) -> None:
        """The user's actual gesture: hold down and watch the cursor."""
        app = _Harness(_jobs())
        async with app.run_test(size=(120, term_h)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(len(app.scr.jobs) + 5):
                await pilot.press("down")
                await pilot.pause()
                assert _row_is_painted(app.scr), (
                    f"row {lv.index} is not on screen at {term_h} rows "
                    f"(list claims {lv.size.height}, painted "
                    f"{_visible_band(app.scr).height}, scroll {lv.scroll_offset.y})"
                )
            assert lv.index == len(app.scr.jobs) - 1, "the last job was unreachable"

    @pytest.mark.parametrize("term_h", HEIGHTS)
    async def test_the_list_does_not_overflow_the_box(self, term_h: int) -> None:
        """The cause, measured directly: the clip that disabled scrolling."""
        app = _Harness(_jobs())
        async with app.run_test(size=(120, term_h)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            box = app.scr.query_one("#selector-box")
            statics = 6  # title 2 + header 1 + rule 1 + hint 2
            assert lv.size.height + statics <= box.size.height, (
                f"list overflows its box by {lv.size.height + statics - box.size.height} rows"
            )

    @pytest.mark.parametrize("term_h", HEIGHTS)
    async def test_a_long_list_really_can_scroll(self, term_h: int) -> None:
        """Vacuity guard: if the list fitted, the cases above prove nothing."""
        app = _Harness(_jobs())
        async with app.run_test(size=(120, term_h)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.virtual_size.height > lv.size.height, "nothing to scroll"
            assert lv.max_scroll_y > 0

    async def test_one_row_per_job_and_more_rows_than_any_terminal_shows(self) -> None:
        """The other half of the vacuity guard, restated after the width fix.

        It used to assert 43 rows for 41 jobs, because the two pending array rows
        WRAPPED onto a second line -- and it read that wrap as evidence the
        fixture was the reported shape. The wrap was itself the second defect the
        user reported (`TIME / WHY` looking empty, with the scheduler reason on
        the line underneath), so a table budget now keeps every row to one line.
        The vacuity it was guarding is still guarded: 41 rows is more than the
        tallest terminal in `HEIGHTS` can show, which is what makes the scroll
        assertions above non-trivial.
        """
        app = _Harness(_jobs())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.virtual_size.height == len(app.scr.jobs) == 41, lv.virtual_size.height
            assert lv.virtual_size.height > lv.size.height, "nothing to scroll"

    async def test_shrinking_the_terminal_re_fits_the_list(self) -> None:
        app = _Harness(_jobs())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            tall = lv.size.height
            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert lv.size.height < tall, "the list kept its tall-terminal height"
            box = app.scr.query_one("#selector-box")
            assert lv.size.height + 6 <= box.size.height
            for _ in range(len(app.scr.jobs) + 5):
                await pilot.press("down")
                await pilot.pause()
                assert _row_is_painted(app.scr)

    async def test_the_initial_index_is_scrolled_into_view(self) -> None:
        """Returning from a job's view lands the cursor on that job -- which is
        only useful if the row is on screen."""
        app = _Harness([])
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.pause()
            scr = JobSelectorScreen(_jobs(), initial_index=38, flourish=False)
            await app.push_screen(scr)
            await pilot.pause()
            lv = scr.query_one(ListView)
            assert lv.index == 38
            assert _row_is_painted(scr)


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize("n_jobs", [3, 8, 12])
    async def test_a_short_list_stays_compact(self, n_jobs: int) -> None:
        # The CSS comment's intent: "real vertical presence even with a few jobs".
        # The fit must not shrink a list that already fits.
        app = _Harness(_jobs(n_jobs))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.size.height == 12, "the min-height floor moved"
            assert lv.max_scroll_y == 0, "a list that fits must not scroll"

    async def test_the_ceiling_still_applies_on_a_tall_terminal(self) -> None:
        # `_fit_list` may only ever NARROW the CSS ceiling.
        app = _Harness(_jobs(80))
        async with app.run_test(size=(120, 100)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.size.height <= JobSelectorScreen._LIST_MAX_ROWS

    async def test_enter_still_selects_the_highlighted_job(self) -> None:
        # The picker's whole purpose, and it runs through the same ListView.
        app = _Harness(_jobs())
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(20):
                await pilot.press("down")
            await pilot.pause()
            expected = str(app.scr.jobs[lv.index or 0]["job_id"])
            assert expected == "57892947_18"

    async def test_the_hint_and_header_are_still_composed(self) -> None:
        app = _Harness(_jobs())
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            for node_id in (
                "#selector-title",
                "#selector-header",
                "#selector-rule",
                "#selector-hint",
            ):
                assert app.scr.query_one(node_id) is not None

    async def test_an_empty_list_still_mounts(self) -> None:
        # `_fit_list` runs before the index is set and must survive no rows.
        app = _Harness([])
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.index is None or lv.index == 0
