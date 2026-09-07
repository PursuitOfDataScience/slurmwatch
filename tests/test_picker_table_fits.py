"""The picker's table was sized to its content, with no width budget at all.

Found by driving the real picker against 73 live jobs at four terminal sizes and
auditing every row, rather than by reading the code. `_column_widths` made each
column as wide as its widest value and nothing ever asked whether the total fit:

    widths=[19, 7, 13, 9, 5, 19]  gap 3  ->  table 87 columns

against 82 usable at a 100-column terminal, 72 at 90, 60 at 78. Every row over
budget wrapped, and the wrapped part is the TAIL of the line — the `TIME / WHY`
column — which is why that column read as empty with the scheduler reason
underneath. Capping the job-id column in an earlier round only moved the
overflow: the reason text grew into the space instead.

Two measurements drove the fix, and both corrected an assumption:

* **A row widget's content width is `list_width - 4`, not `- 2`.** Measured at
  every size (89->85, 86->82, 76->72), so `ListItem { padding: 0 1 }` is not the
  whole cost. Budgeting for 2 left the table two columns over, and exactly the
  longest row wrapped — on a 120-column terminal, where there was obviously room.
* **At a wide terminal the box is sized from the CONTENT**, not from the
  terminal, so `content_w + 10` sized the box to the table exactly and the item
  padding pushed the longest row over regardless of how much room there was.
  It is `+ 12` now.

The free-text columns give ground, widest first, never below their own heading —
a column narrower than its label is unreadable. `_fit` then clips cells to the
width they are given, because `.ljust` does not enforce a budget: a 13-column
name in an 8-column slot came back 13 wide and the row overflowed anyway.

**Known limit, measured rather than hidden:** at 78 columns the six columns
cannot fit even at their heading floors (69 needed against 60 available). Getting
below ~90 needs a column to be *dropped*, which is a layout decision, not a
width one. `TestControls` pins that the floors are still respected there instead
of collapsing to nothing.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from rich.text import Text as RText
from textual.app import App
from textual.widgets import ListView, Static

from slurmwatch.tui import JobSelectorScreen


def _job(num: int, state: str = "R", **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_id": f"5790263{num % 10}_{num}",
        "state": state,
        "name": "speccurvC",
        "partition": "amd",
        "nodes": "1",
        "wall_time": "1-06:35:06",
        "reason": "JobArrayTaskLimit",
    }
    row.update(kw)
    return row


#: The reported shape: long pending array ranges plus a spread of running tasks.
JOBS: list[dict[str, Any]] = [
    _job(1, "PD", job_id="57902634_[32-48%18]"),
    _job(2, "PD", job_id="57904134_[2023-2026%18]"),
    _job(3, "PD", job_id="57902028_[37-48%18]"),
    *[_job(n) for n in range(4, 24)],
]

#: Sizes a real terminal is. 78 is below what six columns can hold — see the
#: module docstring — so it is exercised only in `TestControls`.
FITTING = [90, 100, 110, 120]


class _Harness(App[None]):
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        super().__init__()
        self._jobs_arg = jobs
        self.scr: JobSelectorScreen | None = None

    async def on_mount(self) -> None:
        self.scr = JobSelectorScreen(self._jobs_arg, reference=time.time(), flourish=False)
        self.push_screen(self.scr)


def _table_width(scr: JobSelectorScreen) -> int:
    return sum(scr._widths) + len(scr._COL_GAP) * (len(scr._widths) - 1)


class TestNoRowWraps:
    @pytest.mark.parametrize("term_w", FITTING)
    async def test_every_row_is_one_line(self, term_w: int) -> None:
        """The reported symptom, driven at the sizes a terminal really is."""
        app = _Harness(JOBS)
        async with app.run_test(size=(term_w, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            tall = [i for i, c in enumerate(lv.children) if c.size.height > 1]
            assert tall == [], (term_w, tall, app.scr._widths)

    @pytest.mark.parametrize("term_w", FITTING)
    async def test_the_table_fits_the_row_widget(self, term_w: int) -> None:
        app = _Harness(JOBS)
        async with app.run_test(size=(term_w, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            item_w = lv.children[0].size.width
            assert _table_width(app.scr) <= item_w, (term_w, _table_width(app.scr), item_w)

    @pytest.mark.parametrize("term_w", FITTING)
    async def test_each_rendered_line_is_within_the_row(self, term_w: int) -> None:
        # The line itself, not just the width arithmetic.
        app = _Harness(JOBS)
        async with app.run_test(size=(term_w, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            scr = app.scr
            item_w = scr.query_one(ListView).children[0].size.width
            for job in scr.jobs:
                width = len(RText.from_markup(scr._job_line(job, scr._widths)).plain)
                assert width <= item_w, (term_w, width, item_w, job["job_id"])

    async def test_only_the_free_text_columns_give_ground(self) -> None:
        scr = JobSelectorScreen(JOBS, reference=time.time(), flourish=False)
        full = scr._column_widths()
        tight = scr._column_widths(budget=70)
        for i, (_head, key) in enumerate(scr._COLUMNS):
            if key in scr._SHRINKABLE:
                continue
            assert tight[i] == full[i], (key, full[i], tight[i])
        assert sum(tight) + len(scr._COL_GAP) * (len(tight) - 1) <= 70

    async def test_a_narrowed_column_really_clips_its_cells(self) -> None:
        scr = JobSelectorScreen(JOBS, reference=time.time(), flourish=False)
        assert scr._fit("speccurvC", 5) == "spec…"
        assert len(scr._fit("speccurvC", 5)) == 5


class TestResizingReBudgets:
    """Found by resizing a LIVE picker rather than mounting a fresh one per size.

    `compose` and `_poll_jobs` were the only places that sized the table, so the
    first version of this fix was clean at every size it was *composed* at and
    still wrapped every row the moment the terminal changed. Measured on 23 rows
    resized 120 -> 90 with `refresh=None`: `tall=[0..22]`, all of them, with the
    columns still at their 120-column widths.
    """

    @pytest.mark.parametrize("term_w", [90, 100, 110])
    async def test_narrowing_the_terminal_keeps_every_row_on_one_line(self, term_w: int) -> None:
        app = _Harness(JOBS)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(4):
                await pilot.pause()
            assert app.scr is not None
            await pilot.resize_terminal(term_w, 24)
            for _ in range(4):
                await pilot.pause()
            lv = app.scr.query_one(ListView)
            tall = [i for i, c in enumerate(lv.children) if c.size.height > 1]
            assert tall == [], (term_w, tall, app.scr._widths, lv.children[0].size.width)

    async def test_widening_again_restores_the_full_columns(self) -> None:
        # The other direction: the box's own width is composed from the terminal
        # too, so a box left sized for 78 overflows once the budget grows.
        app = _Harness(JOBS)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(4):
                await pilot.pause()
            assert app.scr is not None
            wide = list(app.scr._widths)
            await pilot.resize_terminal(90, 24)
            for _ in range(4):
                await pilot.pause()
            assert app.scr._widths != wide, "the columns never gave ground"
            await pilot.resize_terminal(120, 40)
            for _ in range(4):
                await pilot.pause()
            assert app.scr._widths == wide, (app.scr._widths, wide)
            lv = app.scr.query_one(ListView)
            assert [c.size.height for c in lv.children] == [1] * len(JOBS)

    @pytest.mark.parametrize("term_w", [110, 120])
    async def test_a_picker_opened_small_and_then_grown_fits(self, term_w: int) -> None:
        """The gesture that gives the box's own width its teeth.

        `compose` sets `#selector-box`'s width from the terminal it was composed
        at, so a picker opened in an 84-column window and then maximised kept a
        box sized for 84 while `_table_budget` grew to the new terminal's.
        Measured with the box re-set removed: `box=73 item=69` against a table of
        81, i.e. **all 23 rows wrapped on a 120-column terminal**.
        """
        app = _Harness(JOBS)
        async with app.run_test(size=(84, 20)) as pilot:
            for _ in range(4):
                await pilot.pause()
            assert app.scr is not None
            await pilot.resize_terminal(term_w, 40)
            for _ in range(4):
                await pilot.pause()
            lv = app.scr.query_one(ListView)
            item_w = lv.children[0].size.width
            assert _table_width(app.scr) <= item_w, (_table_width(app.scr), item_w)
            tall = [i for i, c in enumerate(lv.children) if c.size.height > 1]
            assert tall == [], (term_w, tall, app.scr._widths, item_w)

    async def test_the_header_and_rule_follow_the_new_widths(self) -> None:
        # A row rendered at the new widths under a header still at the old ones is
        # a table whose columns do not line up.
        app = _Harness(JOBS)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(4):
                await pilot.pause()
            assert app.scr is not None
            await pilot.resize_terminal(90, 24)
            for _ in range(4):
                await pilot.pause()
            scr = app.scr
            head = RText.from_markup(scr._header_line(scr._widths)).plain
            rule = str(scr.query_one("#selector-rule", Static).content)
            row = RText.from_markup(scr._job_line(scr.jobs[0], scr._widths)).plain
            assert set(rule) <= {"-", " "}, rule
            assert len(rule) == len(head), (len(rule), len(head))
            assert len(row) == len(head), (len(row), len(head))


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_no_budget_means_the_old_content_sizing(self) -> None:
        scr = JobSelectorScreen(JOBS, reference=time.time(), flourish=False)
        widths = scr._column_widths()
        for (_head, key), w in zip(scr._COLUMNS, widths, strict=True):
            longest = max(len(scr._cell(j, key)) for j in JOBS)
            assert w == max(
                longest,
                len(dict(zip([k for _h, k in scr._COLUMNS], scr._headings(), strict=True))[key]),
            )

    def test_a_value_that_fits_is_padded_not_clipped(self) -> None:
        scr = JobSelectorScreen(JOBS, reference=time.time(), flourish=False)
        assert scr._fit("amd", 9) == "amd      "
        assert scr._fit("", 4) == "    "
        assert scr._fit("exact", 5) == "exact"

    def test_a_column_never_shrinks_below_its_heading(self) -> None:
        scr = JobSelectorScreen(JOBS, reference=time.time(), flourish=False)
        tight = scr._column_widths(budget=20)  # far below anything achievable
        for head, w in zip(scr._headings(), tight, strict=True):
            assert w >= len(head), (head, w)

    async def test_the_scheduler_reason_is_still_shown(self) -> None:
        app = _Harness(JOBS)
        async with app.run_test(size=(110, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            scr = app.scr
            line = RText.from_markup(scr._job_line(scr.jobs[0], scr._widths)).plain
            assert "JobArrayTaskLimit" in line or "JobArrayTaskLimi" in line, line

    async def test_the_headings_are_all_still_present(self) -> None:
        app = _Harness(JOBS)
        async with app.run_test(size=(110, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            head = RText.from_markup(app.scr._header_line(app.scr._widths)).plain
            for label in ("JOB ID", "STATE", "PARTITION", "NODES"):
                assert label in head, (label, head)

    async def test_a_terminal_too_narrow_for_six_columns_keeps_its_floors(self) -> None:
        # 78 columns cannot hold this table (69 needed at the floors, 60 usable).
        # It must degrade to the floors, not to nothing, and must not crash.
        app = _Harness(JOBS)
        async with app.run_test(size=(78, 16)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            for head, w in zip(app.scr._headings(), app.scr._widths, strict=True):
                assert w >= len(head), (head, w)

    async def test_the_cursor_is_still_on_a_real_row(self) -> None:
        app = _Harness(JOBS)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(6):
                await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert lv.index is not None
            assert 0 <= lv.index < len(app.scr.jobs)
