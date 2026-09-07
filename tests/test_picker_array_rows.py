"""A pending array row could not be opened, and its WHY column looked empty.

Two reports, one shape of row: the single line `squeue` prints for a whole
unstarted array, `57902634_[31-48%18]`.

**Could not be opened.** `scontrol show job 57902634_[31-48%18]` answers
`Invalid job id specified` (measured), so selecting a row the picker had just
drawn exited with `sw: Job 57902634_[31-48%18] not found`. The command line has
resolved this since `cli._job_id_without_array_range`, whose docstring calls the
old refusal "a closed loop" because "pasting what squeue printed is the entire
way anyone arrives here". Selecting it in the picker is the same arrival by a
different door -- and it was the door without the resolution. The resolver now
lives in `slurm.array_range_base`, pure, with `cli` delegating to it so there is
ONE pattern rather than two that can tolerate different shapes.

**WHY looked empty.** It was not empty; it was on the next line. Measured with
the reported rows, a 23-character id (`57904134_[2023-2026%18]`) makes the table
85 columns, `ListItem { padding: 0 1 }` leaves `list_width - 2`, and the box is
capped by `max-width: 96%` -- so at a 100-column terminal the row overflowed by
ONE character and wrapped, putting the tail of the line (the `TIME / WHY` column)
underneath. Every pending row showed a blank WHY with `(JobArrayTaskLimit)`
below it:

    term_w=100  box=86  list=86  table=85  row_heights=[2, 2, 2, 1]

`_elide_job_id` caps the column, the same remedy `_elide_job_name` already
applies to names for the same reason. It elides the RANGE and keeps the brackets
BALANCED -- `57904134_[…]`, not `57904134_[2023-202…`, which would read as a
different, narrower range and hand an unbalanced `[` to a markup parser. Eliding
is display-only: `action_select_job` reads the id out of `self.jobs`, never off
the screen.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest import mock

import pytest
from rich.text import Text as RText
from textual.app import App
from textual.widgets import ListView

from slurmwatch import tui
from slurmwatch.cli import _job_id_without_array_range
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.slurm import array_range_base
from slurmwatch.tui import JobSelectorScreen, _elide_job_id

#: Forms squeue really prints, including the two truncations `SLURM_BITSTR_LEN`
#: produces (the closing bracket may be absent; the tail may be `...`).
RANGES = {
    "57902634_[31-48%18]": "57902634",
    "57904134_[2023-2026%18]": "57904134",
    "56622046_[0-30,32-44,47-83...]": "56622046",
    "56622046_[0-30,32-44,47-83,85-8": "56622046",
}
#: Not ranges: a started task, a bare id, and shapes Slurm does not print.
NOT_RANGES = ["57902634_31", "57902634", "12345_[]", "12345_[bogus]", "abc", ""]


def _job(job_id: str, state: str = "PD", **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_id": job_id,
        "state": state,
        "name": "engonly",
        "partition": "amd",
        "nodes": "1",
        "reason": "JobArrayTaskLimit",
    }
    row.update(kw)
    return row


REPORTED = [
    _job("57902634_[32-48%18]"),
    _job("57902028_[37-48%18]"),
    _job("57904134_[2023-2026%18]"),
    _job("57902634_31", state="R", wall_time="0:13"),
]


def _plain(markup: str) -> str:
    return RText.from_markup(markup).plain.replace("\\[", "[")


class _Harness(App[None]):
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        super().__init__()
        self._jobs_arg = jobs
        self.scr: JobSelectorScreen | None = None

    async def on_mount(self) -> None:
        self.scr = JobSelectorScreen(self._jobs_arg, reference=time.time(), flourish=False)
        self.push_screen(self.scr)


class TestAnArrayRangeResolvesToItsJob:
    @pytest.mark.parametrize(("text", "base"), sorted(RANGES.items()))
    def test_the_base_is_recovered(self, text: str, base: str) -> None:
        assert array_range_base(text) == base

    @pytest.mark.parametrize("text", NOT_RANGES)
    def test_a_non_range_is_left_alone(self, text: str) -> None:
        assert array_range_base(text) is None

    def test_the_pattern_has_exactly_one_definition(self) -> None:
        """Two copies could tolerate different shapes -- and did.

        `cli` had its own `_ARRAY_RANGE_RE`; the picker had none, so it accepted
        nothing. Asserted as "one definition in the package" rather than "the two
        names are identical", because keeping a second NAME is what
        `test_no_dead_import_aliases` (rightly) rejects -- it caught exactly that
        when this fix first left an alias behind.
        """
        import pathlib as _p

        src = _p.Path(__file__).resolve().parent.parent / "src" / "slurmwatch"
        needle = "(?P<range>"  # the array-range pattern specifically
        defs = sorted(f.name for f in src.glob("*.py") if needle in f.read_text())
        assert defs == ["slurm.py"], defs

    @pytest.mark.parametrize(("text", "base"), sorted(RANGES.items()))
    def test_the_command_line_path_still_agrees(self, text: str, base: str) -> None:
        assert _job_id_without_array_range(text) == base

    async def test_the_picker_queries_the_base_not_the_range(self) -> None:
        """The reported failure: the picker used to pass the bracket string on.

        Driven against `SlurmwatchApp`, which is where `_open_job` lives -- an
        earlier draft used a bare `App` harness, so the call raised
        `AttributeError` and a too-broad `pytest.raises(Exception)` hid it while
        the id list stayed empty.
        """
        asked: list[str] = []
        shown: list[Any] = []

        def fake_pending(job_id: str) -> Any:
            asked.append(job_id)
            return object()  # a stand-in PendingJob; the id is what matters here

        app = tui.SlurmwatchApp(jobs=REPORTED, config=SlurmwatchConfig())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with (
                mock.patch.object(tui, "resolve_pending_job", fake_pending),
                mock.patch.object(tui, "PendingScreen", lambda *a, **k: shown.append(a)),
                mock.patch.object(app, "push_screen_wait", mock.AsyncMock()),
            ):
                ok = await app._open_job(
                    "57902634_[32-48%18]", REPORTED, asyncio.get_running_loop()
                )
        assert asked == ["57902634"], asked
        assert ok is True


class TestThePendingRowFitsOnOneLine:
    def test_the_reason_lands_in_the_time_why_column(self) -> None:
        scr = JobSelectorScreen(REPORTED, reference=time.time(), flourish=False)
        scr._widths = scr._column_widths()
        line = _plain(scr._job_line(REPORTED[2], scr._widths))
        header = _plain(scr._header_line(scr._widths))
        assert "JobArrayTaskLimit" in line, line
        # It sits under the heading, not on a line of its own.
        assert line.index("JobArrayTaskLimit") >= header.index("TIME / WHY") - 1, (
            header,
            line,
        )

    def test_the_table_fits_the_list_including_item_padding(self) -> None:
        scr = JobSelectorScreen(REPORTED, reference=time.time(), flourish=False)
        widths = scr._column_widths()
        table_w = sum(widths) + len(scr._COL_GAP) * (len(widths) - 1)
        assert table_w <= tui._JOB_ID_MAX + 62, table_w  # the reported shape's budget

    @pytest.mark.parametrize(
        ("text", "shown"),
        [
            ("57904134_[2023-2026%18]", "57904134_[…]"),
            ("56622046_[0-30,32-44,47-83,85-8", "56622046_[…]"),
        ],
    )
    def test_a_long_range_keeps_balanced_brackets(self, text: str, shown: str) -> None:
        assert _elide_job_id(text) == shown

    def test_the_ascii_form_uses_dots(self) -> None:
        assert _elide_job_id("57904134_[2023-2026%18]", True) == "57904134_[...]"


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize("term_w", [100, 110, 120, 140])
    async def test_no_row_wraps(self, term_w: int) -> None:
        """Passes in BOTH states in this harness -- verified by neutering.

        The wrap WAS measured directly on the reported rows (`term_w=100`,
        box=86, list=86, table=85 -> `row_heights=[2, 2, 2, 1]` before, all 1
        after), but it does not reproduce through `run_test` here, so this cannot
        be the teeth for the elision -- `test_the_table_fits_the_list_including_item_padding`
        is, and it reddens when the cap is removed. Kept because a wrapped row is
        the user-visible symptom and this is the assertion that names it.
        """
        app = _Harness(REPORTED)
        async with app.run_test(size=(term_w, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            heights = [c.size.height for c in lv.children]
            assert heights == [1] * len(REPORTED), (term_w, heights)

    @pytest.mark.parametrize("text", ["57902634_[32-48%18]", "57902634_31", "57902634"])
    def test_an_id_that_already_fits_is_untouched(self, text: str) -> None:
        assert _elide_job_id(text) == text

    def test_eliding_does_not_change_what_gets_opened(self) -> None:
        # `action_select_job` reads `self.jobs`, never the screen, so a shortened
        # display must not change the id handed on.
        scr = JobSelectorScreen(REPORTED, reference=time.time(), flourish=False)
        assert str(scr.jobs[2]["job_id"]) == "57904134_[2023-2026%18]"

    def test_a_started_task_is_still_opened_as_itself(self) -> None:
        assert array_range_base("57902634_31") is None

    def test_a_running_row_still_shows_its_elapsed_time(self) -> None:
        scr = JobSelectorScreen(REPORTED, reference=None, flourish=False)
        scr._widths = scr._column_widths()
        assert "0:13" in _plain(scr._job_line(REPORTED[3], scr._widths))

    def test_the_heading_still_says_time_and_why(self) -> None:
        # The label is right -- both halves are real, and now both are visible.
        scr = JobSelectorScreen(REPORTED, reference=time.time(), flourish=False)
        assert "TIME / WHY" in _plain(scr._header_line(scr._column_widths()))

    def test_the_name_elision_is_unchanged(self) -> None:
        from slurmwatch.tui import _elide_job_name

        assert _elide_job_name("short") == "short"
        assert _elide_job_name("x" * 200).endswith("…")

    async def test_every_reported_row_is_still_listed(self) -> None:
        app = _Harness(REPORTED)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            assert len(lv.children) == len(REPORTED)
