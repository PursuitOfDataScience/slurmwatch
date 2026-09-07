"""Pressing Enter could open a job other than the highlighted one.

Reported: "hit enter to a job ... after you see all the job details and would
like to return back to this interface, the highlightor isn't on this job anymore
but somewhere else."

`_poll_jobs` published `self.jobs = new_jobs` **before** the `await`s that rebuild
the rows. `action_select_job` reads `self.jobs[lv.index]`, so between those two
points the model was the NEW list while the rows on screen were still the OLD
ones. Measured on a 20-job list where three finished:

    the row under the cursor showed  57850045
    self.jobs[lv.index] was          57850048

so Enter dismissed with `57850048`. The selector loop then computed the next
picker's `initial_index` from *that* job, which is why the cursor came back
somewhere the user had never put it. `_tick` broke in the same window and for the
same reason, raising `ValueError: zip() argument 2 is longer than argument 1`,
because it zips `self.jobs` against `self._rows` with `strict=True`.

The window is small but the poll runs every 3 seconds, `lv.clear()` and
`lv.extend()` of ~90 rows are real async work, and a poll cancelled by an
overlapping slow `squeue` leaves the desync standing until the next one.

The fix computes every field that has to agree with the rows into a local and
swaps them together after the rows exist, with no `await` between the publish and
the screen matching it. That is the discipline the method already applied to
`_rendered_key` ("commit ONLY after the rebuild actually completed"), extended to
`jobs`, `_rows`, `_row_text` and `_widths`. A cancelled poll now publishes
nothing, so the previous list and its rows stay paired.

`_reference` is deliberately still set first: it is a timestamp, not list-shaped,
and `_job_line` needs the fresh one to render the new rows' TIME column. Nothing
indexes it, so it cannot pair a row with the wrong job.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from textual.app import App
from textual.widgets import ListView

from slurmwatch.tui import JobSelectorScreen


def _job(num: int, state: str = "R") -> dict[str, Any]:
    return {
        "job_id": str(num),
        "state": state,
        "name": f"exp-{num}",
        "partition": "gpu",
        "nodes": "1",
        "wall_time": "1-06:35:06",
        "reason": "None",
    }


LIST_A: list[dict[str, Any]] = [_job(57850040 + k) for k in range(20)]
#: Three finished and two were submitted -- an ordinary few minutes on an array.
LIST_B: list[dict[str, Any]] = [*LIST_A[3:], _job(57899001), _job(57899002)]
CURSOR_AT = 5
TARGET = LIST_A[CURSOR_AT]["job_id"]


class _Harness(App[None]):
    def __init__(self, refresh: Any = None) -> None:
        super().__init__()
        self._refresh_arg = refresh
        self.scr: JobSelectorScreen | None = None

    async def on_mount(self) -> None:
        self.scr = JobSelectorScreen(
            LIST_A,
            initial_index=CURSOR_AT,
            reference=time.time(),
            refresh=self._refresh_arg,
            flourish=False,
        )
        self.push_screen(self.scr)


def _consistent(scr: JobSelectorScreen, lv: ListView) -> list[str]:
    """Every way the model can disagree with what is on screen."""
    problems: list[str] = []
    if len(scr.jobs) != len(lv.children):
        problems.append(f"jobs={len(scr.jobs)} rows_on_screen={len(lv.children)}")
    if len(scr._rows) != len(scr.jobs):
        problems.append(f"_rows={len(scr._rows)} jobs={len(scr.jobs)}")
    if len(scr._row_text) != len(scr.jobs):
        problems.append(f"_row_text={len(scr._row_text)} jobs={len(scr.jobs)}")
    if lv.index is not None and 0 <= lv.index < len(scr.jobs):
        model = str(scr.jobs[lv.index]["job_id"])
        text = scr._row_text[lv.index] if lv.index < len(scr._row_text) else ""
        if model not in text:
            problems.append(f"cursor model={model} but its row reads {text[:40]!r}")
    return problems


class TestTheModelNeverNamesADifferentJobThanTheRow:
    async def test_enter_inside_the_rebuild_window_opens_the_displayed_job(self) -> None:
        """The reported failure, with the window deterministically held open.

        Sampling on ordinary `pause()` ticks does NOT land inside it -- the
        window is microseconds on a 20-row list -- so `lv.extend` is parked on an
        event here and the keypress happens while the poll is mid-rebuild. That
        is the state a 90-row list and a 3-second poll reach on their own.
        """
        app = _Harness()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            scr = app.scr
            lv = scr.query_one(ListView)
            displayed = scr._row_text[CURSOR_AT]
            released = asyncio.Event()
            real_clear = lv.clear
            opened: list[str] = []

            # Parked on `clear`, not `extend`. The dangerous stretch is BEFORE the
            # list is torn down: that is where `self.jobs` used to be published
            # while the old rows were still on screen and still highlighted. Once
            # `clear()` has run the view is legitimately empty and there is no
            # highlight to be wrong about.
            async def parked() -> None:
                await released.wait()
                await real_clear()

            lv.clear = parked  # type: ignore[assignment, method-assign]

            def record(value: str = "") -> Any:
                opened.append(str(value))

            scr.dismiss = record  # type: ignore[assignment, method-assign]
            scr._refresh = lambda: LIST_B
            task = asyncio.create_task(scr._poll_jobs())
            for _ in range(5):  # let it reach the parked clear
                await pilot.pause()

            # INSIDE the window: the model must still name the displayed job.
            assert _consistent(scr, lv) == [], _consistent(scr, lv)
            scr.action_select_job()
            assert opened, "Enter did not dismiss"
            assert opened[0] in displayed, (
                f"Enter opened {opened[0]} while the highlighted row showed {displayed[:40]!r}"
            )
            assert opened[0] == TARGET

            released.set()
            await task
            await pilot.pause()

    async def test_a_cancelled_poll_publishes_nothing(self) -> None:
        """A poll killed mid-rebuild must leave the old list and rows paired."""
        app = _Harness()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            scr = app.scr
            lv = scr.query_one(ListView)
            before_jobs = list(scr.jobs)
            before_key = set(scr._rendered_key)

            async def die() -> None:
                raise asyncio.CancelledError

            scr._refresh = lambda: LIST_B
            lv.clear = die  # type: ignore[assignment, method-assign]
            with pytest.raises(asyncio.CancelledError):
                await scr._poll_jobs()
            assert scr.jobs == before_jobs, "the new list was published anyway"
            assert scr._rendered_key == before_key
            assert _consistent(scr, lv) == []

    async def test_the_tick_survives_a_poll(self) -> None:
        # `_tick` zips `self.jobs` against `self._rows` with strict=True, so a
        # length desync raised ValueError rather than mis-rendering.
        app = _Harness(refresh=lambda: LIST_B)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            for _ in range(30):
                await pilot.pause()
                app.scr._tick()  # must not raise at any point


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    async def test_enter_dismisses_the_job_the_row_shows(self) -> None:
        """Passes in BOTH states -- verified by neutering, so a control.

        Ordinary `pause()` ticks never land inside the rebuild window on a
        20-row list, so this cannot distinguish the fix; the window-held
        test above is the one with teeth. Kept because it pins the settled
        state after a poll, which is what the reader actually sees.
        """
        app = _Harness(refresh=lambda: LIST_B)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):  # let the poll rebuild from LIST_B
                await pilot.pause()
            shown = app.scr._row_text[lv.index or 0]
            model = str(app.scr.jobs[lv.index or 0]["job_id"])
            assert model in shown, (model, shown)
            assert model == TARGET, "the cursor left the job it started on"

    async def test_it_holds_at_every_step_of_a_poll(self) -> None:
        app = _Harness(refresh=lambda: LIST_B)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            violations: list[str] = []
            for _ in range(40):
                await pilot.pause()
                violations.extend(_consistent(app.scr, lv))
            assert violations == [], violations[:5]

    async def test_the_poll_still_adopts_the_new_list(self) -> None:
        app = _Harness(refresh=lambda: LIST_B)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            for _ in range(30):
                await pilot.pause()
            assert [str(j["job_id"]) for j in app.scr.jobs] == [str(j["job_id"]) for j in LIST_B]
            assert app.scr._rendered_key == {
                (str(j["job_id"]), str(j.get("state", ""))) for j in LIST_B
            }

    async def test_the_cursor_still_follows_the_same_job(self) -> None:
        app = _Harness(refresh=lambda: LIST_B)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):
                await pilot.pause()
            assert lv.index == 2, lv.index  # three jobs ahead of it finished
            assert str(app.scr.jobs[lv.index]["job_id"]) == TARGET

    async def test_an_unchanged_key_set_still_early_returns(self) -> None:
        app = _Harness()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            first = lv.children[0]
            app.scr._refresh = lambda: list(LIST_A)
            await app.scr._poll_jobs()
            await pilot.pause()
            assert lv.children[0] is first, "rebuilt for nothing"

    async def test_a_failing_refresh_still_changes_nothing(self) -> None:
        app = _Harness()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            before = list(app.scr.jobs)

            def boom() -> list[dict[str, Any]]:
                raise RuntimeError("squeue fell over")

            app.scr._refresh = boom
            await app.scr._poll_jobs()
            await pilot.pause()
            assert app.scr.jobs == before

    async def test_the_header_and_rule_still_resize_with_the_list(self) -> None:
        # They are published in the same swap now, so they must still update.
        wide = [*LIST_A, _job(57899003)]
        wide[-1]["name"] = "a-considerably-longer-job-name"
        app = _Harness(refresh=lambda: wide)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            # PAUSE the screen's own poll timer before reading `before`. Driving
            # `_poll_jobs` by hand (which this test already does) removed the wait
            # on `_REFRESH_S` = 3 wall-clock seconds but not the INTERLEAVING: under
            # load, three seconds elapse during mount, the timer's own poll lands
            # first and publishes the wide widths, so `before` already holds them
            # and the explicit poll below changes nothing. That is why the failure
            # printed two IDENTICAL width lists (`[8,7,30,9,5,10] != [8,7,30,9,5,10]`)
            # while passing 12/12 alone. Patching the method is not enough:
            # `set_interval` was handed a BOUND method, so the timer keeps calling
            # the original -- pausing the Timer is what stops it.
            for timer in list(app.scr._timers):
                if getattr(timer._callback, "__name__", "") == "_kick_poll":
                    timer.pause()
            before = list(app.scr._widths)
            await app.scr._poll_jobs()
            for _ in range(4):
                await pilot.pause()
            assert app.scr._widths != before, "widths never re-sized"
            assert len(app.scr._row_text) == len(wide)

    def test_column_widths_still_defaults_to_the_published_list(self) -> None:
        scr = JobSelectorScreen(LIST_A, flourish=False)
        assert scr._column_widths() == scr._column_widths(LIST_A)
        assert scr._column_widths(LIST_B) != []


#: The cursor's job finished while the user was in its dashboard.
GONE: list[dict[str, Any]] = [j for j in LIST_A if j["job_id"] != TARGET]
#: ...and the list also shrank past where the cursor was.
GONE_SHORT: list[dict[str, Any]] = GONE[:3]


class TestAVanishedJobStillLeavesACursor:
    """The cursor's job finishing must not leave the list with no highlight.

    `lv.clear()` sets `index` to None and the id lookup then found nothing to put
    it back, so returning from a job that had finished gave rows and **no
    highlight at all** -- measured, `lv.index is None` on a 19-row list. Arrow
    keys start from nowhere and the reader cannot see where they are.

    The position is held instead of the job: the row that now occupies where the
    cursor was, clamped to the end. That is what a list does when the selected
    item is deleted.
    """

    async def test_the_highlight_does_not_disappear(self) -> None:
        app = _Harness(refresh=lambda: GONE)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):
                await pilot.pause()
            assert app.scr.jobs == GONE, "the poll did not adopt the new list"
            assert lv.index is not None, "rows on screen and no cursor"
            assert 0 <= lv.index < len(app.scr.jobs)

    async def test_it_holds_the_position_it_was_on(self) -> None:
        app = _Harness(refresh=lambda: GONE)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):
                await pilot.pause()
            assert lv.index == CURSOR_AT

    async def test_it_clamps_when_the_list_shrank_past_that_row(self) -> None:
        app = _Harness(refresh=lambda: GONE_SHORT)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):
                await pilot.pause()
            assert lv.index == len(GONE_SHORT) - 1, lv.index
            assert _consistent(app.scr, lv) == []

    async def test_an_empty_list_keeps_no_cursor(self) -> None:
        # Nothing to point at is the one case where None is right.
        app = _Harness(refresh=lambda: [])
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            for _ in range(30):
                await pilot.pause()
            assert app.scr.jobs == []
            assert lv.index is None
