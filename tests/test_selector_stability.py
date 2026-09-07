"""The picker's highlight flickered when moved, because the list re-laid-out under it.

Reported after the scroll fix landed: "the highlighter flickers when getting it
moved, which looks very cheap." The highlight is not animated and Textual 8.2.8
already scrolls the cursor with `animate=False` (checked, not assumed), so the
movement itself was never the problem. The list was being rebuilt underneath it.

`Static.update()` defaults to **`layout=True`** -- a full layout pass per call --
and `_tick` called it once a second on every running row. Measured on the
reported 41-job array (39 running, 2 pending):

    one _tick()  ->  39 layout-triggering refreshes

so once a second the whole list reflowed, and a keypress-driven scroll landing in
that window is what read as flicker. Two conditions remove it, and neither costs
the live clock:

* a row whose text has not moved is not touched at all (0 refreshes for a tick
  inside the same wall second, which is most of them around a keypress);
* a row whose text changed but kept its WIDTH gets `layout=False` -- the size
  provably cannot have changed, so one row repaints and nothing reflows.

Only a real width change asks for layout. A growing clock cannot cause one --
`_fit` clips every cell to its column width -- but a RESIZE can: `on_resize`
re-budgets the columns from the terminal, so a tick can meet rows still recorded
at the previous widths. `test_a_wider_line_still_asks_for_layout` drives that
branch through exactly that, the width comparison being the thing that keeps a
re-widthed row from being repainted at the wrong size.

The second half is the 3-second poll. Tearing the list down and rebuilding it is
four mutations, and each used to reach the screen alone: 5 layout passes for one
poll in which three array tasks finished, with the reader watching the list
resize and then the cursor land. `batch_update` defers the repaints so the screen
shows the finished state instead of the steps -- measured 5 -> 2, and the
intermediate frame carrying the OLD row count is gone.

What must not change is in `TestControls`: the clock still advances, pending rows
still hold their reason, and a poll still keeps the cursor on the same job with
its row on screen.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
from textual.app import App
from textual.geometry import Region
from textual.widgets import ListView, Static

from slurmwatch.tui import JobSelectorScreen


def _jobs(n: int = 41) -> list[dict[str, Any]]:
    """The reported shape: two pending array rows, then running tasks."""
    out: list[dict[str, Any]] = [
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


N_RUNNING = 39


class _Counter:
    """Counts `Static.refresh` calls, split by whether they force a layout."""

    def __init__(self) -> None:
        self.layout = 0
        self.plain = 0

    def reset(self) -> None:
        self.layout = 0
        self.plain = 0

    @property
    def total(self) -> int:
        return self.layout + self.plain


@pytest.fixture
def refreshes() -> Iterator[_Counter]:
    counter = _Counter()
    original = Static.refresh

    def counting(self: Static, *regions: Region, layout: bool = False, **kw: Any) -> Any:
        if layout:
            counter.layout += 1
        else:
            counter.plain += 1
        return original(self, *regions, layout=layout, **kw)

    with mock.patch.object(Static, "refresh", counting):
        yield counter


class _Harness(App[None]):
    def __init__(self, jobs: list[dict[str, Any]], reference: float | None) -> None:
        super().__init__()
        self._jobs = jobs
        self._reference = reference
        self.scr: JobSelectorScreen | None = None

    async def on_mount(self) -> None:
        # `refresh=None`, so no poll timer is armed and a test drives `_poll_jobs`
        # itself. An earlier draft passed one and the 0.3s timer fired mid-test,
        # so the "before" reading was already a post-poll list.
        self.scr = JobSelectorScreen(self._jobs, reference=self._reference, flourish=False)
        self.push_screen(self.scr)


class TestTheTickDoesNotReLayOutTheList:
    async def test_a_tick_inside_the_same_second_touches_nothing(self, refreshes: _Counter) -> None:
        now = time.time()
        app = _Harness(_jobs(), now)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            refreshes.reset()
            with mock.patch("slurmwatch.tui.time.time", return_value=now):
                assert app.scr is not None
                app.scr._tick()
            await pilot.pause()
            assert refreshes.total == 0, (
                f"{refreshes.total} refreshes for a tick where no clock moved"
            )

    async def test_a_tick_that_moves_every_clock_forces_no_layout(
        self, refreshes: _Counter
    ) -> None:
        now = time.time()
        app = _Harness(_jobs(), now)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            refreshes.reset()
            with mock.patch("slurmwatch.tui.time.time", return_value=now + 1):
                assert app.scr is not None
                app.scr._tick()
            await pilot.pause()
            assert refreshes.layout == 0, f"{refreshes.layout} layout passes in one tick"
            assert refreshes.plain == N_RUNNING, refreshes.plain

    async def test_a_wider_line_still_asks_for_layout(self, refreshes: _Counter) -> None:
        """The guard's own branch, driven by the width change that really happens.

        An earlier version drove it by shrinking the TIME column to 4 and letting
        the elapsed clock grow past it, on the reasoning that `.ljust(4)` stops
        padding so the line gets wider. `_fit` now CLIPS a cell to the width it is
        given (that is what fixed the wrapped rows), so a growing clock can no
        longer widen a line and that driver stopped reaching the branch.

        The reachable cause is a widths change between the recorded text and the
        tick: `on_resize` re-budgets the columns from the terminal, and a tick can
        land on rows still recorded at the old widths. So seed the text at the
        wide widths, narrow the columns as a resize does, and tick -- the line
        really is a different width, and layout is the right answer.
        """
        now = time.time()
        app = _Harness(_jobs(4), now)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            # Recorded at the WIDE widths, as compose left them.
            app.scr._row_text = [app.scr._job_line(j, app.scr._widths) for j in app.scr.jobs]
            wide = len(app.scr._row_text[-1])
            app.scr._widths = [*app.scr._widths[:-1], 4]
            refreshes.reset()
            with mock.patch("slurmwatch.tui.time.time", return_value=now + 4000):
                app.scr._tick()
            await pilot.pause()
            assert len(app.scr._row_text[-1]) != wide, "the line width did not change"
            assert refreshes.layout == 2, (
                f"a re-widthed row must ask for layout; got layout={refreshes.layout} "
                f"plain={refreshes.plain}"
            )


class TestTheRebuildRepaintsOnce:
    async def test_the_poll_is_batched(self) -> None:
        jobs_a = _jobs()
        jobs_b = [j for j in jobs_a if j["job_id"] not in ("57892947_3", "57892947_4")]
        seen: list[int] = []
        app = _Harness(jobs_a, time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            screen = app.screen
            real_layout = type(screen)._refresh_layout

            def spy(self: Any, *a: Any, **kw: Any) -> Any:
                seen.append(len(lv.children))
                return real_layout(self, *a, **kw)

            app.scr._refresh = lambda: jobs_b
            with mock.patch.object(type(screen), "_refresh_layout", spy):
                await app.scr._poll_jobs()
                await pilot.pause()
            assert seen, "no layout pass was observed at all"
            # The old row count must never reach the screen mid-rebuild.
            assert len(jobs_a) not in seen, f"an intermediate frame leaked: {seen}"
            assert len(seen) <= 3, f"{len(seen)} layout passes for one poll: {seen}"

    async def test_the_rebuild_still_replaces_the_rows(self) -> None:
        jobs_a = _jobs()
        jobs_b = [j for j in jobs_a if j["job_id"] != "57892947_3"]
        app = _Harness(jobs_a, time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            app.scr._refresh = lambda: jobs_b
            await app.scr._poll_jobs()
            await pilot.pause()
            assert len(lv.children) == len(jobs_b)
            assert len(app.scr._row_text) == len(jobs_b)
            assert app.scr._rendered_key == {
                (str(j["job_id"]), str(j.get("state", ""))) for j in jobs_b
            }


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    async def test_the_clock_still_advances(self) -> None:
        # The whole point of `_tick`. Skipping unchanged rows must not mean
        # skipping changed ones.
        now = time.time()
        app = _Harness(_jobs(), now)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            before = app.scr._row_text[5]
            with mock.patch("slurmwatch.tui.time.time", return_value=now + 65):
                app.scr._tick()
            await pilot.pause()
            after = app.scr._row_text[5]
            assert after != before, "the TIME column stopped ticking"
            assert "8:56" in after, after  # 7:51 + 65s

    async def test_the_pending_rows_are_never_touched(self, refreshes: _Counter) -> None:
        """Pre-existing behaviour -- passes in BOTH states, verified by neutering.

        `_tick` skipped pending rows before this change too, so this cannot
        distinguish the fix. Kept as a control: the skip is what makes the
        `N_RUNNING` counts in the tests above mean what they say, and a future
        change that started ticking a pending row's static reason would both
        break the count and put a clock where a scheduler reason belongs.
        """
        now = time.time()
        app = _Harness(_jobs(), now)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            before = list(app.scr._row_text[:2])
            refreshes.reset()
            with mock.patch("slurmwatch.tui.time.time", return_value=now + 600):
                app.scr._tick()
            await pilot.pause()
            assert refreshes.total == N_RUNNING, "a pending row was updated"
            assert app.scr._row_text[:2] == before, "a pending row's text moved"

    async def test_a_pending_row_still_shows_its_reason(self) -> None:
        app = _Harness(_jobs(), time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            assert "JobArrayTaskLimit" in app.scr._row_text[0]

    async def test_a_poll_still_keeps_the_cursor_on_the_same_job(self) -> None:
        jobs_a = _jobs()
        jobs_b = [
            j for j in jobs_a if j["job_id"] not in ("57892947_3", "57892947_4", "57892947_5")
        ]
        app = _Harness(jobs_a, time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            box = app.scr.query_one("#selector-box")
            for _ in range(20):
                await pilot.press("down")
            await pilot.pause()
            before = str(app.scr.jobs[lv.index or 0]["job_id"])
            app.scr._refresh = lambda: jobs_b
            await app.scr._poll_jobs()
            await pilot.pause()
            after = str(app.scr.jobs[lv.index or 0]["job_id"])
            assert after == before, f"the cursor moved from {before} to {after}"
            # ...and the row it landed on is still on screen.
            row = lv.children[lv.index or 0].region
            assert lv.region.intersection(box.region).contains_region(row)

    async def test_an_unchanged_job_set_does_not_rebuild(self) -> None:
        jobs = _jobs()
        app = _Harness(jobs, time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            lv = app.scr.query_one(ListView)
            first = lv.children[0]
            app.scr._refresh = lambda: list(jobs)
            await app.scr._poll_jobs()
            await pilot.pause()
            assert lv.children[0] is first, "the list was rebuilt for nothing"

    async def test_a_failing_poll_does_not_commit_the_key(self) -> None:
        jobs = _jobs()
        app = _Harness(jobs, time.time())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.scr is not None
            before = set(app.scr._rendered_key)

            def boom() -> list[dict[str, Any]]:
                raise RuntimeError("squeue fell over")

            app.scr._refresh = boom
            await app.scr._poll_jobs()
            await pilot.pause()
            assert app.scr._rendered_key == before
