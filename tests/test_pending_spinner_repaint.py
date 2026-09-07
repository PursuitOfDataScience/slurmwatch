"""The pending screen rewrote its whole panel eight times a second.

Reported from use: in the `Where It Could Run` table, "all these YES are
flickering". They were not wrong and they were not animated -- they were being
**re-sent to the terminal 8 times a second** while the "calculating…" spinner
turned.

The panel is ONE `Static`: `PendingView.render()` returns why the job waits, when
it might start, and the WHERE table as a single string. `_tick_spinner` runs at
`set_interval(0.12, ...)` and ended in a bare `view.refresh()`, which dirties the
widget's whole region -- and Textual's partial update converts a dirty region to
spans and re-emits **every line in it, without comparing content**.

Measured on a **110x40** terminal, on the panel this module mounts: a pending job
the scheduler has not yet planned (the only state the spinner runs in), reason
`Priority`, and a `Where It Could Run` table listing `build` (current, 0 idle
nodes) and `broadwl` (53 idle nodes, 1101 idle cores, verdict `YES`) with the
complete requeue tip under it -- **17 rows**. Eight ticks of the real 0.12 s
timer:

    bare view.refresh():   41,016 bytes to the terminal, 8 compositor updates
    scoped repaint:         2,856 bytes to the terminal, 8 compositor updates
    lines that actually moved, per tick:  1 of 17  (the spinner's)
    lines that moved in the WHERE section: 0
    spinner frames reaching the terminal:  8 of 8, either way

So a table whose text was byte-identical on every frame was redrawn on every
frame, and scoping the repaint to the lines that moved is a **14x** cut with the
animation unchanged. The same figures are in `CHANGELOG.md`; both were taken on
the panel described above, which is why they are not the numbers an earlier draft
of each carried (those two disagreed with each other, having measured different
panels).

Two things this is NOT, both measured rather than assumed:

* not the layout-pass storm that `JobSelectorScreen._tick` was fixed for --
  `Widget.refresh` defaults to `layout=False`, so a spinner tick produced **0**
  layout passes. The volume was the problem, not the reflow.
* not a wrong verdict: `YES ▸` is byte-identical frame to frame.

The 10-second poll is the second half. `_refresh_once` ended in
`refresh(layout=True)` unconditionally, so a poll that resolved the same numbers
re-laid-out the panel anyway (measured `layout=1, plain=0` per poll). It now
repaints only when the markup changed -- the guard the selector screen already
states -- and that is where the win is: an unchanged panel does nothing at all.

Asking for layout only when `markup.count("\n")` changed was a **regression**,
and `TestAChangedPanelIsAlwaysRelaidOut` is why. The newline count is not the
panel's height, because this panel WRAPS: the reason code, the estimate prose and
the requeue tip are single long lines that the `Static` folds to the widget's
width. Measured at 70 columns (a 60-column content box), a controller changing
`Reason` from `Priority` to `ReqNodeNotAvail, UnavailableNodes:midway3-0001,...
,midway3-0025`:

    newlines in the markup:   14 -> 14      (so no layout pass was requested)
    rows the content needs:   22 -> 28
    rows the widget kept:     22            -- the last SIX were clipped

and it persisted: `_repaint_spinner` renders only the stale 22-row region, so its
own `len(was) != len(painted)` height check could not see the growth either. What
was clipped was the whole `Where It Could Run` verdict table. Any change now asks
for layout.

`TestControls` pins what must not move: the spinner still advances on every tick,
a planned job still animates nothing at all, a poll that changes the panel still
repaints it, and the panel's text is unchanged by any of this.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from textual.app import App
from textual.geometry import Region
from textual.widgets import Static

from slurmwatch import pending as pmod
from slurmwatch import tui
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.pending import PartitionResources
from slurmwatch.tui import PendingScreen, PendingView

#: The user may submit to both, and `broadwl` names a QOS after itself, so the tip
#: under the table is complete. Same shape as `test_pending_assoc_off_render`'s.
ASSOC = {"build": ["build"], "broadwl": ["broadwl"]}


def _parts() -> list[PartitionResources]:
    """The current partition (full) plus one alternative with room.

    Borrowed from `test_pending_assoc_off_render._parts` rather than invented: it
    is the shape that makes the WHERE table say something -- one row that reads
    `waiting (current)` and one that reads `YES`. THIS FILE'S SUBJECT IS THAT
    TABLE being re-sent 8 times a second, so it has to exist. With
    `resolve_cluster_partitions` stubbed to `[]` (as it was) the panel says
    `cluster partition info unavailable` instead, and the assertion that the
    verdict rows are identical frame to frame was filtering an empty panel: `[] ==
    []`, five times, passing with the fix removed.
    """
    return [
        PartitionResources(
            "build", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
        ),
        PartitionResources("broadwl", True, idle_nodes=53, cpus_idle=1101, max_node_cpus=48),
    ]


def _unplanned() -> Any:
    """A pending job with no start estimate -- the state the spinner runs in."""
    job = pmod._mock_pending_job("57902634")
    job.start_time_estimate = None
    # `build` and a GPU-free 4-CPU request, so `_parts()` above can answer for it:
    # the mock job asks for 2 A100s, which neither partition has, and every verdict
    # row would read `no gpu` rather than the `YES` this file is about.
    job.reason, job.req_gpus, job.req_cpus = "Priority", 0, 4
    job.partition = "build"
    return job


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cluster. Without this the mounted screen's own resolve pass finds no such
    job, calls `_mark_started`, sets `_done` -- and every spinner tick returns at
    its first guard, so the whole file measured a screen that had stopped."""
    monkeypatch.setattr(tui, "resolve_pending_job", lambda *a, **k: _CURRENT["job"])
    # Real partitions, not `[]`: see `_parts`. `_poll` below stubs the same values,
    # so a poll that changes nothing really changes nothing.
    monkeypatch.setattr(tui, "resolve_cluster_partitions", lambda *a, **k: _parts())
    monkeypatch.setattr(tui, "resolve_queue_counts", lambda *a, **k: None)
    monkeypatch.setattr(tui, "resolve_priority_rank", lambda *a, **k: None)
    # The poll resolves the association table too now (D23), and "no cluster" has to
    # cover it or this file shells out to the real `sacctmgr`.
    monkeypatch.setattr(tui, "resolve_user_associations", lambda *a, **k: ASSOC)


#: The job the stubbed resolver hands back, set by whichever harness is mounted.
_CURRENT: dict[str, Any] = {"job": None}


class _Harness(App[None]):
    def __init__(self, job: Any) -> None:
        super().__init__()
        self._job_arg = job
        _CURRENT["job"] = job
        self.scr: PendingScreen | None = None

    async def on_mount(self) -> None:
        self.scr = PendingScreen(self._job_arg, SlurmwatchConfig())
        self.push_screen(self.scr)


class _Spy:
    """Every `Static.refresh` call: its regions and whether it forced layout."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Region, ...], bool]] = []

    @property
    def regions(self) -> list[Region]:
        return [r for regions, _ in self.calls for r in regions]

    @property
    def whole_widget(self) -> int:
        """Calls that named no region, i.e. "repaint all of me"."""
        return sum(1 for regions, _ in self.calls if not regions)

    @property
    def layout(self) -> int:
        return sum(1 for _, forced in self.calls if forced)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    counter = _Spy()
    original = Static.refresh

    def counting(self: Static, *regions: Region, layout: bool = False, **kw: Any) -> Any:
        if isinstance(self, PendingView):
            counter.calls.append((regions, bool(layout)))
        return original(self, *regions, layout=layout, **kw)

    monkeypatch.setattr(Static, "refresh", counting)
    return counter


async def _mounted(pilot: Any, app: _Harness) -> PendingView:
    """Mounted AND settled: the screen's first resolve pass must have FINISHED.

    `on_mount` fires `run_worker(self._refresh())`, and that pass publishes its
    results and refreshes the view whenever it lands -- inside whatever window a
    test here is counting refreshes in. Eight `pause()`es were enough while it made
    three executor calls; D23 gave it a fourth, and under full-suite load the pass
    finished mid-measurement instead. Both symptoms were in the same run: a
    whole-widget refresh this file never asked for, and an "identical" poll that had
    something new to publish. Waiting on `_refresh_in_flight` is what makes the
    counts belong to the ticks the test drove.
    """
    for _ in range(8):
        await pilot.pause()
    assert app.scr is not None
    for _ in range(80):
        if not app.scr._refresh_in_flight:
            break
        await pilot.pause()
    assert not app.scr._refresh_in_flight, "the first resolve pass never finished"
    for _ in range(2):
        await pilot.pause()
    return app.scr.query_one(PendingView)


def _freeze(app: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the live 0.12 s timer from ticking during a measurement.

    It keeps firing inside `pilot.pause()`, so a test that counts ticks or diffs
    two frames is otherwise measuring an unknown number of them -- an earlier
    draft read `frame == 8` where it had asked for 5.

    **Patching the method is not enough, and this helper used to do only that.**
    `on_mount` calls `set_interval(0.12, self._tick_spinner)`, which hands the
    `Timer` a BOUND METHOD; a bound method carries the function it was built from,
    so replacing the class attribute afterwards leaves the timer calling the
    original. Measured: with only the `monkeypatch.setattr` below, `view.frame`
    still went 3 -> 14 over a 0.5 s window. Pausing the `Timer` object is what
    actually stops it (`frame` then held at 0 over the same window). The
    monkeypatch stays, because the tests that call `_tick_spinner` BY HAND also
    want the timer's own path inert.
    """
    assert app.scr is not None
    for timer in list(app.scr._timers):
        # By callback name rather than by interval, so it survives a cadence
        # change; `_kick_refresh`'s 10 s timer and the app's own screen-update
        # timer must keep running -- pausing those was measured to stall the
        # repaint path this file drives.
        if getattr(timer._callback, "__name__", "") == "_tick_spinner":
            timer.pause()
    monkeypatch.setattr(type(app.scr), "_tick_spinner", lambda self: None)


class TestASpinnerTickRepaintsOnlyTheSpinner:
    async def test_a_tick_does_not_repaint_the_whole_panel(self, spy: _Spy) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            app.scr._tick_spinner()  # first tick: no previous paint to diff against
            await pilot.pause()
            spy.calls.clear()
            for _ in range(4):
                app.scr._tick_spinner()
                await pilot.pause()
            assert spy.whole_widget == 0, spy.calls
            assert spy.regions, "nothing was repainted at all"
            assert all(r.height == 1 for r in spy.regions), spy.regions
            assert view.size.height > 4, view.size

    async def test_the_tick_itself_goes_through_the_diff(self, spy: _Spy) -> None:
        # Otherwise only one assertion in this file notices if the call site is
        # reverted: `_painted` is written by the new path and by nothing else.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            view._painted = None
            app.scr._tick_spinner()
            assert view._painted, "the tick did not record what it painted"
            assert len(view._painted) == view.size.height

    async def test_the_repainted_line_is_the_one_that_changed(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            app.scr._repaint_spinner(view)  # seed `_painted`
            before = list(view._painted or [])
            spy.calls.clear()
            view.frame += 1
            app.scr._repaint_spinner(view)
            after = view._painted or []
            moved = [y for y, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
            assert moved, "the spinner did not move"
            assert [r.y for r in spy.regions] == moved, (moved, spy.regions)

    async def test_the_where_table_is_identical_across_ticks(self, spy: _Spy) -> None:
        """The reported symptom's premise: the verdict cells never change.

        The rows have to BE there for that to mean anything. This filtered
        `view._painted` for the verdict glyphs against a panel whose partitions
        were stubbed `[]` -- so the panel read `cluster partition info
        unavailable`, the filter returned `[]` on all five frames, and the test
        asserted `[] == []` five times. It passed with `_repaint_spinner`
        reverted, which is the same emptiness from the other direction: nothing
        writes `_painted` then either. `_parts()` supplies the table, and the
        emptiness check below is what refuses to pass on an absent one.
        """
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            frames = []
            for _ in range(5):
                app.scr._tick_spinner()
                await pilot.pause()
                frames.append([ln for ln in (view._painted or []) if "▸" in ln or "YES" in ln])
            assert frames[0], "the tick painted no verdict rows -- nothing is being compared"
            assert any("YES" in ln for ln in frames[0]), frames[0]
            for frame in frames[1:]:
                assert frame == frames[0], (frames[0], frame)

    async def test_a_first_tick_still_paints_everything_once(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing to diff against yet, so the whole panel is owed one repaint.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            view._painted = None
            spy.calls.clear()
            app.scr._repaint_spinner(view)
            assert spy.whole_widget == 1, spy.calls

    def test_a_missing_styles_cache_falls_back_to_a_full_refresh(self) -> None:
        """`_styles_cache` is a Textual private: its absence must not raise at 8 fps.

        Driven against a stand-in rather than a real widget, because Textual's own
        `_set_dirty` calls `self._styles_cache.clear()` -- taking the attribute off
        a live `PendingView` breaks Textual, not this guard, which is a different
        thing to test.
        """

        class _NoCache:
            _painted = None

            def __init__(self) -> None:
                self.refreshed: list[tuple[Any, ...]] = []
                self.size = type("S", (), {"width": 40, "height": 5})()

            def refresh(self, *regions: Any, **kw: Any) -> None:
                self.refreshed.append(regions)

            def render_lines(self, region: Any) -> list[Any]:  # pragma: no cover
                raise AssertionError("must not render without the cache to clear")

        stub = _NoCache()
        PendingScreen._repaint_spinner(stub)  # type: ignore[arg-type]
        assert stub.refreshed == [()], stub.refreshed

    def test_a_zero_sized_view_falls_back_too(self) -> None:
        # Before the first layout the widget has no size, so there are no lines to
        # diff and nothing to scope a region to.
        class _Unsized:
            _painted = None
            _styles_cache = type("C", (), {"clear": lambda self: None})()

            def __init__(self) -> None:
                self.refreshed: list[tuple[Any, ...]] = []
                self.size = type("S", (), {"width": 0, "height": 0})()

            def refresh(self, *regions: Any, **kw: Any) -> None:
                self.refreshed.append(regions)

        stub = _Unsized()
        PendingScreen._repaint_spinner(stub)  # type: ignore[arg-type]
        assert stub.refreshed == [()], stub.refreshed


class TestThePollDoesNotReLayOutAnUnchangedPanel:
    async def test_an_identical_poll_repaints_nothing(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `_freeze` for the reason its neighbours call it, and this one did not:
        # the live 0.12 s timer keeps ticking inside the measured window and the
        # spinner's own height-1 region lands in `spy.calls`, indistinguishable
        # from a repaint by the poll. Measured over 40 runs of this test alone:
        # **16 / 40 failed** with no `_freeze` call, **3 / 40** with the call but
        # the old `_freeze` (which patched the method and left the timer running),
        # and **0 / 40** with both. See `TestTheMeasurementWindowCanBeMadeQuiet`.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            view._markup = view.render()
            spy.calls.clear()
            await _poll(app, same=True)
            await pilot.pause()
            assert spy.calls == [], spy.calls

    async def test_a_changed_poll_still_repaints(self, spy: _Spy) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            view._markup = "something else entirely"
            spy.calls.clear()
            await _poll(app, same=True)
            await pilot.pause()
            assert spy.calls, "a changed panel must repaint"


class TestTheMeasurementWindowCanBeMadeQuiet:
    """Seven tests here rest on `_freeze`, and it did nothing to the timer.

    Not cosmetic: the test above counts every refresh in a window, and a live
    spinner tick lands in that count as a height-1 region indistinguishable from
    one the poll asked for -- 3 of 40 runs failed with the old helper in place.
    The helper meant to prevent that was patching a method the `Timer` does not
    read, so it silenced only the hand-driven calls. This class is what stops it
    regressing to that again, since a helper that quietly does nothing leaves no
    other trace.
    """

    async def test_freezing_stops_the_live_spinner_timer(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Real time, not `pilot.pause()`: the point is the wall-clock timer.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            _freeze(app, monkeypatch)
            start = view.frame
            await asyncio.sleep(0.4)  # >= 3 intervals at 0.12 s
            for _ in range(5):
                await pilot.pause()
            assert view.frame == start, f"the timer ticked {view.frame - start} times"

    async def test_an_unfrozen_screen_still_animates_by_itself(self, spy: _Spy) -> None:
        # The control: freezing is opt-in per test and does not outlive the app it
        # was applied to, so a screen nobody froze still animates on its own timer.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            start = view.frame
            await asyncio.sleep(0.4)
            for _ in range(5):
                await pilot.pause()
            assert view.frame > start, "the live 0.12 s timer never fired"


#: 25 node names in one `Reason` -- a real `ReqNodeNotAvail` from this cluster.
#: Long enough that the panel is 6 rows taller at 70 columns, with the SAME number
#: of newlines in the markup, which is exactly what the old gate could not see.
_WRAPPING_REASON = "ReqNodeNotAvail, UnavailableNodes:" + ",".join(
    f"midway3-{n:04d}" for n in range(1, 26)
)


class TestAChangedPanelIsAlwaysRelaidOut:
    """A wrapped panel that grew was clipped, because newlines are not height.

    `layout=` was gated on `markup.count("\n")`, and this panel wraps: a longer
    reason code re-flows into more ROWS without adding a single newline. The
    measurement is in the module docstring; these drive the real
    `PendingView.render` at a width where the reason wraps, and watch the panel's
    painted height rather than its newline count.
    """

    @staticmethod
    def _painted(view: PendingView) -> list[str]:
        return [strip.text for strip in view.render_lines(Region(0, 0, *view.size))]

    async def test_a_longer_wrapping_reason_grows_the_painted_panel(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 70 columns: narrow enough that the reason and the requeue tip wrap, wide
        # enough that the WHERE table still has its columns. 60 rows so the screen
        # can afford the taller panel -- what is under test is whether it is ASKED
        # for, not whether it fits.
        app = _Harness(_unplanned())
        async with app.run_test(size=(70, 60)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            before_markup, before_height = view.render(), view.size.height
            assert before_height == len(self._painted(view))

            app.scr._job.reason = _WRAPPING_REASON
            await _poll(app, same=True)
            for _ in range(3):
                await pilot.pause()

            after_markup, painted = view.render(), self._painted(view)
            # The premise: the markup got longer without getting more lines, so the
            # newline count the old gate read is IDENTICAL across this change.
            assert after_markup != before_markup
            assert after_markup.count("\n") == before_markup.count("\n"), (
                after_markup.count("\n"),
                before_markup.count("\n"),
            )
            # The panel is taller, and the widget was re-laid-out to that height.
            needed = view.get_content_height(view.size, view.size, view.size.width)
            assert needed > before_height, (needed, before_height)
            assert view.size.height == needed, (view.size.height, needed)
            assert len(painted) == needed

    async def test_the_verdict_table_is_not_clipped_off_the_bottom(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What the six lost rows WERE: the whole `Where It Could Run` table and the
        # requeue tip under it -- the panel's only actionable content.
        app = _Harness(_unplanned())
        async with app.run_test(size=(70, 60)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            app.scr._job.reason = _WRAPPING_REASON
            await _poll(app, same=True)
            for _ in range(3):
                await pilot.pause()
            painted = self._painted(view)
            for phrase in ("Where It Could Run", "YES", "broadwl"):
                assert any(phrase in line for line in painted), (phrase, painted[-4:])

    async def test_the_poll_asks_for_the_layout_pass(
        self, spy: _Spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The mechanism, at the call site: the refresh that publishes a changed
        # panel carries `layout=True`, whatever the newline count did.
        app = _Harness(_unplanned())
        async with app.run_test(size=(70, 60)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            _freeze(app, monkeypatch)
            before = view.render()
            app.scr._job.reason = _WRAPPING_REASON
            spy.calls.clear()
            await _poll(app, same=True)
            await pilot.pause()
            assert view.render().count("\n") == before.count("\n")
            assert spy.calls, "a changed panel must repaint"
            # The publishing refresh: whole widget, layout forced. (A second
            # `((), True)` follows it here, from Textual's own `virtual_size`
            # watcher reacting to the resize the layout pass produced -- a
            # consequence of the pass happening, so it is not asserted on.)
            assert spy.calls[0] == ((), True), spy.calls


async def _poll(app: _Harness, same: bool) -> None:
    """Run one `_refresh_once` with every resolver stubbed."""
    from unittest import mock

    assert app.scr is not None
    job = app.scr._job
    with (
        mock.patch.object(tui, "resolve_pending_job", lambda *a, **k: job),
        # The same values the mounted screen already resolved (`_offline`), so
        # `same=True` really does mean "this poll found nothing new". Stubbing
        # `[]` here against a mounted panel that HAS partitions would have made
        # every poll a change, which is the opposite of what these two measure.
        mock.patch.object(tui, "resolve_cluster_partitions", lambda *a, **k: _parts()),
        mock.patch.object(tui, "resolve_queue_counts", lambda *a, **k: None),
        mock.patch.object(tui, "resolve_priority_rank", lambda *a, **k: None),
        mock.patch.object(tui, "resolve_user_associations", lambda *a, **k: ASSOC),
    ):
        await app.scr._refresh_once()


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    async def test_the_spinner_still_advances_every_tick(self, spy: _Spy) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            start = view.frame
            for _ in range(5):
                app.scr._tick_spinner()
            # `>=`, not `==`: the live 0.12 s timer is ticking too.
            assert view.frame >= start + 5

    async def test_a_planned_job_still_animates_nothing(self, spy: _Spy) -> None:
        job = _unplanned()
        job.start_time_estimate = time.time() + 3600
        app = _Harness(job)
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            spy.calls.clear()
            before = view.frame
            for _ in range(5):
                app.scr._tick_spinner()
                await pilot.pause()
            assert view.frame == before
            assert spy.calls == [], spy.calls

    async def test_the_panel_still_says_the_same_things(self, spy: _Spy) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            markup = view.render()
            for phrase in ("Why It", "estimated start", "Where It Could Run"):
                assert phrase in markup, (phrase, markup[:400])

    async def test_a_torn_down_screen_does_not_raise(self, spy: _Spy) -> None:
        # `_tick_spinner` is wrapped in `suppress(NoMatches)` for this reason.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            await _mounted(pilot, app)
            assert app.scr is not None
            app.scr.query_one(PendingView).remove()
            await pilot.pause()
            app.scr._tick_spinner()  # must not raise
            await pilot.pause()

    async def test_the_view_is_still_one_static(self, spy: _Spy) -> None:
        # The premise of the fix: one widget renders the whole panel.
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert isinstance(view, Static)
            assert len(app.scr.query(PendingView)) == 1 if app.scr else False
