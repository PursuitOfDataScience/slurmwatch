"""D23: the pending view ran `sacctmgr` from inside `render()`.

``PendingView._where`` built its requeue tip from ``resolve_user_associations(
job.username or "")`` -- called on every render. The panel is one `Static` and
`PendingScreen` animates the "calculating…" spinner at ``set_interval(0.12, ...)``,
so `render()` runs about eight times a second for as long as the scheduler has no
start estimate. Measured on a 110x40 headless screen with a real timer and a real
one-second window:

    before:  8 spinner ticks -> 8 render() calls -> 8 `sacctmgr` spawns  (8.0/s)
    after:   8 spinner ticks -> 8 render() calls -> 0 `sacctmgr` spawns  (0.0/s)

Eight spawns a second only happens when the lookup FAILS, and that is deliberate:
`pending.resolve_user_associations` caches a success for the life of the process but
returns ``None`` without caching on error, so one bad `sacctmgr` cannot latch
"unknown" over a whole session. Correct policy, wrong place to pay for it -- the
retry was landing once per frame, on the event loop, with each attempt allowed
``SLURM_CMD_TIMEOUT`` (15 s). With the same window and a `sacctmgr` that takes
200 ms to fail:

    before:  1.00 s of a 1.14 s window spent blocked inside render(), 5 ticks
    after:   0.00 s blocked, 8 ticks -- the spinner keeps full rate

The fix moves the lookup, it does not cache the failure: ``PendingView.assoc`` is
now an input the 10 s poll fills in, like ``partitions`` and ``queue_rank``, and
``PendingScreen._refresh_once`` resolves it in the executor alongside its three
siblings. So the worst-case stale window is ONE POLL CYCLE (10 s, or sooner on `r`)
-- the same freshness every other figure on this panel already has -- and a
`sacctmgr` that starts working is reflected on the next poll rather than being
remembered as broken. `pending.py`'s no-cache-on-failure rule is untouched, and
``TestControls`` pins that.

`TestControls` also pins what must not move: the tip still names the QOS from the
table, an unknown table still softens the command, a partition the user cannot
submit to is still dropped, the loading state still needs no table at all, and the
poll is still every 10 s. Every control passes with the fix in OR out.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from rich.text import Text
from textual.app import App
from textual.screen import Screen

from slurmwatch import pending as pmod
from slurmwatch import tui
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import SlurmCommandError
from slurmwatch.pending import PartitionResources
from slurmwatch.slurm import _is_mock
from slurmwatch.tui import PendingScreen, PendingView

#: broadwl fits, is permitted, and names a QOS after itself -- so a complete
#: `scontrol update ... QOS=broadwl` is offered and its absence is visible.
ASSOC = {"build": ["build"], "broadwl": ["broadwl"]}
#: Same shape, minus any right to broadwl.
ASSOC_WITHOUT_BROADWL = {"build": ["build"]}
SACCTMGR_TABLE = "build|build\nbroadwl|broadwl\n"


def _parts() -> list[PartitionResources]:
    """The current partition (full) plus one alternative with room."""
    return [
        PartitionResources(
            "build", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
        ),
        PartitionResources("broadwl", True, idle_nodes=53, cpus_idle=1101, max_node_cpus=48),
    ]


def _unplanned() -> Any:
    """A pending job with no start estimate -- the one state the spinner runs in."""
    job = pmod._mock_pending_job("57902634")
    job.start_time_estimate = None
    job.reason, job.req_gpus, job.req_cpus = "Priority", 0, 4
    job.partition = "build"
    return job


class _Sacctmgr:
    """Counts, and optionally slows, every Slurm subprocess `pending` would launch."""

    def __init__(self, *, table: str | None = None, delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.table = table
        self.delay = delay

    @property
    def spawns(self) -> int:
        return sum(1 for cmd in self.calls if cmd and cmd[0] == "sacctmgr")

    def __call__(self, cmd: list[str], timeout: int = 15) -> str:
        self.calls.append(list(cmd))
        if self.delay:
            time.sleep(self.delay)  # stands in for a slow / timing-out sacctmgr
        if self.table is None:
            raise SlurmCommandError("sacctmgr: Slurmdbd connection failed")
        return self.table


@pytest.fixture
def sacctmgr(monkeypatch: pytest.MonkeyPatch) -> _Sacctmgr:
    """A FAILING sacctmgr by default -- the case pending.py refuses to cache."""
    stub = _Sacctmgr()
    monkeypatch.setattr(pmod, "_run_slurm_cmd", stub)
    assert not _is_mock(), "mock mode would short-circuit resolve_user_associations"
    return stub


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cluster, but a job that stays pending and partitions that reach the tip.

    Without the `resolve_pending_job` stub the mounted screen's own pass finds no
    such job, calls `_mark_started`, and every spinner tick returns at its first
    guard -- the whole file would then measure a screen that had stopped.
    `resolve_user_associations` is deliberately NOT stubbed: it is what is being
    counted.
    """
    monkeypatch.setattr(tui, "resolve_pending_job", lambda *a, **k: _CURRENT["job"])
    monkeypatch.setattr(tui, "resolve_cluster_partitions", lambda *a, **k: _parts())
    monkeypatch.setattr(tui, "resolve_queue_counts", lambda *a, **k: None)
    monkeypatch.setattr(tui, "resolve_priority_rank", lambda *a, **k: None)


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


async def _mounted(pilot: Any, app: _Harness) -> PendingView:
    """Mount, then wait for the first poll so the WHERE table has partitions."""
    assert app.scr is not None
    for _ in range(40):
        await pilot.pause()
        view = app.scr.query_one(PendingView)
        if view.partitions:
            break
    view = app.scr.query_one(PendingView)
    # Guard against a green run that never reached the code under test.
    assert view.partitions, "partitions never landed -- the tip branch is unreachable"
    assert "broadwl" in view.render(), "the WHERE table is not being rendered"
    return view


def _card(**attrs: Any) -> str:
    """An unmounted `PendingView`, markup stripped -- as every other test renders it."""
    view = PendingView()
    view.job = _unplanned()
    view.config = SlurmwatchConfig()
    view.resolved = True
    view.partitions = _parts()
    for key, value in attrs.items():
        setattr(view, key, value)
    return Text.from_markup(view.render()).plain


class TestTheRenderPathSpawnsNoSubprocess:
    async def test_the_spinner_turns_without_running_sacctmgr(
        self, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        """The headline measurement, on the real 0.12 s timer and a real window."""
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            sacctmgr.calls.clear()
            frame0 = view.frame
            await asyncio.sleep(0.5)  # the 10 s poll cannot fire inside this window
            ticks = view.frame - frame0
        assert ticks >= 2, f"the spinner did not turn ({ticks} ticks)"
        assert sacctmgr.spawns == 0, (ticks, sacctmgr.calls)

    async def test_a_failing_sacctmgr_does_not_stall_the_repaint(
        self, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        """Five ticks against a `sacctmgr` that takes 200 ms to fail.

        The tick count is fixed, but the elapsed time is NOT a reading of the
        subprocess alone: it also contains whatever those five repaints cost,
        which is real work and is not free on a loaded machine. Measured failing
        at `5 ticks blocked for 0.24 s` against a 0.2 s bound while
        `sacctmgr.spawns == 0` on the same run -- i.e. the property held and the
        stopwatch still lost, immediately after an 11-minute coverage run on the
        same host.

        So the bound is set where it separates what it claims to. Before the fix
        every tick paid the 200 ms failure: 5 x 200 ms = 1.0 s of blocked event
        loop. Four delays (0.8 s) still catches that with room to spare, and the
        SHARP statement of the property is the assertion below it --
        `spawns == 0` catches even a single call, and
        `test_render_never_calls_the_resolver_at_all` booby-traps the resolver so
        any call at all raises. A stopwatch tightened to one delay was the
        weakest of the three layers pretending to be the strongest.
        """
        sacctmgr.delay = 0.2
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            await _mounted(pilot, app)
            assert app.scr is not None
            sacctmgr.calls.clear()
            started = time.perf_counter()
            for _ in range(5):
                app.scr._tick_spinner()
            elapsed = time.perf_counter() - started
        assert sacctmgr.spawns == 0, sacctmgr.calls
        assert elapsed < 4 * sacctmgr.delay, f"5 ticks blocked for {elapsed:.2f} s"

    def test_render_never_calls_the_resolver_at_all(
        self, monkeypatch: pytest.MonkeyPatch, sacctmgr: _Sacctmgr
    ) -> None:
        """Sharper than counting: the resolver is booby-trapped, so any call fails.

        `_mounted` is not needed here -- an unmounted view renders the same string,
        which is why the rest of the suite tests it that way.
        """

        def _explode(*_a: object, **_k: object) -> None:
            raise AssertionError("render() must not resolve associations (D23)")

        monkeypatch.setattr(tui, "resolve_user_associations", _explode)
        out = _card(assoc=ASSOC)
        assert "Partition=broadwl QOS=broadwl" in out, out
        assert sacctmgr.spawns == 0, sacctmgr.calls


class TestThePollResolvesItInstead:
    async def test_one_poll_resolves_it_once_and_hands_it_to_the_view(
        self, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        sacctmgr.table = SACCTMGR_TABLE
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            assert view.assoc == ASSOC, view.assoc
            assert sacctmgr.spawns == 1, sacctmgr.calls
            assert "Partition=broadwl QOS=broadwl" in Text.from_markup(view.render()).plain
            # A second poll is free: a SUCCESS is cached per process on purpose
            # (account configuration does not change while a dashboard is open).
            await app.scr._refresh_once()
            assert sacctmgr.spawns == 1, sacctmgr.calls

    async def test_a_failure_is_retried_next_poll_not_remembered_for_ever(
        self, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        """The visibility half: moving the call must not turn into caching it."""
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            assert view.assoc is None, "a failed lookup must read as UNKNOWN"
            failed_after = sacctmgr.spawns
            assert failed_after >= 1, "the poll never asked"
            assert not pmod._ASSOC_QOS_CACHE, "a failure must not be cached (pending.py policy)"
            sacctmgr.table = SACCTMGR_TABLE  # the site's slurmdbd comes back
            await app.scr._refresh_once()
            assert view.assoc == ASSOC, view.assoc
            assert sacctmgr.spawns == failed_after + 1, sacctmgr.calls
            assert "Partition=broadwl QOS=broadwl" in Text.from_markup(view.render()).plain

    async def test_a_transient_sibling_failure_carries_the_last_table_forward(
        self, sacctmgr: _Sacctmgr, offline: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gather's all-or-nothing fallback keeps the partitions; same for this."""
        sacctmgr.table = SACCTMGR_TABLE
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            assert app.scr is not None
            assert view.assoc == ASSOC

            def _boom(*_a: object, **_k: object) -> None:
                raise SlurmCommandError("squeue timed out")

            monkeypatch.setattr(tui, "resolve_queue_counts", _boom)
            await app.scr._refresh_once()
            assert view.assoc == ASSOC, "a squeue failure must not blank the assoc table"
            assert view.partitions, "nor the partitions"


class TestControls:
    """Behaviour that must not change. Every one passes with the fix in OR out.

    The rendering controls wire the table BOTH ways -- the attribute the poll now
    fills in and the module-level resolver `render()` used to call -- so they are a
    statement about the output, not about where the table came from.
    """

    def test_the_tip_still_names_the_qos_from_the_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tui, "resolve_user_associations", lambda *a, **k: ASSOC)
        out = _card(assoc=ASSOC)
        assert "broadwl has room for this request right now" in out, out
        assert "Partition=broadwl QOS=broadwl" in out, out

    def test_an_unknown_table_still_softens_the_command(
        self, monkeypatch: pytest.MonkeyPatch, sacctmgr: _Sacctmgr
    ) -> None:
        """UNKNOWN withholds nothing and promises nothing: the partition is still
        offered, the QOS is not guessed, and the hedge says why."""
        monkeypatch.setattr(tui, "resolve_user_associations", lambda *a, **k: None)
        out = _card(assoc=None)
        assert "scontrol update JobId=57902634 Partition=broadwl" in out, out
        assert "QOS=" not in out, out
        assert "the QOS moves with the job, not the partition" in out, out

    def test_a_partition_the_user_cannot_submit_to_is_still_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tui, "resolve_user_associations", lambda *a, **k: ASSOC_WITHOUT_BROADWL)
        out = _card(assoc=ASSOC_WITHOUT_BROADWL)
        assert "broadwl has room" not in out, out
        assert "no partition currently has enough free capacity" in out, out

    def test_the_loading_state_needs_no_table_at_all(self, sacctmgr: _Sacctmgr) -> None:
        """Before the first poll there are no partitions, so `_where` returns above
        the tip -- which is why an unresolved `assoc` costs the first frame nothing."""
        view = PendingView()
        view.job = _unplanned()
        view.config = SlurmwatchConfig()
        out = Text.from_markup(view.render()).plain
        assert "querying partitions" in out, out
        assert sacctmgr.spawns == 0, sacctmgr.calls

    def test_pending_py_still_refuses_to_cache_a_failure(self, sacctmgr: _Sacctmgr) -> None:
        """The policy this fix deliberately did NOT change."""
        assert pmod.resolve_user_associations("someone") is None
        assert not pmod._ASSOC_QOS_CACHE, pmod._ASSOC_QOS_CACHE
        assert pmod.resolve_user_associations("someone") is None
        assert sacctmgr.spawns == 2, "a failure must be retried, not remembered"
        sacctmgr.table = SACCTMGR_TABLE
        assert pmod.resolve_user_associations("someone") == ASSOC
        assert pmod.resolve_user_associations("someone") == ASSOC
        assert sacctmgr.spawns == 3, "a success IS cached, once per process"

    async def test_the_poll_is_still_every_ten_seconds(
        self, monkeypatch: pytest.MonkeyPatch, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        """The staleness window, pinned: a stale table cannot outlive one cycle."""
        seen: list[tuple[float, str]] = []
        original = Screen.set_interval

        def spy(self: Screen[Any], interval: float, callback: Any = None, **kw: Any) -> Any:
            if isinstance(self, PendingScreen):
                seen.append((interval, getattr(callback, "__name__", "")))
            return original(self, interval, callback, **kw)

        monkeypatch.setattr(Screen, "set_interval", spy)
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            await _mounted(pilot, app)
        assert (10.0, "_kick_refresh") in seen, seen

    async def test_the_panel_still_says_the_same_things(
        self, sacctmgr: _Sacctmgr, offline: None
    ) -> None:
        app = _Harness(_unplanned())
        async with app.run_test(size=(110, 40)) as pilot:
            view = await _mounted(pilot, app)
            markup = view.render()
        for phrase in ("Why It", "estimated start", "Where It Could Run", "broadwl"):
            assert phrase in markup, (phrase, markup[:400])
