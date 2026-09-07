"""A remote that connects and then stops answering.

``_read_remote`` had three ways a stream could fail and resolved only two of
them. EOF reads the step's stderr and diagnoses it (retiring the node when the
failure is permanent); an unparseable line is counted toward
``_STREAM_MAX_PARSE_FAILS`` and retires the node at five. **Silence had
neither.** A stream that connected and then blocked -- a hung GPFS stat on the
far side is the realistic cause -- produced a 0.5s ``readline`` timeout every
tick and a bare ``return None``, forever: ``read_stream_error`` is only reached
at EOF, so no diagnosis was ever produced, and nothing tore the stream down or
relaunched it. That is the same latch ``N5`` exists to end, reached through the
one path that never gets as far as a line to count.

Two things made it survivable rather than silent, and neither ends it: the stuck
watchdog flips the banner amber at ``_SWITCH_STUCK_S``, and once one frame has
arrived the header shows ``<N>s old`` and grows. Before the first frame there is
nothing to age.

Timed rather than counted, because the readline timeout is *shorter* than a frame
interval: a healthy stream trips that branch routinely, so a counter would retire
it. No test here waits on the clock -- the timestamp is pushed back, the way
``test_tui.py`` ages ``_switch_started`` past the watchdog.
"""

import asyncio
import time
from typing import Any

import pytest

from slurmwatch import tui as tuimod
from slurmwatch.model import JobContext
from tests.test_remote import _snapshot
from tests.test_tui import _DashApp, _StubCollector


class _SilentStream:
    """Connected, and never answers again."""

    def __init__(self) -> None:
        self.stdout = self
        self.returncode: int | None = None
        self.killed = False
        self.reads = 0

    async def readline(self) -> bytes:
        self.reads += 1
        raise asyncio.TimeoutError

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class _TalkingStream(_SilentStream):
    """Delivers real frames, after an optional run of silence."""

    def __init__(self, silent_reads: int = 0) -> None:
        super().__init__()
        self._silent = silent_reads
        self._frame = _snapshot().to_json().encode() + b"\n"

    async def readline(self) -> bytes:
        self.reads += 1
        if self.reads <= self._silent:
            raise asyncio.TimeoutError
        return self._frame


def _app(node: str = "cn002") -> _DashApp:
    job = JobContext(
        job_id="12345",
        username="ada",
        partition="gpu",
        nodelist="cn[001-002]",
        hostname="cn001",
        cpus_allocated=8,
        mem_limit_bytes=8 * 1024**3,
        gpu_count_requested=0,
        gpu_indices=[],
        step_id="0",
        uid=1001,
        nodelist_resolved=["cn001", node],
    )
    return _DashApp(_StubCollector(), job)


def _attach(scr: Any, node: str, proc: Any) -> None:
    """Put a running stream in place so `_read_remote` skips the launch path."""
    scr._stream_node = node
    scr._stream_proc = proc


def _deadline(scr: Any) -> float:
    cadence = max(scr.config.poll_interval, 1.0)
    return tuimod._stream_silence_deadline(cadence)


@pytest.mark.asyncio
async def test_a_silent_stream_is_retired_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the deadline: torn down, backed off, and the banner says what happened."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scr = app.scr
        backoffs: list[str] = []

        async def _record(node: str) -> None:
            backoffs.append(node)

        monkeypatch.setattr(scr, "_stream_backoff", _record)
        fake = _SilentStream()
        proc: Any = fake
        _attach(scr, "cn002", proc)
        scr._stream_silent_since = time.monotonic() - (_deadline(scr) + 1.0)

        assert await scr._read_remote("cn002") is None
        assert scr._stream_proc is None, "the stream must be torn down"
        assert fake.killed is True
        assert backoffs == ["cn002"], "and handed to the existing backoff/relaunch path"
        assert "not answering" in scr._stream_error, scr._stream_error
        assert "no telemetry" in scr._stream_error, scr._stream_error


@pytest.mark.asyncio
async def test_the_first_silence_only_starts_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One timeout is normal. It records when the quiet began and nothing else."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scr = app.scr
        backoffs: list[str] = []
        monkeypatch.setattr(scr, "_stream_backoff", lambda n: backoffs.append(n))
        fake = _SilentStream()
        proc: Any = fake
        _attach(scr, "cn002", proc)
        assert scr._stream_silent_since is None

        assert await scr._read_remote("cn002") is None
        assert scr._stream_silent_since is not None, "the clock must start"
        assert scr._stream_proc is proc, "and the stream must survive one timeout"
        assert backoffs == []


@pytest.mark.asyncio
async def test_control_a_stream_delivering_frames_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONTROL, passing with the deadline present or absent.

    The normal case: a frame arrives, a snapshot comes back, and no silence is
    ever recorded. A fix that tore down on any timeout would fail this.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scr = app.scr
        backoffs: list[str] = []
        monkeypatch.setattr(scr, "_stream_backoff", lambda n: backoffs.append(n))
        fake = _TalkingStream()
        proc: Any = fake
        _attach(scr, "cn002", proc)

        assert await scr._read_remote("cn002") is not None
        assert scr._stream_proc is proc
        assert scr._stream_silent_since is None
        assert backoffs == []
        assert fake.killed is False


@pytest.mark.asyncio
async def test_control_a_slow_but_alive_stream_is_not_retired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONTROL, passing in both states. Three timeouts, then a frame.

    Well inside the deadline, so the stream is alive and merely slow -- the case
    that must not be killed, and the reason the deadline is timed instead of
    counted. The frame then clears the clock, so a later quiet spell starts over.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scr = app.scr
        backoffs: list[str] = []
        monkeypatch.setattr(scr, "_stream_backoff", lambda n: backoffs.append(n))
        fake = _TalkingStream(silent_reads=3)
        proc: Any = fake
        _attach(scr, "cn002", proc)

        for _ in range(3):
            assert await scr._read_remote("cn002") is None
        assert scr._stream_proc is proc, "still alive"
        assert backoffs == []

        assert await scr._read_remote("cn002") is not None
        assert scr._stream_silent_since is None, "a real frame resets the quiet clock"
        assert fake.killed is False


def test_the_deadline_is_derived_not_picked() -> None:
    """It outlasts the watchdog it follows, and scales with the frame cadence.

    Ordering matters: the amber "still reaching / retrying" must appear FIRST and
    be resolved by this, rather than a teardown contradicting a banner that is
    still claiming progress. And `config.MAX_INTERVAL` is an hour -- a stream
    emitting every 60s must not be retired at 33s.
    """
    from slurmwatch.config import MAX_INTERVAL

    assert tuimod._STREAM_SILENCE_TIMEOUT > tuimod._SWITCH_STUCK_S
    # the floor applies at ordinary cadences
    assert tuimod._stream_silence_deadline(1.0) == tuimod._STREAM_SILENCE_TIMEOUT
    # and the cadence takes over once five frames outlast the floor
    assert tuimod._stream_silence_deadline(60.0) == 60.0 * tuimod._STREAM_MAX_PARSE_FAILS
    assert tuimod._stream_silence_deadline(MAX_INTERVAL) > MAX_INTERVAL
