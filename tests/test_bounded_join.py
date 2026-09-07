"""`join_bounded` is the half of `aio` nothing tested directly.

Its sibling in the same 14-statement module has two tests of its own
(`test_cli.py::test_the_reap_does_not_eat_the_callers_cancellation` and
`test_it_still_reaps_the_future_it_cancelled`); `join_bounded` had none, and
four call sites — two in `collector.teardown`, one in `collector`'s sample path,
one in `tui`'s poll-task join. Found by mining partial-branch coverage: line 42,
the `raise exc`, was the one uncovered statement in the module.

That line is a stated rule: *"A failure the awaited work reported is still
surfaced, because teardown should not quietly eat a real error."* And the
function's central claim is about the primitive it chose:

    ``asyncio.wait`` is the right primitive and ``wait_for`` is not: wait_for
    CANCELS its awaitable on timeout and then waits for that cancellation to
    land, so a task which does not honour cancellation makes
    ``wait_for(task, T)`` wait forever — a bound in name only.

Both are pinned here, the second with a task that genuinely ignores
cancellation, because that is the only shape that tells the two primitives
apart.

Every test carries its own tight timeout: the failure mode under test IS a hang,
and a test that hangs reports nothing until the suite-wide 120s cap.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from slurmwatch import aio


def _uncooperative(release: asyncio.Event) -> asyncio.Task[None]:
    """A task that swallows cancellation until `release` is set.

    Same shape as `test_cli.py`'s helper for `reap_cancelled`, kept local so
    this file stands alone.
    """

    async def _body() -> None:
        while True:
            try:
                await asyncio.wait_for(release.wait(), timeout=5.0)
                return
            except (asyncio.CancelledError, asyncio.TimeoutError):
                continue

    return asyncio.create_task(_body())


class TestAReportedFailureIsSurfaced:
    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_the_exception_reaches_the_caller(self) -> None:
        """Line 42. Teardown must not quietly eat a real error."""
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("the collector blew up"))
        with pytest.raises(RuntimeError, match="the collector blew up"):
            await aio.join_bounded(fut, timeout=1.0)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_a_task_that_raised_is_surfaced_too(self) -> None:
        """The shape the call sites actually pass: a Task, not a bare Future."""

        async def _boom() -> None:
            raise ValueError("sample failed")

        task = asyncio.create_task(_boom())
        with pytest.raises(ValueError, match="sample failed"):
            await aio.join_bounded(task, timeout=1.0)


class TestTheBoundIsRealRatherThanNominal:
    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_a_task_ignoring_cancellation_does_not_hang_the_join(self) -> None:
        """The docstring's central claim, and the only shape that shows it.

        `wait_for(task, T)` would cancel and then wait for a cancellation this
        task refuses to honour, so it would not return until `release` is set.
        `asyncio.wait` abandons the join instead.
        """
        release = asyncio.Event()
        target = _uncooperative(release)
        await asyncio.sleep(0.05)  # let it reach its suspension point
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            await aio.join_bounded(target, timeout=0.2)
            elapsed = loop.time() - started
            assert elapsed < 3.0, f"the join was not bounded: {elapsed:.2f}s"
            assert not target.done(), "the join should abandon, not await"
        finally:
            release.set()
            target.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(target, timeout=2.0)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_it_does_not_cancel_what_it_gave_up_on(self) -> None:
        """`wait_for` cancels on timeout; `wait` does not. The call sites cancel
        deliberately elsewhere, so this one must not do it for them."""
        release = asyncio.Event()
        target = _uncooperative(release)
        await asyncio.sleep(0.05)
        try:
            await aio.join_bounded(target, timeout=0.2)
            assert not target.cancelled(), "the join cancelled the task it abandoned"
        finally:
            release.set()
            target.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(target, timeout=2.0)


class TestControls:
    """Each passes with the `raise exc` removed as well as with it -- they cover
    the outcomes that are NOT a reported failure. Verified by running that
    neuter."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_a_clean_result_returns_quietly(self) -> None:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        fut.set_result("done")
        await aio.join_bounded(fut, timeout=1.0)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_a_cancelled_future_is_not_a_failure(self) -> None:
        """The `if not finished.cancelled()` guard: a cancellation is how these
        tasks are asked to stop, so it must not surface as an error and must not
        reach `.exception()`, which would raise `CancelledError` itself.

        Belongs to both halves, which running the neuters established. It is a
        CONTROL for the `raise exc` removal -- a cancellation was never a
        reported failure -- and a FINDING test for the primitive: swap
        `asyncio.wait` for `wait_for` and awaiting an already-cancelled future
        raises `CancelledError` at the caller, so it reddens with the two bound
        tests above.
        """
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        fut.cancel()
        await asyncio.wait({fut})
        await aio.join_bounded(fut, timeout=1.0)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_a_task_that_finishes_within_the_bound_is_awaited(self) -> None:
        """The ordinary teardown case: the work stops promptly and is joined."""

        async def _quick() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(_quick())
        await aio.join_bounded(task, timeout=2.0)
        assert task.done() and not task.cancelled()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_the_sibling_still_reaps(self) -> None:
        """`reap_cancelled` shares the module and is not what this round touches."""
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await aio.reap_cancelled(fut)
        assert fut.cancelled()
