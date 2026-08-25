"""Asyncio joins that cannot hang, in one place.

Cancelling a task is a request, not a guarantee: `cancel()` raises at the task's
next suspension point, and everything it does on the way out — a `finally` that
awaits, a caught CancelledError, a submission queued behind a saturated executor —
happens on the canceller's clock. Both idioms here exist because the obvious
spellings get that wrong in opposite directions, and both were live bugs: a `--log`
run that swallowed its own cancellation and became unkillable, and teardown joins
that could wait forever on a task which refused to die.

They live together, and away from any one caller, for the reason the shared display
rules do: three copies of a subtle rule is three places to get it wrong.
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["join_bounded", "reap_cancelled"]


async def join_bounded(fut: asyncio.Future[Any], timeout: float) -> None:
    """Wait up to ``timeout`` for ``fut``, then give up on it.

    ``asyncio.wait`` is the right primitive and ``wait_for`` is not: wait_for CANCELS
    its awaitable on timeout and then waits for that cancellation to land, so a task
    which does not honour cancellation makes ``wait_for(task, T)`` wait forever — a
    bound in name only. ``asyncio.wait`` abandons the join instead. It also never
    raises the awaited future's exception, so unlike
    ``suppress(CancelledError): await task`` it cannot swallow a cancellation aimed
    at *this* task.

    A failure the awaited work reported is still surfaced, because teardown should
    not quietly eat a real error.
    """
    done, _ = await asyncio.wait({fut}, timeout=timeout)
    for finished in done:
        if not finished.cancelled():
            exc = finished.exception()
            if exc is not None:
                raise exc


async def reap_cancelled(fut: asyncio.Future[Any]) -> None:
    """Cancel ``fut`` and wait for it, without swallowing OUR OWN cancellation.

    The obvious spelling is wrong::

        fut.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fut

    because that ``suppress`` cannot tell ``fut``'s cancellation from the enclosing
    task's. A cancel landing inside that window — a Ctrl-C, a SIGTERM, a caller
    cancelling the task — is caught, discarded, and the loop carries on: not slow,
    but UNKILLABLE, the failure the poll loops were fixed for. Once per sample is a
    narrow window, which is why it read as one hung CI job in twenty and never
    reproduced locally.
    """
    fut.cancel()
    await asyncio.wait({fut})
