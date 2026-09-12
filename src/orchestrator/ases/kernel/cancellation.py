"""Cooperative cancellation.

There is no way to forcibly kill an in-flight LLM call or tool invocation
without leaving the sandbox in an unknown state. Cancellation here is
cooperative: a `CancelToken` is threaded through the scheduler, and it is
checked *between* steps, never used to interrupt one. A node that is already
running is allowed to finish that one step; the scheduler simply does not
start any new work once cancellation is requested.

This is what safe-stop is built from: request cancellation, let the current
step land, emit `RUN_HALTED`, and stop. No compensator runs on a cancelled
node's own work-in-progress - only on effects that already completed and were
classified `external` (that is `kernel.recovery`'s job, not this module's).
"""

from __future__ import annotations

import asyncio


class CancelToken:
    """A one-way flag. Once set, it never resets - a run does not un-cancel."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    def cancel(self, reason: str) -> None:
        if self._event.is_set():
            return  # first reason wins; do not overwrite it
        self._reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancellationRequestedError(self._reason or "cancelled")

    async def wait(self) -> str:
        """Block until cancelled; returns the reason. Used by long-lived tasks."""
        await self._event.wait()
        assert self._reason is not None
        return self._reason


class CancellationRequestedError(RuntimeError):
    """Raised by `raise_if_cancelled`. Caught only at the scheduler's step boundary."""
