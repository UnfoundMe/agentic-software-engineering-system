"""CancelToken: cooperative, one-way, and inspectable."""

from __future__ import annotations

import asyncio

import pytest

from ases.kernel.cancellation import CancellationRequestedError, CancelToken


def test_starts_uncancelled() -> None:
    token = CancelToken()
    assert not token.is_cancelled
    assert token.reason is None
    token.raise_if_cancelled()  # must not raise


def test_cancel_sets_the_reason() -> None:
    token = CancelToken()
    token.cancel("operator requested stop")
    assert token.is_cancelled
    assert token.reason == "operator requested stop"


def test_raise_if_cancelled_raises_with_the_reason() -> None:
    token = CancelToken()
    token.cancel("budget exhausted")
    with pytest.raises(CancellationRequestedError, match="budget exhausted"):
        token.raise_if_cancelled()


def test_first_reason_wins() -> None:
    """A cancellation is a fact about the run; it should not be rewritten by
    whichever caller happens to notice second."""
    token = CancelToken()
    token.cancel("first")
    token.cancel("second")
    assert token.reason == "first"


async def test_wait_unblocks_on_cancel() -> None:
    token = CancelToken()

    async def canceller() -> None:
        await asyncio.sleep(0.01)
        token.cancel("done")

    waiter = asyncio.create_task(canceller())
    reason = await token.wait()
    await waiter
    assert reason == "done"
