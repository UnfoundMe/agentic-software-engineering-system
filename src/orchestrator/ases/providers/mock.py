"""Synthetic provider for unit tests (`ASES_LLM_MODE=mock`).

Unlike `CassetteProvider`, which replays what a real model actually said,
`MockProvider` never had a real completion to record - it is a test double
the caller programs with canned responses. It refuses to *invent* a
schema-conformant value on the caller's behalf: guessing field values for an
arbitrary Pydantic model would silently pass tests against data nobody
decided was realistic. A test that needs structured output registers one
explicitly via `MockProvider.respond_with` or `MockProvider.respond_to`.
"""

from __future__ import annotations

from collections import deque

from ases.providers.base import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    ProviderError,
)


class NoScriptedResponseError(ProviderError):
    """`MockProvider.complete` was called with nothing queued for it."""


class MockProvider:
    """A queue of canned `CompletionResult`s, or a default deterministic echo.

    `respond_with` queues one result per call, consumed in order regardless
    of which prompt asked for it - useful for exercising a repair sequence
    (first response fails schema validation, second succeeds) where exactly
    one caller drives every call.

    `respond_to(prompt_version, result)` is the alternative for a *shared*
    provider instance driving several different agents at once (the realistic
    production shape - one `LLMProvider`, routed by `ModelRouter`, not one per
    agent): responses keyed by `prompt_version` are matched regardless of call
    order, which a plain FIFO queue cannot guarantee under `asyncio.gather`
    concurrency (e.g. two parallel implementation tasks sharing one provider).
    Checked before the generic queue, and does not consume from it.

    With nothing scripted for a request and no `output_schema` requested, it
    echoes a deterministic, content-derived response so unrelated tests don't
    need to script every call.
    """

    def __init__(self) -> None:
        self._queue: deque[CompletionResult] = deque()
        self._by_prompt_version: dict[str, deque[CompletionResult]] = {}

    def respond_with(self, result: CompletionResult) -> None:
        self._queue.append(result)

    def respond_to(self, prompt_version: str, result: CompletionResult) -> None:
        self._by_prompt_version.setdefault(prompt_version, deque()).append(result)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        keyed = self._by_prompt_version.get(request.prompt_version)
        if keyed:
            return keyed.popleft()
        if self._queue:
            return self._queue.popleft()
        if request.output_schema is not None:
            raise NoScriptedResponseError(
                f"MockProvider has no scripted response for prompt_version="
                f"{request.prompt_version!r}, output_schema="
                f"{request.output_schema.__name__!r}; call respond_with() or "
                "respond_to() first."
            )
        return CompletionResult(
            text=f"mock:{len(request.rendered_prompt)}",
            model_id=request.model_id,
            stop_reason="end_turn",
            usage=CompletionUsage(),
        )
