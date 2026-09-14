"""Structured-output validation with exactly one bounded repair attempt
(docs/02 Phase 2).

A provider never raises on a schema-validation failure (see
`base.CompletionResult.schema_error`); this module is what turns that failure
into either a second, corrected attempt or a definite exhaustion - never a
silent unbounded retry loop. "Exactly one" is a deliberate, small number:
this is the LLM boundary's own repair, distinct from and much narrower than
the workflow-level `REPAIRING` cycle (docs/05 section 5), which can afford a
larger, workflow-author-declared budget because it re-runs a whole node
rather than one completion.
"""

from __future__ import annotations

from ases.providers.base import (
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    TruncatedCompletionError,
)

#: How much more headroom a truncated request is retried with. A
#: multiplier, not a fixed number, so it scales with whatever ceiling
#: the calling agent chose; bounded to one application, exactly like the
#: schema repair below.
_TRUNCATION_HEADROOM = 2

#: Absolute ceiling the headroom retry will not exceed, so a badly-
#: configured caller cannot turn one retry into an unbounded spend. Well
#: inside the catalog models' per-response output limit.
_MAX_TOKENS_CEILING = 32_000


class StructuredOutputExhaustedError(RuntimeError):
    """The repair attempt also failed schema validation."""

    def __init__(self, schema_name: str, first_error: str, second_error: str) -> None:
        super().__init__(
            f"{schema_name}: repair attempt did not fix the schema error. "
            f"first attempt: {first_error} | repair attempt: {second_error}"
        )
        self.schema_name = schema_name
        self.first_error = first_error
        self.second_error = second_error


def _retry_after_truncation(request: CompletionRequest) -> CompletionRequest:
    """The same request with more room, and told to use it economically.

    Deliberately **not** `_repair_request`: a truncated response is not a
    model that answered wrongly, it is a model that never finished
    answering, and feeding it a "your previous response did not validate"
    instruction under the same ceiling reproduces the truncation at the same
    place. Live run `009ea59f-...` did exactly that and produced two
    byte-identical failures - the bounded repair was spent without ever
    changing the condition it was repairing.
    """
    assert request.output_schema is not None
    raised = min(request.max_tokens * _TRUNCATION_HEADROOM, _MAX_TOKENS_CEILING)
    instruction = (
        "Your previous response was cut off before it finished. Answer again, "
        "more concisely: include only what the schema requires, and keep any "
        "reasoning brief so the whole response fits."
    )
    return request.model_copy(
        update={
            "rendered_prompt": f"{request.rendered_prompt}\n\n{instruction}",
            "max_tokens": raised,
        }
    )


def _repair_request(request: CompletionRequest, schema_error: str) -> CompletionRequest:
    """The same request, with the validation failure appended as an
    additional user turn - so the model sees exactly what was wrong with its
    own prior output, rather than being asked to guess again from scratch."""
    assert request.output_schema is not None
    schema_name = request.output_schema.__name__
    repair_instruction = (
        f"Your previous response did not validate against the {schema_name} schema: "
        f"{schema_error}\nRespond again, correcting the error, and nothing else."
    )
    return request.model_copy(
        update={"rendered_prompt": f"{request.rendered_prompt}\n\n{repair_instruction}"}
    )


async def complete_structured(
    provider: LLMProvider, request: CompletionRequest
) -> CompletionResult:
    """Call `provider.complete(request)`; if the result's schema validation
    failed, retry exactly once with the error fed back, then give up.

    Requires `request.output_schema` to be set - this function exists
    entirely to bound a schema-repair loop, and has nothing to add over a
    plain `provider.complete(request)` for an unstructured request.
    """
    if request.output_schema is None:
        raise ValueError("complete_structured requires request.output_schema to be set")

    schema_name = request.output_schema.__name__

    first = await provider.complete(request)
    if first.schema_error is None:
        return first

    # Two different conditions, two different retries. Truncation gets more
    # room and a brevity instruction (`_retry_after_truncation`); a genuine
    # schema violation gets the validation error fed back under the same
    # ceiling (`_repair_request`). Telling them apart is the whole point -
    # see this module's note and `providers/base.CompletionResult.truncated`.
    if first.truncated:
        retried = await provider.complete(_retry_after_truncation(request))
        if retried.schema_error is None:
            return retried
        if retried.truncated:
            raise TruncatedCompletionError(
                schema_name,
                f"first attempt at max_tokens={request.max_tokens}, "
                f"retry at max_tokens={_retry_after_truncation(request).max_tokens}",
            )
        # It stopped truncating but is now failing validation for a real
        # reason: that is a schema failure, reported as one.
        raise StructuredOutputExhaustedError(
            schema_name, first.schema_error, retried.schema_error or ""
        )

    repaired = await provider.complete(_repair_request(request, first.schema_error))
    if repaired.schema_error is None:
        return repaired

    raise StructuredOutputExhaustedError(schema_name, first.schema_error, repaired.schema_error)
