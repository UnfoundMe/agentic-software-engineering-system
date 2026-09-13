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

from ases.providers.base import CompletionRequest, CompletionResult, LLMProvider


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

    first = await provider.complete(request)
    if first.schema_error is None:
        return first

    repaired = await provider.complete(_repair_request(request, first.schema_error))
    if repaired.schema_error is None:
        return repaired

    raise StructuredOutputExhaustedError(
        request.output_schema.__name__, first.schema_error, repaired.schema_error
    )
