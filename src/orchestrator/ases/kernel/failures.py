"""How a node failed, as a closed vocabulary rather than a prose string.

Live run `009ea59f-...` is the case this exists for. `repair_api` failed
twice with `CodePatch: Invalid JSON: EOF while parsing a value at line 1
column 0 [input_value='']` - which reads like the model emitted malformed
JSON, and is not what happened. The model emitted *nothing*: the response
hit `max_tokens` while still inside its thinking block, so no text block was
ever opened, and `anthropic_provider._first_text` returned `""`, which
pydantic then reported as invalid JSON. Two very different conditions - "the
model wrote bad JSON" and "the model never got to write anything" - arrived
at the operator as the same sentence, and the run's own halt reason blamed
`build_api`'s exhausted `cycle_budget` rather than the protocol failure that
actually caused it.

Classification is the *executor's* job, never the kernel's: only the layer
that made the call knows whether a non-zero exit was a compiler error, a
denied tool, or a model that returned nothing. The scheduler records what it
is told (plus `TIMEOUT`, which only it can observe) and routes on the graph's
edges exactly as before - `failure_kind` is diagnostic metadata on the
`NODE_FAILED` event, deliberately not an input to routing. Making it one
would put control flow back into the failing component's hands, which is the
one thing CLAUDE.md section 15 reserves for the kernel.

There is deliberately no `REPAIR_FAILURE` member. "A repair failed" is not a
*kind* of failure, it is a failure at a particular graph position, and the
event log already records position (`Event.node_id`) alongside kind. A
repair node that failed with `AGENT_PROTOCOL_FAILURE` tells an operator
strictly more than one labelled `REPAIR_FAILURE` would, and a taxonomy
carrying both would let the same event be described two ways.
"""

from __future__ import annotations

from enum import StrEnum


class FailureKind(StrEnum):
    """Recorded on `NODE_FAILED`; see the module docstring."""

    #: An LLM produced output the agent plane could not use as an artifact:
    #: it failed schema validation, or was truncated before any output was
    #: emitted at all. The agent's *code* worked; the model's response did
    #: not satisfy the wire contract. Bounded repair at the LLM boundary
    #: (`providers.structured.complete_structured`) is already exhausted by
    #: the time this is reported.
    AGENT_PROTOCOL_FAILURE = "agent_protocol_failure"

    #: A compile or test step reported failure - the ordinary, expected
    #: outcome the repair cycle exists to act on. Not an orchestration
    #: problem: the system is working correctly when this happens.
    BUILD_FAILURE = "build_failure"

    #: A registered tool ran and reported failure, other than a build/test.
    TOOL_FAILURE = "tool_failure"

    #: The policy engine or an agent's own capability manifest refused the
    #: action. Distinct from `TOOL_FAILURE`: nothing ran.
    POLICY_DENIED = "policy_denied"

    #: The system failed, not the work: an unreachable provider, a handler
    #: with no registered executor, a budget exhausted mid-run.
    ORCHESTRATION_FAILURE = "orchestration_failure"


__all__ = ["FailureKind"]
