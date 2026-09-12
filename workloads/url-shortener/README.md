# URL Shortener — workload harness

The **experimental apparatus** around the agent-generated service. Everything
here is hand-authored and is *not* produced by agents.

The generated code itself lives at [`src/url-shortener/`](../../src/url-shortener/).

| Path | What it is | Agent-visible? |
|---|---|---|
| `REQUIREMENTS.md` | The requirement texts fed to the orchestrator, one per run | **Yes** — this is the input |
| `CONTRACT.md` | The fixed HTTP contract, restated inside the requirement | **Yes** |
| `conformance/` | Black-box acceptance oracle | **No — never** |
| `baseline/` | Frozen, reviewed output of the greenfield run | Read-only input in Stage 2 |
| `GROUND_TRUTH.md` | Expected impact sets for brownfield analysis | **No** *(Stage 2)* |

## Why `conformance/` sits here and not beside the code

The sandbox copies `src/url-shortener/` for agents to work in. The oracle must
never travel with it.

The implementation *and its tests* are both agent-generated, so without an
external oracle the deterministic validation layer would be grading its own
homework — and the architectural rule "deterministic validation is
authoritative" would be hollow. Keeping the oracle physically outside the
agent's working tree is what makes its independence structural rather than a
matter of prompt discipline.

## Why `baseline/` exists

The greenfield run's output is frozen and committed here once it passes the
oracle and Gate 3. Later runs analyse and enhance the *baseline*, not whatever
the most recent live generation happened to produce — so brownfield impact
analysis is reproducible and gradeable regardless of run-to-run variance.

`GROUND_TRUTH.md` can only be written after the baseline exists, which is why it
is a Stage 2 artifact rather than a Phase 0 one.
