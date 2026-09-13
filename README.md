# ASES — Agentic Software Engineering System

A governed orchestration kernel that drives LLM agents through the full SDLC,
turning one natural-language requirement into a reviewable engineering outcome.

> **Status: Phase 0, Phase 1, Phase 2 (LLM boundary) and Phase 3
> (tooling/sandbox/policy) complete.** Phase 4 (first vertical slice: the
> greenfield agents) not started.
>
> | Area | State |
> |---|---|
> | Scaffold, infra, CI, docs | done |
> | Kernel: events, hash chain, state fold, graph validation | **done and tested** |
> | Kernel: scheduler, gates, cancellation, checkpoint, lineage | **done and tested** — parallel dispatch, bounded repair cycles, human approval incl. resuming across separate `run()` calls, budget-based safe-stop |
> | `JsonlEventStore` (portable) + `PostgresEventStore` (source of truth) | **done and tested against a live container** — see docs/06 for how to verify this yourself |
> | Alembic (control schema: `events` table + 4 derived views) | done |
> | `ases db bootstrap` / `upgrade` / `reset-workload` | done, idempotency verified |
> | `ases export` / `replay` | done — replay verified to work with Postgres stopped entirely |
> | Pydantic artifact contracts (14, one per workflow `produces:` kind) | done |
> | `workflows/greenfield.yaml` (Stage 1 graph, loads and validates) | done |
> | `LLMProvider` protocol, Anthropic adapter, cassette + mock providers | **done and tested** — cassette key excludes the model id (a router swap replays unchanged); mapping to the Anthropic SDK tested against an injected fake client, never the network |
> | Capability-based `ModelNeeds` / `ModelSpec` / `ModelRouter` | done — catalog is exactly `claude-opus-5` / `claude-sonnet-5` per docs/02 Phase 2 |
> | Versioned prompt registry (`providers/prompts/registry.py`) | done as a mechanism — no real prompt is registered yet; the first one lands with the first agent (Phase 4) |
> | Structured-output validation, exactly one bounded repair attempt | done and tested (`providers/structured.py`) |
> | Scoped context retriever (`context/retriever.py`) | **partial, by necessity** — artifact summaries from `RunState` work now; full-content `fetch()` deliberately raises `ArtifactContentUnavailableError`, since no content-addressed artifact store exists yet to resolve a hash back to a payload (see the module docstring and the Known Gaps note below) |
> | Token/cost metadata on every completion | done at the boundary (`CompletionResult.usage`); not yet flowing into a live run's event log, since no `NodeExecutor` calls an agent yet — `kernel.state`'s `LLM_COMPLETED` fold case already exists and is ready for Phase 4 to wire up |
> | Deny-by-default `ToolRegistry` (`kernel/tools/registry.py`) | **done and tested** — unknown tool, write outside `writable_paths`, and timeout are all refused before a handler ever runs |
> | `dotnet` tools: `new`, `restore`, `build`, `test`, `format`, `list package --vulnerable` | done and tested against an injected fake subprocess runner — exactly the six docs/02 Phase 3 names, no more |
> | `fs` tools (`read_file`, `write_file`), confined to the sandbox root | done and tested, including two confirmed-then-fixed path-traversal exploits (see Known Issues Found below) |
> | `git` tools: `status`, `diff` (read-only only) | done — deliberately no mutating action; see the module's own scoping note |
> | Secret-scanning tool (`kernel/tools/security.py`) | done and tested — pattern-based (AWS/Anthropic/OpenAI/GitHub/Slack keys, private-key blocks, hardcoded credential assignments) |
> | `PolicyEngine` + 4 YAML rule packs (change control, security, compliance, autonomy) | done and tested — a pack can only tighten a tool's own `requires_approval` floor, never loosen it |
> | `CapabilityManifest` + self-escalation denial | done and tested — a request for a tool outside an actor's manifest is `DENY`, never silently dropped |
> | Sandbox workspaces (`sandbox/workspace.py`) | **done and tested for real** — actual `git worktree add`/`remove`, not mocked; sandbox rollback (discarding the worktree) verified to leave the real repository untouched |
> | Prompt-injection containment (`providers/untrusted.py`) | done and tested — content-derived nonce delimiters, with a redaction pass as defence in depth |
> | Real agents, dashboard (Phase 4+) | not started, by design — see docs/02 |
> | Dynamic subgraph admission wired into a live run; full re-planning (Phase 7) | not started — see `kernel/scheduler.py`'s module docstring for the exact boundary |
>
> 365 tests (unit + integration + invariants), mypy `--strict`, ruff clean.
> `./scripts/dev-up.sh` now works end to end - see
> [`docs/06-VALIDATION-GUIDE.md`](docs/06-VALIDATION-GUIDE.md) to reproduce
> and cross-check everything above yourself.
>
> **Known gap carried forward:** `docs/02`'s repository layout names
> `context/store.py` (a content-addressed artifact store) but no phase bullet
> ever builds it, and no code path persists an artifact's actual payload
> anywhere a hash could resolve back to it — the event log only ever records
> the hash (by design, docs/05 §8). This was surfaced, not silently patched
> over, while building the scoped context retriever above; it blocks nothing
> in Phase 2 and should be picked up when Phase 4 gives agents real content
> worth storing.
>
> **Security issues found and fixed while building Phase 3, not merely
> claimed fixed:** `kernel.tools.fs`'s path confinement and the registry's
> `writable_paths` check both initially relied on `PurePosixPath`, which is
> purely lexical and Windows-unaware. Verified interactively on this
> machine: a backslash-form Windows path (`C:\Windows\System32\x`) becomes a
> single opaque path segment under `PurePosixPath`, which `pathlib.Path`
> then resolves as a real absolute path on a Windows host — silently
> discarding the sandbox root during a join. Both the `fs.write_file`
> handler and the registry's structural `path_args` check shared this gap;
> both are now fixed via one shared validator
> (`kernel.tools.classification.is_safe_relative_path`), and both fixes are
> pinned by regression tests (`test_tool_fs.py`,
> `test_tool_registry.py`) that fail again if the check regresses.

---

## The idea in one paragraph

A **deterministic control plane** executes an explicit workflow graph, recording
every transition to an append-only, hash-chained event log from which all state
is folded. It dispatches work to **stateless LLM agents** that return typed
artifacts and nothing else — they never decide what runs next, never call each
other, and never write to the real repository. They reach the outside world only
through a deny-by-default tool registry, inside a disposable sandbox, after a
policy engine and entry/exit gates allow it. Humans hold approval authority,
bound to artifact content hashes. Audit, lineage and reliability metrics are
*derived* from the event log rather than separately instrumented.

The inversion everything rests on:

> **The engine owns control flow. Agents own content.**

Let the model drive control flow and you can no longer *enforce* governance,
only request it. Bounded retries, rollback, gates, budget caps and replay are
only implementable when control flow lives in deterministic code.

---

## Two systems

| | Path | Written by |
|---|---|---|
| **Orchestrator** | [`src/orchestrator/`](src/orchestrator/) | Hand-written (Python 3.13) |
| **URL Shortener** | [`src/url-shortener/`](src/url-shortener/) | **Generated by the orchestrator** (.NET 10) |

The workload is empty by design. Hand-writing it would pre-empt the capability
the prototype exists to demonstrate.

Each is independent of the other: the orchestrator contains no URL-shortener
knowledge, and the workload contains no orchestrator knowledge.

---

## Getting started

Requires **Docker**, **uv**, and **.NET 10 SDK** (for the generated workload).

```powershell
Copy-Item .env.example .env
uv sync --all-extras
.\scripts\dev-up.ps1
```

```bash
cp .env.example .env
uv sync --all-extras
./scripts/dev-up.sh
```

`dev-up` starts Postgres and Redis, waits on the healthcheck, creates roles,
schemas and grants (idempotently), then applies control-plane migrations. It
runs **zero manual SQL** and is safe to re-run — a second run must change
nothing and fail nothing.

```powershell
.\scripts\dev-down.ps1           # stop, keep data
.\scripts\dev-down.ps1 -Purge    # stop and DESTROY the audit log
```

`make` is not installed on the primary dev machine, so the scripts are the
single source of truth. The `Makefile` only delegates to them, for CI on Linux.

---

## Layout

```
src/orchestrator/ases/
  kernel/        deterministic control plane — contains zero LLM calls
  agents/        LLM workers; one typed artifact each
  context/       content-addressed artifacts, lineage DAG, scoped retrieval
  providers/     LLM protocol, Anthropic + cassette + mock, capability router
  codebase/      CodebaseAnalyzer protocol (Stage 1); Roslyn client (Stage 2)
  validation/    L1 schema, L2 static, L3 dynamic, L4 critic
  sandbox/       git-worktree workspaces with writable-path enforcement
  observability/ correlation IDs, structured logs, OTel file export
  interfaces/    Typer CLI, FastAPI + SSE dashboard
  contracts/     pydantic artifact models — the shared vocabulary
  workflows/     workflow graphs as YAML, one per requirement class
  policies/      guardrail rule packs as YAML
  migrations/    Alembic, control schema only

workloads/url-shortener/   requirements, contract, conformance oracle, baseline
tools/RoslynIndexer/       Stage 2 .NET analyzer
infra/                     Compose: Postgres + Redis only
tests/invariants/          one test per architectural rule
```

---

## Architectural invariants

Enforced by [`tests/invariants/`](tests/invariants/) — each test *attempts* a
violation and asserts it is refused and logged.

1. The event log is authoritative; state is `fold(events)`. Checkpoints are a
   disposable cache.
2. Agents never control workflow progression.
3. Agents never call other agents.
4. Agents never bypass the policy engine.
5. Agents never write to the real repository.
6. Tools run only through the registry. Unknown tool = DENY. Unknown action = DENY.
7. Destructive or production-affecting actions require human approval.
8. Retry only when the tool is classified safely retryable.
9. Approval binds to an artifact **version hash**; an upstream change revokes it.
10. Deterministic validation is authoritative. **LLM output is a proposal.**

Rule 1 is also a layering rule: `kernel/` may not import `agents/`,
`providers/`, `codebase/` or `validation/`. A static import check fails the
build otherwise — which is what lets the entire kernel test suite run with fake
agents and no LLM in the process.

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/05-ARCHITECTURE.md`](docs/05-ARCHITECTURE.md) | **Start here.** Layers, abstractions, event taxonomy, runtime sequence |
| [`docs/02`](docs/02-ORCHESTRATOR-IMPLEMENTATION-PLAN.md) | Orchestrator plan and build phases |
| [`docs/03`](docs/03-URL-SHORTENER-WORKLOAD-PLAN.md) | Workload runs, contract, conformance oracle |
| [`docs/04`](docs/04-DATA-AND-MIGRATION-STRATEGY.md) | Schema lifecycle, bootstrap, migration safety |
| [`docs/01`](docs/01-PLAN-DELTA-ANALYSIS.md) | Why each decision changed |
| [`docs/00`](docs/00-IMPLEMENTATION-PLAN.md) | Superseded — retained as decision lineage |

---

## Scope

**Stage 1 (current):** greenfield, covering both well-defined and ambiguous
requirements.

**Stage 2 (deferred):** brownfield reasoning via Roslyn, impact analysis, and
the reduced test-and-documentation graph.
