# Plan 1 of 2 — ASES Orchestrator Implementation Plan

**System:** `agentic-software-orchestrator` (Python 3.13)
**Status:** Proposed
**Date:** 2026-09-12
**Companion:** `03-URL-SHORTENER-WORKLOAD-PLAN.md` — the workload this orchestrator drives. The two documents are deliberately separate: **the orchestrator must contain no URL-shortener knowledge, and the workload must contain no orchestrator knowledge** (architectural rules C-16 and C-17).

---

## 0. Locked decisions

| Decision | Choice |
|---|---|
| Runtime | Python 3.13 |
| Event store | PostgreSQL 16 — durable, append-only, source of truth |
| Workload cache/queue | Redis 7 — **workload only**, never used by the kernel |
| LLM access | `LLMProvider` protocol; Anthropic adapter + cassette replay (default) + mock |
| Model selection | Capability-based `ModelRouter`; agents never name a model |
| Workload authorship | **Fully agent-generated from the requirement alone** |
| Stage 1 scope | **Greenfield only** — well-defined and ambiguous requirement classes |
| Brownfield analyzer | **Deferred to Stage 2.** Roslyn, out-of-process, once greenfield is solid |
| Database topology | One database `ases`; schemas `control` + `workload_test`; two roles. See `04-DATA-AND-MIGRATION-STRATEGY.md` |
| Observability | Custom dashboard only; OpenTelemetry exported to file. No Grafana, no Prometheus |
| Execution topology | Orchestrator on host; Docker Compose for Postgres + Redis only |

---

## 1. Architectural invariants (non-negotiable, enforced in code)

These come from rule set C and are implemented as assertions, not conventions.

1. The event log is authoritative; all state is `fold(events)`. Checkpoints are a derived cache and may be deleted at any time without loss.
2. Agents never control workflow progression. An agent's return value is an artifact, never a routing decision.
3. Agents never call other agents.
4. Agents never bypass the `PolicyEngine`.
5. Agents never write to the real repository — only to a sandbox workspace.
6. Tool execution occurs only through the registry. **Unknown tool = DENY. Unknown action = DENY.**
7. Destructive or production-affecting actions require human approval.
8. Retry occurs only when the tool is classified safely retryable.
9. Human approval is bound to an artifact **version hash**; an upstream change invalidates dependent approvals.
10. Deterministic validation is authoritative. **LLM output is a proposal, not a source of truth.**

**Enforcement test:** a dedicated `tests/invariants/` suite attempts to violate each rule (an agent returning a next-node hint, an agent calling an unregistered tool, a write outside the sandbox root, a retry of a non-idempotent external tool) and asserts each attempt is refused and logged as a policy violation.

---

## 2. Repository layout

```
agentic-software-engineering-system/
  src/
    orchestrator/                # THE ORCHESTRATOR — no workload knowledge
      ases/
        kernel/                  # CONTROL PLANE — may not import agents/providers/codebase
          events.py              event types, hash chain
          store/                 EventStore protocol; postgres.py, jsonl.py
          state.py               RunState fold, node state machine
          graph.py               WorkflowGraph, DAG validation, cycle budgets
          scheduler.py           readiness, parallel dispatch, barrier/join
          gates.py               entry/exit gates -> PASS | FAIL | ESCALATE
          policy.py              declarative guardrail engine
          tools/                 registry.py, classification.py, dotnet.py, fs.py, git.py
          recovery.py            retry, timeout, fallback, compensation, safe-stop
          replan.py              dirty propagation, minimal re-execution, approval revocation
          checkpoint.py          derived snapshots
          cancellation.py        cooperative cancellation tokens
          metrics.py             reliability metrics derived from the log
        agents/                  requirements, architect, decomposer, implementer,
                                 tester, reviewer, docs, release
        context/                 store.py, lineage.py, retriever.py
        providers/               base, anthropic, cassette, mock, router; prompts/, cassettes/
        codebase/                base.py only in Stage 1; Roslyn client in Stage 2
        validation/              schema, static, dynamic, critic; pipelines/{dotnet,python}
        sandbox/                 workspace.py — git worktree, writable-path enforcement
        observability/           correlation, logging, otel (file exporter)
        interfaces/              cli.py (typer); web/ (FastAPI + SSE + dashboard)
        contracts/               pydantic artifact models — the shared vocabulary
        workflows/               greenfield.yaml, greenfield_ambiguous.yaml  (Stage 1)
                                 brownfield.yaml, test_docs.yaml            (Stage 2)
        policies/                change_control, security, compliance, autonomy (YAML)
        migrations/              Alembic — control schema only
    url-shortener/               THE WORKLOAD — agent-generated; empty by design
      src/  tests/  README.md
  workloads/                     WORKLOAD HARNESS — hand-authored, agent-invisible
    url-shortener/
      REQUIREMENTS.md            the run inputs                    (agent-visible)
      CONTRACT.md                the fixed HTTP contract           (agent-visible)
      conformance/               black-box acceptance oracle       (NEVER agent-visible)
      baseline/                  frozen, reviewed greenfield output
      GROUND_TRUTH.md            impact expectations               (Stage 2)
  tools/RoslynIndexer/           STAGE 2 — .NET analyzer, JSON over stdio
  tests/                         unit/  integration/  invariants/
  infra/                         docker-compose.yml (postgres + redis only), README
  scripts/                       dev-up / dev-down (.ps1 and .sh) — the real entry points
  runs/                          exported event logs, artifacts, OTel traces
  docs/
  pyproject.toml  Makefile  .env.example  .gitignore  .gitattributes
```

**Why the workload appears in two places.** `src/url-shortener/` is the tree the
sandbox copies for agents to work in. `workloads/url-shortener/` holds the
harness — and critically the conformance oracle, which must *never* travel into
the sandbox. Since the implementation and its tests are both agent-generated,
physical separation is what makes the oracle's independence structural rather
than a matter of prompt discipline.

---

## 3. Workflow graph — now parameterized per requirement class

The base plan had one graph. The scope expansion (greenfield / brownfield / test-and-docs / ambiguous) means **the graph shape is selected by requirement class**. This is not a convenience — it is the clearest evidence for the "non-linear" claim, because it shows the orchestrator is not a fixed pipeline.

| Stage | Class | Graph | Notably |
|---|---|---|---|
| **1** | `greenfield` | Full graph, no `CODEBASE_INDEX` or `IMPACT_ANALYSIS`; includes `SCAFFOLD` | Nothing to analyze yet |
| **1** | `greenfield_ambiguous` | Full graph + a **clarification cycle**: `REQ_ANALYSIS → GATE_1 → (rejected) → REQ_ANALYSIS`, bounded at 3 | Demonstrates a governed backward edge driven by a human |
| 2 | `brownfield` | Full graph including Roslyn index + impact | The canonical path |
| 2 | `test_docs` | **Reduced graph** — no `ARCH_DESIGN`, no Gate 2, migration policy disabled | Demonstrates class-dependent gate topology |

Stage 1 ships two shapes, which is enough to prove the graph is selected rather than fixed. The strongest evidence — the reduced `test_docs` graph — arrives in Stage 2.

Node lifecycle, gate semantics and join semantics (`ALL` / `ANY` / `QUORUM(n)`) are unchanged from `00-IMPLEMENTATION-PLAN.md` §4.3–§4.4.

---

## 4. Phased build

Dependency note: phases are sequential except **6 and 7, which may proceed in parallel** once 5 is complete.

### Phase 0 — Foundation & contracts

- `git init`; repo scaffold; `orchestrator/` and `workloads/` boundaries created empty
- Python project: `uv`, `ruff`, `mypy --strict`, `pytest`, `pytest-asyncio`
- `infra/docker-compose.yml`: Postgres 16 + Redis 7. **One database, two schemas, two roles** — `control` (owner `ases_control`) and `workload_test` (owner `workload_app`), with grants and resource limits per `04-DATA-AND-MIGRATION-STRATEGY.md` §1
- Control-plane schema via **Alembic**: `runs`, `events`, `artifacts`, `lineage`, `approvals`, `policy_violations`. `events` made append-only by both privilege revocation and a `BEFORE UPDATE OR DELETE` trigger; `prev_hash` unique
- **Automated database bootstrap** — `ases db bootstrap` (idempotent roles/schemas/grants/limits) and `ases db upgrade` (Alembic), both wired into `scripts/dev-up.ps1` / `dev-up.sh` behind a Compose healthcheck. **No manual SQL at any point.** Deliberately *not* using `/docker-entrypoint-initdb.d/`, which only fires on an empty volume — see `04` §1.3
- `ases db reset-workload` (drops and recreates `workload_test`, preserves the audit log) for use between runs
- Named volume for Postgres; Redis deliberately ephemeral — `04` §1.5
- Startup refuses to run if the Alembic head does not match the database
- Pydantic artifact contracts: `RequirementSpec`, `AmbiguityRegister`, `ImpactReport`, `DesignSpec`, `ADR`, `TaskGraph`, `CodePatch`, `TestSuite`, `ReviewReport`, `PolicyViolation`, `ReleaseReport`, `RunSummary`
- `.env.example`; CI skeleton (lint, typecheck, test, plus a .NET leg that builds and tests the generated workload)
- **No hand-written .NET skeleton** — see §6, item 2

*Exit:* on a machine with Docker and `uv`, `git clone && cp .env.example .env && ./scripts/dev-up.sh` yields a working database with correct roles, schemas, grants and control tables having run **zero manual SQL** — and a second `dev-up` changes nothing and fails nothing. Contracts import; CI green.

### Phase 1 — Deterministic kernel (zero LLM)

- Event model; `prev_hash = sha256(canonical_json(prev_event))`; genesis event
- `PostgresEventStore` + `JsonlEventStore`; `ases export <run_id>` / `ases replay <file>`
- `RunState = fold(events)`; node state machine (base plan §4.3)
- `WorkflowGraph`: node/edge model, DAG validation, declared cycle budgets, join semantics
- Dependency resolution and readiness evaluation
- Parallel scheduler (asyncio task group) with barrier synchronization
- Entry/exit gate framework returning `PASS | FAIL | ESCALATE`
- Cooperative cancellation and safe-stop
- Checkpoint/recovery, explicitly derived and disposable
- **Lineage DAG** (`context/lineage.py`) — built here, not later, because Phase 7 depends on it
- Kernel test suite driven entirely by **fake agents**

*Exit:* the full graph executes end to end with fake agents; killing the process mid-run and resuming reproduces identical state; `export → replay` yields a byte-identical fold.

### Phase 2 — LLM boundary

- `LLMProvider` protocol; Anthropic adapter (`claude-opus-5`, `claude-sonnet-5`); cassette provider; mock provider
- Cassette key = `sha256(prompt_version + rendered_prompt + output_schema)` — stable across model swaps
- Model capability taxonomy (`reasoning`, `context_size`, `structured_output`, `latency_class`, `cost_class`) and `ModelRouter`
- Versioned prompt registry
- Structured output validation with exactly one bounded repair attempt
- Token and cost metadata emitted into the event log
- **Scoped context retriever** — agents receive summaries plus explicitly requested artifacts, never full run history

*Exit:* the same run replays identically twice offline; a model swap in the router does not invalidate cassettes.

### Phase 3 — Tooling, sandbox & policy substrate

- Tool registry, **deny-by-default**. Each tool declares `{idempotent, side_effect: none|sandbox|external, compensator, requires_approval, timeout}`
- Sandbox workspace: git worktree, enforced writable-path root, resource and wall-clock caps
- .NET tools: `dotnet new`, `restore`, `build`, `test`, `format`, `list package --vulnerable`
- `PolicyEngine` + YAML rule packs (change control, security, compliance, autonomy)
- Per-agent capability manifests; self-escalation attempts logged as violations
- **Prompt-injection containment**: repository content is delimited, labelled untrusted, never instruction-bearing
- **Secret scanning** of every generated artifact

*Exit:* the invariants suite (§1) passes — every violation attempt is refused and logged.

### Phase 4 — First vertical slice (greenfield)

- Agents: `requirements`, `architect`, `decomposer`, `implementer`, `tester`
- Human Gate 1 (requirement sign-off) and Gate 2 (design approval), both bound to artifact version hash
- **`SCAFFOLD` node — contract-first, inserted between Gate 2 and parallel implementation.** The approved `DesignSpec` is materialised into the solution skeleton *before* any parallel work begins: `.sln`, `.csproj` files, `Directory.Packages.props` with pinned versions, and the interface/DTO signatures the design declared. Still fully agent-derived — it is generated from the Architect's own artifact, not hand-written. See §7.1 for why this node exists.
- **`dotnet build` as an exit gate on every implementation task**, not only at the barrier — compile failures surface per task, not as a pile-up at integration
- Compiler diagnostics (file, line, `CSxxxx`, message) fed back verbatim into a bounded `REPAIR` pass
- Validation L1–L3 on the `.NET` pipeline
- Migration pipeline per `04-DATA-AND-MIGRATION-STRATEGY.md` §4 — generate and classify, never blindly apply
- **End-to-end run: a requirement becomes a compiling, test-passing ASP.NET Core service**
- Independent black-box conformance oracle executed at the quality gate (see companion plan §5)
- **Freeze the output** to `workloads/url-shortener/baseline/` and record cassettes from the successful run

*Exit:* `ases run greenfield` produces a service that passes `dotnet build`, `dotnet test`, and the conformance oracle.

### Phase 5 — Governance & recovery

- Retry policy consulting tool idempotency class; timeout policy; bounded backoff
- Fallback strategy (degraded artifact plus mandatory human flag)
- **Two distinct recovery mechanisms**, selected by side-effect class:
  - *sandbox rollback* — total and free; discard the worktree
  - *external compensation* — partial and ordered; invoke each completed tool's compensator in reverse
- Safe-stop on budget, iteration or policy exhaustion
- Human approval queue; approval records carry the artifact hash
- Approval revocation on upstream change
- Fault-injection harness (`--inject failure@NODE`, `--inject flaky-tool`, `--inject budget-exhaustion`)

*Exit:* each of retry, fallback, sandbox rollback, external compensation, safe-stop, and approval revocation is demonstrated under fault injection and visible in the event log.

### Phase 6 — Brownfield reasoning (Roslyn) — **DEFERRED TO STAGE 2**

> Out of scope until the greenfield path is solid. Retained here so the interface seam is designed for now and built later: `codebase/` sits behind a `CodebaseAnalyzer` protocol that the graph depends on, so Stage 2 is a plug-in rather than a retrofit. `GROUND_TRUTH.md` is deferred with it, since it can only be authored against the frozen baseline that Phase 4 produces.

- `tools/RoslynIndexer/` — .NET 9 console app using `Microsoft.CodeAnalysis.CSharp`, JSON over stdio
- Python `roslyn_client.py` with a versioned JSON contract and a schema-validated response
- Symbol table; project/assembly reference graph; type and member index
- API map: ASP.NET Core attribute routing **and** minimal-API endpoint registration
- Data flow: entity → `DbContext` → migration → endpoint → DTO
- Impact analysis: blast radius from a changed symbol set, ranked by distance
- `ground_truth.py`: precision/recall scoring against `GROUND_TRUTH.md`
- `brownfield_implementer` agent producing patches rather than whole files

*Exit:* impact analysis on the frozen baseline scores against ground truth at an agreed threshold.

### Phase 7 — Re-planning

- Artifact version detection via content hash
- Dirty propagation across the lineage DAG
- Downstream invalidation → `STALE`
- Minimal re-execution set computation
- Approval invalidation for any stale, previously approved node
- Re-plan artifact generation (what will re-run, and why)
- Resume from the nearest valid state

*Exit:* amending an approved `RequirementSpec` mid-run re-runs only the affected subgraph and forces re-approval of Gate 2.

### Phase 8 — Validation stack

- L1 schema; L2 static; L3 dynamic (coverage delta); L4 semantic critic with rubric and mandatory `file:line` citations; L5 human
- `ValidationPipeline` strategy with a `.NET` implementation (the workload) and a `python` implementation (used for the orchestrator's own CI — see §6, item 6)
- Fault-injection tests for the validation path itself

*Exit:* a deliberately defective patch is caught at the lowest applicable layer, and the layer is recorded.

### Phase 9 — Observability & metrics

- Correlation IDs: `run_id` / `node_id` / `attempt_id` / `trace_id`, on every log line and event
- Structured JSON logging
- OpenTelemetry traces exported to `runs/<run_id>/traces.otlp.jsonl` (file exporter — no collector, no Grafana)
- Metrics **derived from the event log**: success rate, retry frequency, rollback frequency, fallback rate, MTTR (node and run), stage and end-to-end latency at p50/p95, human intervention rate, gate rejection rate, first-pass validation rate, token and dollar cost
- Audit-event query API
- CLI (`typer`) and the custom web dashboard: live DAG, event timeline, approval queue, artifact and lineage inspector, policy-violation panel, metrics panel

*Exit:* an entire run is legible in the browser without reading a log file.

### Phase 10 — Demonstration

Four scenarios (companion plan §2) and eleven control demonstrations: parallel execution, barrier join, bounded retry, policy interception, human approval, approval revocation, sandbox rollback, external compensation, safe-stop, dynamic re-plan, offline cassette replay — closing with a final engineering summary generated **from the event log**.

---

## 5. Proposed priority tiers (your Q4 is still open)

**Stage 1 (agreed): Phases 0, 1, 2, 3, 4, 5, 7, 8, 9. Phase 6 deferred.**

Within Stage 1, if time compresses:

- **Tier 1 — the thesis.** Phases 0, 1, 2, 3, 5, 7 plus the governance demos. Fully satisfies the brief's differentiator on its own.
- **Tier 2 — the evidence.** Phase 4 (the vertical slice), Phase 8 (validation), and the dashboard.
- **Tier 3 — polish, cut first.** OTel file tracing, metrics refinement, the second greenfield run.

**Stage 2 (deferred): Phase 6** — Roslyn indexer, impact analysis, ground-truth scoring, `brownfield` and `test_docs` graphs.

---

## 6. Consequences of the locked answers that change the earlier update set

1. **Legacy Bank and the Python `ast` indexer are out of scope.** Choosing Roslyn/URL-Shortener for brownfield makes them redundant. Phase 10's "Brownfield Legacy Bank scenario" is replaced by the URL Shortener brownfield run.
2. **Phase 0's ".NET URL Shortener skeleton" is removed** — it contradicts "fully agent-generated from the requirement alone." A checked-in skeleton would pre-empt the very generation being demonstrated. Replaced by a registered `dotnet new webapi` **tool** the implementer agent may invoke.
3. **`GROUND_TRUTH.md` moves to Stage 2.** It cannot be authored before the code it describes exists, and it has no consumer until the Roslyn analyzer is built. Phase 4 still freezes and commits the baseline it will be written against, so the sequencing is preserved.
4. **An independent conformance oracle is added** (companion plan §5). With the implementation *and its tests* both agent-generated, deterministic validation would otherwise be self-graded. The oracle is black-box, hand-authored, fixed in advance against a contract stated in the requirement, and never shown to the implementer agent.
5. **Grafana and Prometheus are removed** from `docker-compose.yml`. OTel goes to file.
6. **The Python validation pipeline loses its workload.** The strategy interface is retained and pointed at the orchestrator's own source for CI, which keeps update #13's two-pipeline design honest rather than vestigial.
7. ~~Two separate Postgres databases are required.~~ **Superseded** by `04-DATA-AND-MIGRATION-STRATEGY.md` §1: one database with two schemas and two roles. Isolation here is a privilege problem, not a container problem — the database boundary was never the enforcing mechanism, the grants were.
8. **Brownfield and Roslyn deferred to Stage 2.** Stage 1 covers both requirement-quality classes (well-defined and ambiguous) within greenfield, so the declared scope is still exercised.
9. **Schema lifecycle is now specified** for both systems — Alembic for the control plane, gated EF Core migrations for the workload. It was missing from every earlier revision.

---

## 7. Risks specific to this revision

### 7.1 On "agent-generated code may not compile" — corrected

My earlier framing of this as a demo-blocking risk was overstated, and the challenge to it was fair. The correction, and why the agent roster does not by itself resolve it:

**Agents are producers and reviewers; none of them is a compiler.** `requirements`, `architect`, `decomposer` and `reviewer` all operate on text. `reviewer` is an LLM and is therefore L4 *advisory* validation — rule C-15 ("LLM output is a proposal, not a source of truth") applies to the reviewer exactly as much as to the implementer. The only thing that proves compilation is `dotnet build`.

The realistic failure modes are mundane rather than dramatic:

| Mode | Cause | Addressed by |
|---|---|---|
| **Interface drift across parallel tasks** — the largest by far | Task A defines `IUrlRepository`, task B consumes an assumed signature; each compiles alone, integration fails at the barrier | `SCAFFOLD` node freezes interfaces before fan-out |
| Hallucinated APIs or package versions | Training data vs. .NET 9 / EF Core 9 specifics | Pinned Central Package Management + a `dotnet restore` probe in `SCAFFOLD` |
| Nullable-reference warnings under `-warnaserror` | CS8618 and friends are endemic in generated C# | Compiler diagnostics fed back verbatim to `REPAIR` |
| Missing `ProjectReference` or DI registration | Compiles, fails at runtime | `SCAFFOLD` owns project wiring; integration tests catch DI |

With `dotnet build` as a per-task exit gate and a closed diagnostic-feedback loop, the risk changes character: **it is no longer "the demo is blocked," it is "the run takes more repair iterations."** That costs time and tokens, not correctness. The residual blocker only appears if repair exhausts its bound, which is handled by the escalating ladder below.

**Escalating repair ladder** (bounded, in order): repair in place from diagnostics → regenerate the single failing file → regenerate the whole task from its contract → emit a minimal compiling stub with a `TODO` and a mandatory human flag. The last rung guarantees the graph always reaches a terminal state with a legible artifact, never a hang.

### 7.2 Remaining risks

| Risk | Why it is new | Mitigation |
|---|---|---|
| Live generation varies run to run | LLM non-determinism | Cassettes recorded from a successful run; the demo replays deterministically |
| Agent-generated EF migrations reach a real database | New external side effect | Generate-and-classify pipeline, raw SQL banned, human gate on the SQL, transactional DDL, disposable schema — `04-DATA-AND-MIGRATION-STRATEGY.md` §4 |
| Availability coupling on a shared Postgres instance | Consequence of collapsing to one database | `statement_timeout`, `lock_timeout`, `idle_in_transaction_session_timeout` and a connection limit on the workload role only |
| Deferring brownfield weakens the brief's Feature 3 | Stage 1 scope cut | `CodebaseAnalyzer` seam designed now; Stage 2 plugs in rather than retrofits |
| One workload carries all scenario classes | Legacy Bank removed | Acceptable, and arguably stronger — a single evolving codebase shows lifecycle progression rather than disconnected demos |
