# ASES — Detailed Architecture

**Status:** Proposed
**Date:** 2026-09-12
**Scope:** Stage 1 (greenfield). Stage 2 seams are marked and designed, not built.

**Relationship to the other documents.** `02` says *what* is built and in what
order; `03` says *what the workload run looks like*; `04` says *how data and
schema are handled*. This document says **how the code is shaped** — the layers,
the abstractions, the contracts between them, and the runtime behaviour they
produce.

---

## 1. The one-paragraph architecture

A **deterministic control plane** (`kernel/`) executes an explicit workflow
graph, persisting every transition to an append-only hash-chained event log from
which all state is folded. It dispatches work to an **agent plane** (`agents/`)
of stateless, LLM-backed workers that return typed artifacts and nothing else.
Agents reach the outside world only through a **deny-by-default tool registry**,
only inside a **disposable sandbox**, and only after a **policy engine** and
**entry/exit gates** have allowed it. Humans hold approval authority at gates,
bound to artifact content hashes. Everything observable — lineage, metrics,
audit — is derived from the event log rather than separately instrumented.

---

## 2. Layering and the dependency rule

```
┌─────────────────────────────────────────────────────────────┐
│  interfaces/      CLI, web dashboard, SSE                   │
├─────────────────────────────────────────────────────────────┤
│  agents/          LLM workers        validation/  codebase/  │
├─────────────────────────────────────────────────────────────┤
│  providers/  context/  sandbox/  observability/             │
├─────────────────────────────────────────────────────────────┤
│  kernel/          graph · scheduler · events · gates ·      │
│                   policy · tools · recovery · replan        │
├─────────────────────────────────────────────────────────────┤
│  contracts/       pydantic artifact models (shared)         │
└─────────────────────────────────────────────────────────────┘
```

**The dependency rule, enforced by a test:** `kernel/` may import `contracts/`
and standard library only. It must never import `agents/`, `providers/`,
`codebase/` or `validation/`.

This is what makes the central claim testable rather than rhetorical. If the
kernel cannot import an agent, it cannot depend on an agent's judgement, and the
entire Phase 1 test suite runs against fake agents with no LLM anywhere in the
process. A static import check in `tests/invariants/test_layering.py` fails the
build if the rule is broken.

The inversion the whole system rests on:

> **The engine owns control flow. Agents own content.**
> An agent's return value is an artifact. It is never a routing decision.

---

## 3. Core abstractions

Signatures are indicative — the shape is the contract, not the exact typing.

### 3.1 Events and state

```python
# kernel/events.py


class EventType(StrEnum):
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_HALTED = "run.halted"  # safe-stop

    NODE_READY = "node.ready"
    NODE_ENTRY_GATE = "node.entry_gate"  # verdict in payload
    NODE_STARTED = "node.started"
    NODE_EXIT_GATE = "node.exit_gate"
    NODE_SUCCEEDED = "node.succeeded"
    NODE_FAILED = "node.failed"
    NODE_RETRY_SCHEDULED = "node.retry_scheduled"
    NODE_FALLBACK_TAKEN = "node.fallback_taken"
    NODE_CANCELLED = "node.cancelled"
    NODE_SKIPPED = "node.skipped"
    NODE_MARKED_STALE = "node.marked_stale"  # upstream changed

    ARTIFACT_PRODUCED = "artifact.produced"
    ARTIFACT_VALIDATED = "artifact.validated"
    ARTIFACT_REJECTED = "artifact.rejected"

    SUBGRAPH_PROPOSED = "subgraph.proposed"  # decomposer output
    SUBGRAPH_ADMITTED = "subgraph.admitted"
    SUBGRAPH_REJECTED = "subgraph.rejected"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_REVOKED = "approval.revoked"  # artifact hash changed

    POLICY_EVALUATED = "policy.evaluated"
    POLICY_VIOLATION = "policy.violation"

    TOOL_INVOKED = "tool.invoked"
    TOOL_SUCCEEDED = "tool.succeeded"
    TOOL_FAILED = "tool.failed"
    TOOL_COMPENSATED = "tool.compensated"

    LLM_REQUESTED = "llm.requested"
    LLM_COMPLETED = "llm.completed"

    REPLAN_TRIGGERED = "replan.triggered"
    REPLAN_COMPUTED = "replan.computed"
    CHECKPOINT_WRITTEN = "checkpoint.written"


class Event(BaseModel, frozen=True):
    seq: int  # monotonic per run, assigned by the store
    run_id: UUID
    event_id: UUID
    type: EventType
    node_id: str | None
    attempt: int | None
    actor: Actor  # kernel | agent:<name> | human:<id> | tool:<name>
    payload: Mapping[str, Any]
    created_at: datetime
    prev_hash: str  # sha256 of the previous event, canonicalised
    hash: str
```

```python
# kernel/store/base.py


class EventStore(Protocol):
    async def append(self, ev: UnsealedEvent) -> Event:
        """Seal (hash-chain) and persist. Raises on a forked chain."""

    def read(self, run_id: UUID, after_seq: int = 0) -> AsyncIterator[Event]: ...

    async def verify_chain(self, run_id: UUID) -> ChainVerification:
        """Recompute every hash. Tamper-evidence is only real if checked."""
```

Two implementations: `PostgresEventStore` (source of truth) and
`JsonlEventStore` (portable export/import, used by `ases export` / `ases replay`
and by unit tests that must run without a database).

```python
# kernel/state.py


def fold(run_id: UUID, events: Iterable[Event]) -> RunState:
    """The only way RunState is ever produced. There is no mutable state of record.

    Strict: applies LEGAL_TRANSITIONS and raises InvalidTransitionError on a
    move the machine does not permit. A log that cannot have happened stops the
    run loudly rather than being smoothed into a plausible-looking state.
    """
```

`RunState` holds node states, produced artifact hashes, open approvals, budget
consumption, and cycle counters. Checkpoints cache a fold at a sequence number
and are **disposable** — deleting every checkpoint must change nothing but
startup time, and a test asserts exactly that.

### 3.2 Workflow graph

```python
# kernel/graph.py


class JoinPolicy(StrEnum):
    ALL = "all"
    ANY = "any"
    QUORUM = "quorum"


class NodeSpec(BaseModel, frozen=True):
    id: str                                  # narrow alphabet; reaches paths and logs
    kind: NodeKind                           # AGENT | TOOL | GATE | BARRIER | TERMINAL
    handler: str | None                      # agent or tool name, resolved by the runner
    join: JoinPolicy = JoinPolicy.ALL
    quorum: int | None = None
    entry_gates: tuple[str, ...] = ()
    exit_gates: tuple[str, ...] = ()
    produces: str | None = None              # artifact kind, for lineage
    retry: RetryPolicy = RetryPolicy()
    timeout_seconds: float = 300.0
    cycle_budget: int | None = None          # required on at least one node of a cycle
    on_exhausted: Exhausted = Exhausted.SAFE_STOP
    requires_approval: bool = False


class WorkflowGraph(BaseModel, frozen=True):
    name: str
    nodes: tuple[NodeSpec, ...]
    edges: tuple[Edge, ...]                  # each carries a closed-enum condition
    entry: tuple[str, ...]                   # declared, never inferred - see below

    def validate_graph(self) -> None:
        """Reachability from `entry`, terminal reachability, quorum sanity, and:
        every cycle must have a cycle_budget on at least one node. An unbounded
        cycle is rejected at load time, not discovered at 3am."""
```

**Entry points are declared, not inferred.** Inferring them from in-degree looks
tidy and breaks exactly where it matters: the moment a node joins a cycle - which
every repair loop and clarification cycle creates - it acquires an incoming edge
and stops looking like a start node. Declaring `entry` also lets a workflow have
several (`REQ_ANALYSIS` alongside `CODEBASE_INDEX`) without that being an
accident of edge layout. This was changed during implementation: the inferred
version rejected a legal bounded repair loop as having "no entry node".

Graphs are authored as YAML in `ases/workflows/` and validated on load. The
`DECOMPOSE` node proposes a **subgraph at runtime**; `with_subgraph` re-validates
the whole graph before returning it, so a proposal that would create a dangling
edge, an unreachable node or an unbounded cycle is rejected at admission and
never executes.

### 3.3 Agents

```python
# agents/base.py


class AgentResult(BaseModel, Generic[TOut]):
    artifact: TOut
    confidence: float  # 0..1, consulted by the exit gate
    rationale: str
    citations: tuple[Citation, ...]  # file:line, required of the critic
    usage: Usage  # tokens, cost, model, prompt_version
    # NOTE: there is deliberately no `next_node`, `status` or `decision` field.
    # Adding one would let an agent steer the graph. tests/invariants guards this.


class Agent(Protocol[TIn, TOut]):
    name: ClassVar[str]
    input_model: ClassVar[type[TIn]]
    output_model: ClassVar[type[TOut]]
    capabilities: ClassVar[CapabilityManifest]  # tools, writable paths, budget
    model_needs: ClassVar[ModelNeeds]  # capabilities, never a model id

    async def run(self, ctx: AgentContext, inp: TIn) -> AgentResult[TOut]: ...
```

`AgentContext` exposes a **scoped retriever**, not the run history. An agent
asks for what it needs (`ctx.artifact(ArtifactRef(...))`) and receives summaries
otherwise. This is the mitigation for late-stage drift: without it, by the time
the release agent runs, its prompt is an unreadable accumulation of everything
that came before.

### 3.4 LLM boundary

```python
# providers/base.py


class LLMProvider(Protocol):
    async def complete(self, req: CompletionRequest) -> CompletionResult: ...


# providers/router.py


class ModelNeeds(BaseModel, frozen=True):
    reasoning: Literal["low", "medium", "high"]
    context: Literal["small", "large"]
    structured_output: bool = True
    latency: Literal["interactive", "batch"] = "batch"


class ModelRouter:
    def resolve(self, needs: ModelNeeds) -> ModelSpec: ...
```

Agents declare *needs*; the router resolves a model. Two consequences worth the
indirection: cost tiering becomes a single-file policy decision rather than
scattered constants, and — because the cassette key is
`sha256(prompt_version + rendered_prompt + output_schema)` and excludes the model
id — swapping models does not invalidate recorded cassettes.

Four modes, selected by `ASES_LLM_MODE`: `replay` (default, offline,
deterministic), `record`, `live`, `mock`.

### 3.5 Tools

```python
# kernel/tools/classification.py


class SideEffect(StrEnum):
    NONE = "none"  # pure; freely retryable
    SANDBOX = "sandbox"  # confined to the worktree; rollback is free and total
    EXTERNAL = "external"  # escapes the sandbox; needs an ordered compensator


@dataclass(frozen=True)
class ToolSpec:
    name: str
    idempotent: bool
    side_effect: SideEffect
    compensator: str | None
    requires_approval: bool
    timeout_s: float
    writable_paths: tuple[PurePosixPath, ...]
```

```python
# kernel/tools/registry.py


class ToolRegistry:
    def invoke(self, name: str, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """Unknown name -> DENY. Unknown action on a known tool -> DENY.
        Write outside writable_paths -> DENY. Every outcome is an event."""
```

There is deliberately **no registered tool for arbitrary SQL or arbitrary shell**.
An agent that wants one has no path, because absence plus deny-by-default is a
stronger control than a blocklist that must anticipate every spelling.

### 3.6 Gates and policy

```python
# kernel/gates.py
class GateVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"


class Gate(Protocol):
    async def evaluate(self, ctx: GateContext) -> GateResult: ...


# kernel/policy.py
class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyEngine:
    def evaluate(self, action: Action, ctx: PolicyContext) -> PolicyEvaluation: ...
```

`ESCALATE` is not a failure — it routes to the human approval queue with the
diff, the rationale and the lineage trail attached. A reviewer is handed
evidence, not a yes/no prompt.

**Approval binding.** An `Approval` records the `artifact_hash` it was granted
against. When re-planning changes that artifact, the approval is *revoked*, not
inherited. This is the governance-critical half of re-planning and the thing
most implementations quietly get wrong.

### 3.7 Validation and the Stage 2 seam

```python
# validation/pipelines/base.py
class ValidationPipeline(Protocol):
    async def validate(self, ws: Workspace, art: Artifact) -> ValidationReport: ...

    # DotnetPipeline: build -warnaserror, format --verify, test+coverage, vuln scan
    # PythonPipeline: ruff, mypy, pytest        (orchestrator's own CI)


# codebase/base.py   <- built in Stage 1; implementations are Stage 2
class CodebaseAnalyzer(Protocol):
    async def index(self, root: Path) -> CodebaseIndex: ...
    async def impact(self, idx: CodebaseIndex, change: ChangeIntent) -> ImpactReport: ...
```

The graph depends on `CodebaseAnalyzer`, never on Roslyn. Stage 2 supplies
`RoslynAnalyzer` (JSON over stdio to `tools/RoslynIndexer/`) and plugs in.

---

## 4. Node lifecycle

```
                    ┌──────────────────────────── STALE ◄── upstream artifact changed
                    ▼                                          │
PENDING ─► READY ─► ENTRY_GATE ─► RUNNING ─► VALIDATING ─► EXIT_GATE ─► SUCCEEDED
             │          │            │            │             │
             │       BLOCKED      RETRYING     REPAIRING   AWAITING_APPROVAL
             │          │            │                          │
             │          │        FALLBACK                    REJECTED
             │          │            │                          │
             └──► CANCELLED      FAILED ─► COMPENSATING ─► ROLLED_BACK
                                    │
                                 HALTED   (safe-stop: budget, cycle or policy exhausted)
```

Every transition is an event. The audit trail is not a side effect of execution;
it *is* execution.

---

## 5. Recovery — which mechanism, and when

Selected by the failed tool's `SideEffect` class, not by exception type:

| Condition | Action |
|---|---|
| Tool `idempotent`, attempt < limit | `RETRYING`, exponential backoff |
| Tool **not** `idempotent` | **No retry.** Straight to fallback or compensation |
| `SideEffect.SANDBOX` | Discard the worktree. Total, free, always correct |
| `SideEffect.EXTERNAL` | Invoke each completed tool's compensator in reverse order |
| Validation failed, repair budget remains | `REPAIRING` with diagnostics fed back verbatim |
| Repair budget exhausted | Fallback: degraded artifact + mandatory human flag |
| Budget / cycle / policy exhausted | `HALTED` — safe-stop with a legible report, never a hang |

The distinction in rows 3 and 4 was a genuine defect in an earlier design that
treated rollback as one mechanism. Sandbox rollback is total and free; external
compensation is partial and ordered. Conflating them produces a rollback that
silently does not roll back.

---

## 6. Runtime sequence — a greenfield run

```
ases run greenfield --requirement workloads/url-shortener/REQUIREMENTS.md#run1
```

| # | Step | Events emitted |
|---|---|---|
| 1 | CLI creates the run; graph loaded and validated | `RUN_CREATED`, `RUN_STARTED` |
| 2 | Scheduler folds state, finds ready nodes | `NODE_READY` |
| 3 | `REQ_ANALYSIS` entry gate: budget, inputs, capabilities | `NODE_ENTRY_GATE` |
| 4 | Router resolves model; provider called | `LLM_REQUESTED`, `LLM_COMPLETED` |
| 5 | Output validated against `RequirementSpec` | `ARTIFACT_PRODUCED`, `ARTIFACT_VALIDATED` |
| 6 | Exit gate escalates — Gate 1 is a human gate | `APPROVAL_REQUESTED` |
| 7 | Operator approves in the dashboard | `APPROVAL_GRANTED` (carries `artifact_hash`) |
| 8 | `ARCH_DESIGN` → Gate 2 | … `APPROVAL_GRANTED` |
| 9 | `SCAFFOLD` materialises the solution, pins packages, **freezes interfaces** | `TOOL_INVOKED` ×n, `NODE_SUCCEEDED` |
| 10 | `DECOMPOSE` proposes a subgraph; admitted after checks | `SUBGRAPH_PROPOSED`, `SUBGRAPH_ADMITTED` |
| 11 | N `IMPL_TASK_*` dispatched **in parallel**; each exits on `dotnet build` | `NODE_STARTED` ×N interleaved |
| 12 | Barrier joins (`ALL`) | `NODE_READY` for `TEST_GEN` |
| 13 | `TEST_RUN` fails; diagnostics fed to `REPAIR`; cycle budget 3 | `NODE_FAILED`, `NODE_RETRY_SCHEDULED` |
| 14 | `CODE_REVIEW` ∥ `SEC_SCAN` ∥ `DOCS_GEN`, joining at `QUALITY_GATE` | parallel |
| 15 | Conformance oracle runs at `QUALITY_GATE` | `ARTIFACT_VALIDATED` |
| 16 | `RELEASE_READINESS` → Gate 3 | `APPROVAL_REQUESTED` |
| 17 | `FINAL_SUMMARY` folds the log into a report | `RUN_COMPLETED` |

Step 7 is worth noting: the approval carries the artifact hash. If the operator
later amends that requirement, `replan` marks descendants `STALE`, computes the
minimal re-execution set, and emits `APPROVAL_REVOKED` for Gate 2 — because
Gate 2 was approved against a design derived from different content.

---

## 7. Concurrency

Single-process `asyncio`. The scheduler holds one `TaskGroup`; each ready node
becomes a task; completion re-folds state and re-evaluates readiness.

Deliberately **not** distributed. A work queue with multiple workers would buy
throughput this prototype does not need, at the cost of distributed-consensus
questions on the event chain — and the hash chain is the thing whose correctness
the entire audit story depends on. Single-writer keeps `prev_hash` trivially
correct.

Parallelism is bounded by a semaphore (`ASES_MAX_PARALLEL_NODES`) so a wide
fan-out cannot exhaust the provider rate limit or the `dotnet build` capacity of
the machine.

Cancellation is cooperative: a `CancelToken` per node, checked at tool
boundaries. Safe-stop cancels in-flight nodes, runs compensators for completed
external effects, writes a checkpoint, and emits `RUN_HALTED`.

---

## 8. Context and lineage

```
Artifact(hash) ──produced_by──► Node(id, attempt)
      │                              │
      └──derived_from──► Artifact(hash) ...    ├── agent, model, prompt_version
                                               └── tool invocations
```

Artifacts are **content-addressed**: the id *is* the sha256 of the canonical
content. Two consequences fall out for free — identical content is stored once,
and "has this changed?" is a string comparison rather than a diff.

The lineage DAG answers both governance questions directly: *why does this line
of code exist* (walk back to the requirement sentence) and *what is affected if
this changes* (walk forward to find `STALE` descendants). `replan` is a forward
walk; the dashboard's inspector is a backward walk.

Lineage is built in Phase 1, not Phase 7, because re-planning depends on it.

---

## 9. Observability

Nothing is instrumented twice. Metrics are **derived by folding the event log**,
which means they cannot drift from what actually happened:

| Metric | Derivation |
|---|---|
| Success rate | `NODE_SUCCEEDED` / terminal node events, per node type |
| Retry frequency | `NODE_RETRY_SCHEDULED` per run |
| Rollback / fallback rate | `TOOL_COMPENSATED`, `NODE_FALLBACK_TAKEN` |
| **MTTR (node)** | first `NODE_FAILED` → next `NODE_SUCCEEDED`, same `node_id` |
| **MTTR (run)** | run-level equivalent; both reported, since the brief is ambiguous |
| Stage / E2E latency | `created_at` deltas, p50 and p95 across runs |
| Human intervention rate | `APPROVAL_REQUESTED` / node count |
| Gate rejection rate | `APPROVAL_REJECTED` / `APPROVAL_REQUESTED` — the autonomy index |
| First-pass validation rate | `ARTIFACT_VALIDATED` without a preceding `ARTIFACT_REJECTED` |
| Token / cost | summed from `LLM_COMPLETED` |

Correlation identity is `(run_id, node_id, attempt, trace_id)` on every log
line, event and span. OpenTelemetry spans go to
`runs/<run_id>/traces.otlp.jsonl` via a file exporter — no collector, no
Grafana, per the locked decision.

The dashboard (FastAPI + SSE) streams the event log directly. Because state is a
fold, the browser runs the *same* fold over the streamed events that the server
runs over the stored ones — there is no second, drifting representation of run
state to keep in sync.

---

## 10. Directory map

| Path | Responsibility |
|---|---|
| `src/orchestrator/ases/kernel/` | Deterministic control plane. No LLM imports, ever |
| `…/kernel/store/` | `PostgresEventStore`, `JsonlEventStore` |
| `…/kernel/tools/` | Deny-by-default registry, tool classification, `dotnet` tools |
| `…/agents/` | LLM workers. Stateless; one artifact each |
| `…/context/` | Content-addressed store, lineage DAG, scoped retriever |
| `…/providers/` | Provider protocol, Anthropic/cassette/mock, model router, versioned prompts |
| `…/codebase/` | `CodebaseAnalyzer` protocol (Stage 1); Roslyn client (Stage 2) |
| `…/validation/` | L1–L4; `.NET` and Python pipelines |
| `…/sandbox/` | Git worktree workspaces, writable-path enforcement |
| `…/observability/` | Correlation, structured logging, OTel file export |
| `…/interfaces/` | Typer CLI, FastAPI + SSE dashboard |
| `…/contracts/` | Pydantic artifact models — the shared vocabulary |
| `…/workflows/` | Workflow graphs as YAML, one per requirement class |
| `…/policies/` | Guardrail rule packs as YAML |
| `…/migrations/` | Alembic, control schema only |
| `tests/invariants/` | One test per architectural rule; each attempts a violation |
| `src/url-shortener/` | **Agent-generated.** Empty by design |
| `workloads/url-shortener/` | Requirements, contract, conformance oracle, frozen baseline |
| `tools/RoslynIndexer/` | Stage 2 .NET analyzer |
| `infra/` | Compose: Postgres + Redis only |
| `runs/` | Exported logs, artifacts, traces. Reproducible, not source |

---

## 11. Extension points

| To add | Do this | Touches the kernel? |
|---|---|---|
| A new agent | Implement `Agent`, declare capabilities and model needs, add a node to a workflow YAML | No |
| A new requirement class | Add a workflow YAML | No |
| A new guardrail | Add a rule to a policy pack | No |
| A new tool | Register a `ToolSpec` with its side-effect class and compensator | No |
| A new language target | Implement `ValidationPipeline` | No |
| Brownfield (Stage 2) | Implement `CodebaseAnalyzer` | No |
| A new event type | `EventType` + a fold case | **Yes** — deliberately the only one |

That last row is the design working as intended: everything that varies is a
plug-in, and the one thing that is hard to change is the thing that must stay
correct.

---

## 12. Decisions and their costs

| Decision | Gains | Costs |
|---|---|---|
| Engine owns control flow | Governance is enforceable; kernel is testable without an LLM | Less emergent autonomy; graph shape is authored |
| Event log as source of truth | Audit, lineage, metrics, replay, resume all fall out of one mechanism | Every state question is a fold; a naive implementation would be slow (mitigated by checkpoints) |
| Content-addressed artifacts | Change detection is a string compare; approvals bind to content | Large artifacts stored whole, not diffed |
| Single-process asyncio | Hash chain is trivially correct; simple to reason about | No horizontal scale |
| Cassette replay by default | Offline, reproducible, free, CI-friendly | Cassettes must be refreshed when prompts change |
| Deny-by-default tools | Whole classes of misbehaviour have no path, rather than being blocked case by case | Every legitimate capability must be registered explicitly |
| One Postgres, two schemas | Simple; integrity isolation via grants | No availability isolation (bounded by role-level timeouts) |
| Agents never touch the real repo | Rollback is structurally correct | An extra promotion step |
