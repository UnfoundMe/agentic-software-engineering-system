# ASES — Delta Analysis of Plan Updates

**Status:** Analysis only — the base plan (`00-IMPLEMENTATION-PLAN.md`) is unchanged pending answers to §5.
**Date:** 2026-09-12
**Purpose:** Assess whether the scope expansion, the 20 plan updates (A), the 11-phase implementation list (B), and the 17 architectural rules (C) can be absorbed into the base plan without architectural conflict.

---

## 1. Headline verdict

**Yes — all of it is absorbable, and none of it threatens the base architecture.**

The two load-bearing decisions in the base plan survive intact and are in fact *reinforced*:

| Base plan thesis | Effect of the updates |
|---|---|
| §2.1 The kernel owns control flow; agents own content | Section C codifies this as six explicit rules (agents never control progression, never call each other, never bypass the policy engine, never write to the real repo, tools only via registry, unknown tool = DENY). This is the same thesis, promoted from design rationale to enforced invariant. |
| §2.2 The event log is the source of truth; state is a fold | Update #4/#5 *strengthen* it — Postgres makes the log durable and queryable, and explicitly demotes checkpoints to a derived optimization, which is exactly the correct relationship. |

Everything else is **additive along existing seams**: a persistence swap, a workload abstraction, a model router, a tool registry, and a second language toolchain. No kernel redesign is implied.

### Classification of the 20 updates

| Class | Items | Notes |
|---|---|---|
| Already in the base plan, now made explicit | 17, 20, 12 | #17 (approval bound to artifact hash) was base §4.2 property 4; #20 and #12 were base §2.1 and §3. Zero cost — tighten wording. |
| Clean additions, no conflict | 4, 5, 6, 7, 8, 9, 10, 16, 18, 19 | Drop into existing modules. Detailed in §2. |
| Additions that widen scope materially | 1, 2, 3, 11, 13, 14, 15 | Introduce a second language (C#/.NET) and a second workload. Detailed in §3. |

### Classification of section C (17 rules)

- **14 rules** restate or sharpen what the base plan already assumed. No change required.
- **3 rules are genuinely new and valuable:**
  1. *Tool execution only through registered tools; unknown tools/actions = DENY.* Turns the capability manifest from an allowlist into a deny-by-default registry. Strictly better.
  2. *Retry only when the action is safely retryable.* Makes retry **tool-aware**, not just node-aware — this is the correct pairing with update #8 (idempotency classification). The base plan's `recovery.py` retried nodes blindly; that was a real latent defect this rule fixes.
  3. *Workload and orchestrator are mutually independent.* Forces the `workloads/` boundary (#19) to be a hard seam rather than a folder convention.

---

## 2. Clean additions — where each one lands

| # | Update | Lands in | Impact |
|---|---|---|---|
| 4,5,6 | Postgres durable event store; RunState = fold; checkpoints derived | `kernel/store/` (new adapter behind the existing `EventLog` interface) | Low. The base plan already treated persistence as an implementation detail behind the fold. Swap JSONL for a `PostgresEventStore`. Append-only enforced at the DB layer (no UPDATE/DELETE grant, `prev_hash` unique chain). |
| 7 | Sandbox rollback vs external-side-effect compensation | `kernel/recovery.py` | Medium, and **an important correction**. The base plan conflated the two. They are different: sandbox rollback is *free and total* (discard the workspace); external compensation is *partial and ordered* (issue a compensating action per completed side effect, in reverse). Two distinct mechanisms, selected by tool class. |
| 8 | Tool idempotency classification + compensation strategy | `kernel/tools/registry.py` (new) | Medium. Every registered tool declares `{idempotent: bool, side_effect: none\|sandbox\|external, compensator: fn\|None, requires_approval: bool}`. This metadata is what `recovery.py` and `policy.py` consult. Clean, and it makes rule C-11 enforceable rather than aspirational. |
| 9,10 | Capability-based `ModelRouter`; agents request capabilities not models | `providers/router.py` (new) | Low. Agents declare e.g. `needs={reasoning: high, structured_output: true, context: large}`; the router resolves to `claude-opus-5` or `claude-sonnet-5`. Base plan had model tiering hardcoded per agent — this is strictly better and makes the cassette key stable across model swaps. |
| 16 | Repeated local replayable runs, not "single run" | §9 assumption text | Low. Follows directly from Postgres. |
| 18 | Production/destructive tools require approval + explicit execution boundary | `kernel/policy.py` + tool registry | Low. Derives from #8 metadata. |
| 19 | `workloads/` boundary | Top-level layout | Low, but **structurally important** — it is what proves the orchestrator is general-purpose rather than hardcoded to one demo. |

---

## 3. Scope-widening additions — honest assessment

Updates #1, #2, #3, #11, #13, #14, #15 introduce **C#/.NET as a first-class target language** alongside Python.

**Cost, concretely:**

| Subsystem | Python-only (base plan) | Now |
|---|---|---|
| Validation | ruff, mypy, pytest | + `dotnet build`, `dotnet test`, `dotnet format`, dependency scan → validation becomes a **strategy per workload**, selected by workload manifest |
| Codebase reasoning | `ast` stdlib (free, in-process) | C# requires **Roslyn** (`Microsoft.CodeAnalysis`), which is a .NET library — not callable in-process from Python. Needs a small out-of-process analyzer CLI shelled out to, or a fallback to regex/heuristic parsing (materially weaker) |
| Sandbox | git worktree + venv | + NuGet restore, build artifacts, longer cycle times |
| CI | one matrix leg | two |

This is the single largest cost in the update set, and it is concentrated almost entirely in **Phase 6 (brownfield reasoning)**. It is affordable *if* the brownfield analyzer targets one language — and unaffordable-in-a-prototype if it must target both well. See §5 Q2.

**Update #14 (first vertical slice = URL Shortener greenfield) is a good call and I endorse it.** The base plan's kernel-first ordering was correct for correctness but weak for demonstrability: it produced nothing observable until Phase 7. A vertical slice at Phase 3 proves the whole loop early. I would keep the base plan's constraint alongside it: the kernel must still be unit-tested with *fake* agents (Phase 1) before any LLM touches it. The two are compatible — Phase 1 stays, Phase 3 becomes the first end-to-end proof.

---

## 4. Conflicts, contradictions and gaps found

### 4.1 Contradictions in the update set (need resolution)

**C1 — Brownfield workload is specified twice, inconsistently.**
Update A#2 says *"Brownfield workload: URL Shortener"*. Phase 0 says *"Brownfield fixture + GROUND_TRUTH.md"* and Phase 10 says *"Brownfield Legacy Bank scenario"*. These are different targets in different languages.

There is a coherent reading that makes A#1–A#3 elegant: **greenfield run builds the .NET URL Shortener core; the brownfield run then enhances that same service** (Phase 4's cache, expiration, rate limiting, analytics = the brownfield change requests). The system's own output becomes its brownfield input, which is a genuinely strong demo. But that reading requires Roslyn and makes Legacy Bank redundant — yet Phase 10 still demands it. Must be settled. → §5 Q2.

**C2 — Who writes the URL Shortener?**
Phase 3 mixes agent components (Requirement Agent, Planner, Sandbox) with product components (*"URL Shortener core APIs, PostgreSQL persistence, Base62, basic redirect flow"*) in one list. If the agents generate the service, those are *run outputs*, not build tasks, and must not appear in my phase plan as work items. If I hand-write it, the greenfield demo is theatre and Feature 5 (Engineering Output Generation) is unevidenced. This is the highest-impact ambiguity in the whole set. → §5 Q1.

**C3 — Offline reproducibility vs. the new infrastructure.**
The base plan's stated reviewer property was "clone, run, get identical results, no API key, no setup." Postgres + Redis + Grafana replaces that with `docker compose up`. That is a reasonable trade, but the *determinism* claim must not be lost with it.
**Recommendation (no decision needed):** keep Postgres as the source of truth, and add `ases export <run_id>` → portable hash-chained JSONL plus `ases replay <file>` that re-folds state with no database. Preserves audit portability and keeps the kernel's fold testable in pure unit tests. Cheap to build, and it is the honest demonstration that state really is a fold.

**C4 — Grafana vs. the custom dashboard.**
Phase 9 adds OpenTelemetry and a Grafana dashboard; the agreed interface decision was a custom local dashboard. These do not overlap: Grafana cannot render a live DAG, an approval queue, or a lineage inspector; a hand-built dashboard should not be reimplementing metric storage and time-series panels. → §5 Q3.

**C5 — Where does the .NET toolchain execute?**
If the orchestrator runs inside a container, that image needs the .NET 9 SDK (~800MB) plus NuGet cache. **Recommendation:** run the orchestrator on the host (Python 3.13 and .NET 9.0.201 are both already installed here) and use Docker Compose *only* for Postgres, Redis and Grafana. Simpler, faster inner loop, and it keeps the sandbox's `dotnet build` fast. Flagging rather than assuming.

**C6 — "URL safety checks" (Phase 4) implies an external network dependency** (e.g. a Safe Browsing API). That breaks offline determinism and introduces an un-mockable external side effect. **Recommendation:** implement as a local denylist plus a pluggable `SafetyProvider` interface, with the remote implementation stubbed. Keeps the seam visible without the dependency.

**C7 — "Background click processing" (Phase 4) needs a queue/worker.** Phase 0 provisions Redis, but Redis is listed under the *workload's* cache. Needs stating explicitly: is Redis (a) workload cache only, (b) workload cache + workload job queue, or (c) also used by the orchestrator? **Recommendation:** (b). The orchestrator stays on Postgres alone — introducing a second datastore into the kernel would violate rule C-17 ("no unnecessary microservices") in spirit and weaken the "event log is the single source of truth" claim.

### 4.2 Gaps — scope items with no implementation step

**G1 — "Test and documentation improvements" is named in scope but has no demo scenario.** Phase 10 demonstrates greenfield, brownfield and ambiguous — three of the four declared scenario classes. **Recommendation:** add a fourth Phase 10 scenario, e.g. *"raise payments-service test coverage and generate missing API docs"*, routed through a **reduced graph** (no ARCH_DESIGN, no Gate 2, no migration policy). That is valuable beyond box-ticking: it proves the orchestrator supports **variable graph shapes per requirement class**, which is direct evidence for the "non-linear" claim.

**G2 — Decision lineage has no explicit step.** The brief requires "preserve cross-stage context and decision lineage." Phase 7 implies it via artifact version detection, and Phase 9 has audit queries, but `context/lineage.py` (the provenance DAG) appears nowhere. It is a prerequisite for Phase 7's dirty propagation, so it must be built in Phase 1 or 2, not discovered in Phase 7.

**G3 — Scoped context retrieval is missing.** `context/retriever.py` from the base plan (§3) does not appear in any phase. Without it, late-stage agents receive accumulated history and drift — this was a named risk in base §8. Belongs in Phase 2.

**G4 — Prompt-injection containment is missing.** Base §4.5 treated repository content entering prompts as untrusted data. Phase 5 covers capability manifests and allowlists but not input containment. Belongs in Phase 5. Relevant because brownfield runs feed attacker-controllable file content into prompts by design.

**G5 — Secret scanning of generated artifacts is missing.** Phase 4 has a .NET dependency scan and Phase 8 has static validation, but the base plan's secret-scan-every-generated-artifact policy rule is unlisted. Belongs in Phase 8.

**G6 — The repository is not yet a git repository,** but Phase 0 requires a CI skeleton and the sandbox design depends on git worktrees. `git init` is a Phase 0 precondition.

### 4.3 Scope reality check

The update set takes the plan from ~35 build items across 10 phases to **~120 items across 11 phases**, and adds a second language toolchain and a full production-grade web service. That is roughly a 3x increase.

This is buildable, but not uniformly. The honest position is to declare a **cut line** up front rather than discover it at Phase 8. Proposed tiering:

- **Tier 1 — must ship (the thesis):** Phases 0, 1, 2, 3, 5, 7 + the Phase 10 demos for parallel/barrier/retry/policy/approval/rollback/safe-stop/re-plan/replay. This alone satisfies the brief's differentiator in full.
- **Tier 2 — strongly wanted (the evidence):** Phase 6 brownfield reasoning (one language), Phase 8 validation, and the custom dashboard.
- **Tier 3 — polish, cut first under pressure:** Phase 4 advanced URL-shortener features, Phase 9 OpenTelemetry/Grafana, the second language analyzer.

→ §5 Q4.

---

## 5. Open questions blocking the merge

**Q1. Who authors the URL Shortener?** (see C2) — agent-generated, hand-written, or a hybrid where I hand-write only the acceptance spec and tests and the agents generate the implementation against them?

**Q2. What does the brownfield analyzer target?** (see C1) — the .NET URL Shortener via Roslyn, the Python Legacy Bank via `ast`, or both?

**Q3. Observability surface** (see C4) — custom dashboard only, Grafana only, or both with a clear division (DAG/approvals/lineage in the custom UI, metrics/traces in Grafana)?

**Q4. Is the §4.3 cut line acceptable**, or should everything be treated as must-ship?

Recommendations already made and needing no decision unless you disagree: C3 (JSONL export + `replay`), C5 (orchestrator on host, Docker for infra only), C6 (local denylist + stubbed provider), C7 (Redis = workload only), G1–G6 (fold into the phases named).
