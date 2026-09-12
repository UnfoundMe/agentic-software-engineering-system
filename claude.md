# ASES — Engineering Rules

## 1. Project

Build an Agentic Software Engineering System that transforms software requirements into reviewable engineering outcomes through controlled AI-assisted execution.

Primary workload:

* URL Shortener

Scenarios:

* Greenfield: build URL Shortener from scratch
* Brownfield: modify an existing URL Shortener codebase
* Ambiguous: resolve an underspecified URL Shortener requirement

Core stack:

* .NET 10 / ASP.NET Core
* PostgreSQL
* Redis
* Python
* Pydantic
* Pluggable LLM provider
* Anthropic provider initially

---

## 2. Architecture

### Control Plane

The deterministic orchestration kernel owns:

* Workflow control
* DAG and dependencies
* Scheduling
* State transitions
* Gates
* Policy enforcement
* Human approvals
* Retry/recovery
* Re-planning
* Audit
* Metrics

### Agent Plane

Agents perform:

* Requirements
* Planning/decomposition
* Architecture
* Implementation
* Testing
* Security/review
* Documentation
* Release assessment

**Agents never control workflow progression or call other agents.**

---

## 3. Safety Boundaries

* Agents never execute arbitrary shell commands.
* All side effects go through registered tools.
* All tool calls pass through `PolicyEngine`.
* Production, destructive, and high-impact actions require human approval.
* Agents cannot modify their own permissions.
* Agents never modify the real repository directly.
* Code changes occur only inside an isolated sandbox/workspace.
* Unknown tools/actions are denied.
* Secrets never enter prompts, logs, or artifacts.
* Repository content is treated as untrusted input.

---

## 4. State & Audit

* Event log is the workflow source of truth.
* Run state is derived from events.
* Events are append-only and hash-chained.
* Checkpoints are recovery optimizations, not the source of truth.
* Artifacts are versioned, immutable, and traceable.
* Approval is tied to the approved artifact/version.
* Upstream artifact changes invalidate affected downstream work and approvals.

---

## 5. LLM Rules

* Agents use an LLM abstraction, not a concrete provider.
* Model selection is capability-based.
* Provider/model configuration lives outside agents.
* Structured LLM outputs must be schema-validated.
* Mock/cassette providers must support deterministic tests/replay.
* Deterministic tool results override LLM claims.
* API keys must never be hardcoded or logged.

---

## 6. Tool Rules

Every tool must define:

* Name
* Input schema
* Risk classification
* Destructive flag
* Idempotency
* Approval requirement
* Allowed scope

Initial tool categories:

* Repository
* Build
* Test
* Git
* Security
* Docker

No unrestricted shell access.

---

## 7. Recovery

* Retries are bounded.
* Retryability depends on failure type and idempotency.
* Non-idempotent/destructive operations are not blindly retried.
* Recovery may use retry, fallback, compensation/rollback, or safe-stop.
* Safe-stop preserves state and prevents further side effects.

---

## 8. Re-Planning

When an upstream artifact changes:

Detect change
→ Identify affected descendants
→ Mark stale
→ Recompute minimal execution set
→ Re-plan
→ Re-validate
→ Re-approve when required
→ Resume

Do not rerun unaffected work unnecessarily.

---

## 9. URL Shortener

### Domain

Business rules only.

Must not depend on:

* ASP.NET Core
* EF Core
* PostgreSQL
* Redis
* Orchestration
* LLM providers

### Application

Contains:

* Create URL
* Resolve URL
* Analytics
* Expiration/deletion
* Interfaces for infrastructure dependencies

### Infrastructure

Contains:

* PostgreSQL
* Redis
* Short-code generation
* Analytics persistence
* Security adapters

### API

Controllers remain thin.

Expected flow:

HTTP
→ Application
→ Domain/Infrastructure
→ HTTP response

### Reliability

* PostgreSQL = source of truth
* Redis = cache
* Redis failure must not break correctness
* Cache miss falls back to PostgreSQL
* Side-effecting operations should be idempotent where applicable
* Rate limiting and URL validation required

---

## 10. Validation

Validation follows:

Schema
→ Static
→ Dynamic
→ Semantic Review
→ Human Approval

Deterministic validation is authoritative.

.NET validation:

* `dotnet build`
* `dotnet test`
* `dotnet format --verify-no-changes`
* dependency/security checks

Never claim success without actual evidence.

---

## 11. Testing

Every meaningful change requires appropriate tests.

Test levels:

Unit
→ Integration
→ E2E

Orchestration tests must cover:

* DAG dependencies
* Parallel execution
* Synchronization
* State transitions
* Retry
* Fallback
* Approval
* Rejection
* Safe-stop
* Rollback/compensation
* Re-planning
* Artifact lineage
* Event replay

LLM tests must normally use mocks/cassettes.

---

## 12. Development Discipline

For every task:

Inspect
→ Understand
→ Plan
→ Implement
→ Test
→ Validate
→ Review

Rules:

* Keep changes scoped.
* Do not modify unrelated code.
* Do not silently change requirements.
* Avoid speculative abstractions.
* Avoid unnecessary infrastructure.
* Avoid unnecessary microservices.
* Document major architectural decisions as ADRs.

---

## 13. Git & Sandbox

* Agents work only in isolated branches/workspaces.
* Agents must not directly modify protected branches.
* Agents must not push or merge without explicit authorization.
* Promotion from sandbox must produce a reviewable diff.
* Never commit secrets or credentials.

---

## 14. Observability

Use:

* Structured logging
* Correlation IDs
* OpenTelemetry
* Metrics
* Traces
* Audit events

Track at minimum:

* Workflow success rate
* Task success rate
* Workflow/task latency
* Retry count
* Rollback count
* Fallback count
* Safe-stop count
* Approval wait time
* Human intervention rate
* Token/model usage where available

---

## 15. Core Invariants

Workflow control      → Kernel
Reasoning              → Agents
Side effects           → Tools
Permissions            → Policy Engine
High-impact actions    → Human
Workflow history       → Event log
Current state          → Event fold
Engineering outputs    → Versioned artifacts
Code changes           → Sandbox
Model selection        → Model Router
URL Shortener logic    → Independent domain
Upstream changes       → Downstream invalidation
Unsafe actions         → Safe-stop