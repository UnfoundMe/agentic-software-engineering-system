# Validation Guide — Phase 0 & Phase 1

**Status:** current as of commit implementing the Postgres store, Alembic,
bootstrap and the `ases` CLI.
**Purpose:** a from-scratch, no-trust-required walkthrough so you can
independently confirm everything claimed as "done" in docs/02's Phase 0 and
Phase 1 actually works, without taking the test suite's word for it.

Every command below was actually run against a live Postgres 16 container
during development; none of this is speculative.

---

## 0. Prerequisites

- Docker Desktop running
- `uv` installed
- **Windows only:** if you already have a native PostgreSQL service, it may
  already own port 5432. Check before you start:

  ```powershell
  Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
  Get-Process -Id <OwningProcess>
  ```

  If the process is literally named `postgres` (not `com.docker.backend` or
  `wslrelay`), Docker's port mapping will silently lose to it. Set
  `POSTGRES_PORT=5433` in `.env` before continuing - see
  [`infra/README.md`](../infra/README.md) for the full explanation. This is a
  genuine, non-obvious environment conflict found and fixed during this
  project's own development, not a hypothetical.

---

## 1. Bring up infrastructure from nothing

```bash
git clone <this repo> && cd agentic-software-engineering-system
cp .env.example .env          # generate your own passwords; defaults are placeholders
uv sync --all-extras
./scripts/dev-up.sh
```

**Expect:** ends with `==> Ready.` and no errors. This alone proves:

- Docker Compose brings up Postgres 16 and Redis 7, both passing their
  healthchecks
- `ases db bootstrap` creates roles, schemas and grants with zero manual SQL
- `ases db upgrade` applies the control-schema migration

**Cross-check idempotency** (the literal Phase 0 exit criterion, docs/04
section 1.7):

```bash
./scripts/dev-up.sh
```

Run it again. Expect the identical `==> Ready.` outcome - no errors, nothing
duplicated, nothing recreated.

---

## 2. Inspect what bootstrap actually created

```bash
docker exec -e PGPASSWORD=<value of POSTGRES_SUPERUSER_PASSWORD in .env> \
  ases-postgres psql -h localhost -U ases_su -d ases -c "\du" -c "\dn"
```

**Expect:**

```
 Role name   |  Attributes
-------------+---------------------------------
 ases_app    |
 ases_control|
 ases_su     | Superuser, Create role, Create DB, ...
 workload_app| 10 connections

    Name      |     Owner
---------------+---------------
 control       | ases_control
 public        | pg_database_owner
 workload_test | workload_app
```

Three roles, two application schemas, `workload_app` capped at 10 connections
(docs/04 section 1.2's availability-isolation limits).

```bash
docker exec -e PGPASSWORD=<value> ases-postgres psql -h localhost -U ases_su -d ases \
  -c "\dt control.*" -c "\dv control.*"
```

**Expect:** table `control.events` (plus Alembic's own `alembic_version`),
and four views: `runs`, `artifacts`, `approvals`, `policy_violations`.
`lineage` is deliberately absent - see the design note at the top of
`src/orchestrator/ases/migrations/versions/8b0a48dce088_*.py` for why (the
short version: it lives in `context/lineage.py`, operating on the Python
fold, so a second SQL implementation of the same graph walk was not built).

---

## 3. Prove the security boundaries yourself, by hand

These are the two guarantees the entire database design rests on. Don't take
the automated tests' word for it - run them yourself.

### 3a. `ases_app` cannot mutate the event log

```bash
PW=<value of ASES_APP_PASSWORD in .env>
docker exec -e PGPASSWORD="$PW" ases-postgres psql -h localhost -U ases_app -d ases \
  -c "UPDATE control.events SET hash='x';"
```

**Expect:** `ERROR: permission denied for table events`. The grant was never
given - this fails before the append-only trigger even gets a chance to run.

### 3b. Even the table's *owner* cannot bypass the append-only trigger

```bash
PW=<value of ASES_CONTROL_PASSWORD in .env>
docker exec -e PGPASSWORD="$PW" ases-postgres psql -h localhost -U ases_control -d ases \
  -c "UPDATE control.events SET hash='x' WHERE false;"
```

(The `WHERE false` just avoids needing a real row - the trigger fires before
row matching either way.)

**Expect:** `ERROR: control.events is append-only: UPDATE is not permitted`
— a *different* error than 3a, from the trigger, not a missing grant. This is
what "defence in depth" means concretely: two independent mechanisms, and
this step proves the second one works on its own, not merely that the first
one masked whether the second exists.

### 3c. `workload_app` cannot see `control` at all

```bash
PW=<value of WORKLOAD_APP_PASSWORD in .env>
docker exec -e PGPASSWORD="$PW" ases-postgres psql -h localhost -U workload_app -d ases \
  -c "SELECT * FROM control.events;"
```

**Expect:** `ERROR: permission denied for schema control`. Not "permission
denied for table events" - for schema. `workload_app` cannot even see that
the schema exists.

All three of these are also automated, and re-run on every `pytest -m
integration`: see `tests/integration/test_store_postgres.py::test_ases_app_cannot_update_or_delete_events`
and `::test_workload_app_cannot_see_the_control_schema`, plus
`::test_defence_in_depth_tamper_is_still_detected` for a fourth angle (proving
`verify_chain` catches a tamper even when both defences above are deliberately
disabled by a superuser, simulating "what if someone made a mistake").

---

## 4. Run the automated test suite yourself

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

**Expect:** all clean, "Success: no issues found in 41 source files".

```bash
uv run pytest -q
```

**Expect:** `174 passed, 1 skipped` (the skip is a Phase 4 guard that
activates once the agent plane exists - see
`tests/invariants/test_layering.py`). This excludes integration tests by
default (`pyproject.toml`'s `addopts`), so it needs no live database beyond
what you already started in step 1.

```bash
uv run pytest -m integration -v
```

**Expect:** `9 passed`. Watch the names scroll by - they are not generic
smoke tests:

- `test_concurrent_appends_across_real_connections_do_not_fork_the_chain` -
  ten separate `PostgresEventStore` instances (ten real connections)
  appending to the *same* run_id simultaneously, proving the Postgres-side
  advisory lock, not just Python's in-process lock
- `test_defence_in_depth_tamper_is_still_detected`, `test_ases_app_cannot_update_or_delete_events`,
  `test_workload_app_cannot_see_the_control_schema` - the automated versions
  of section 3 above

---

## 5. Run a real scheduler run against Postgres, then replay it with the database turned off

This is the single strongest end-to-end proof available at this phase - not a
unit test, an actual demonstration.

```bash
uv run python -c "
import asyncio
from uuid import uuid4
from ases.config import settings
from ases.kernel.gates import ApprovalDecision, BudgetEntryGate, BudgetLimits
from ases.kernel.graph import Edge, NodeKind, NodeSpec, WorkflowGraph
from ases.kernel.scheduler import Scheduler, NodeExecutionOutcome
from ases.kernel.store.postgres import PostgresEventStore

class FixedExecutor:
    def __init__(self, outcome): self.outcome = outcome
    async def execute(self, node, state): return self.outcome

async def main():
    graph = WorkflowGraph(
        name='validation-check', entry=('req',),
        nodes=(NodeSpec(id='req', kind=NodeKind.AGENT, handler='req'),
               NodeSpec(id='gate1', kind=NodeKind.GATE, requires_approval=True),
               NodeSpec(id='done', kind=NodeKind.TERMINAL)),
        edges=(Edge(source='req', target='gate1'), Edge(source='gate1', target='done')),
    )
    class Approvals:
        async def decide(self, node, state): return ApprovalDecision(granted=True, actor='you')
    executors = {'req': FixedExecutor(NodeExecutionOutcome(ok=True, artifact_kind='RequirementSpec', artifact_payload={'text': 'hello'}))}
    limits = BudgetLimits(max_tokens=10**9, max_usd=1e9, max_wallclock_seconds=3600)
    store = PostgresEventStore(settings().app_dsn)
    run_id = uuid4()
    state = await Scheduler(graph, store, executors, entry_gate=BudgetEntryGate(limits), approvals=Approvals()).run(run_id)
    print('status:', state.status)
    await store.close()
    print(run_id)

asyncio.run(main())
"
```

Copy the printed `run_id`, then:

```bash
uv run ases export <run_id>
uv run ases replay runs/<run_id>/export.jsonl
```

**Expect:** `chain verified (N events)` followed by `status: completed`.

Now the actual proof:

```bash
docker stop ases-postgres
uv run ases replay runs/<run_id>/export.jsonl
```

**Expect: the identical output**, with Postgres not running at all. This is
docs/02's Phase 1 exit criterion - "`export → replay` yields a byte-identical
fold" - demonstrated, not asserted.

```bash
docker start ases-postgres
```

(to leave your environment as you found it.)

---

## 6. What this does *not* yet prove

Read alongside the above, not instead of it:

- No real LLM agent exists yet (Phase 2 and Phase 4). Every "agent" above is
  a fake executor supplied by the test/demo code, exactly as docs/02's Phase
  1 exit criterion specifies ("the full graph executes end to end with fake
  agents").
- `workflows/greenfield.yaml` loads and validates but is not yet wired to a
  live `Scheduler` run - see `tests/unit/test_workflows.py` for what is
  checked, and `kernel/scheduler.py`'s module docstring for what remains.
- Dynamic subgraph admission (`WorkflowGraph.with_subgraph`) and full
  re-planning (Phase 7) are unwired to the live scheduler, by design at this
  phase - both are called out explicitly in the same docstring.
- Policy packs, the sandbox, and the dashboard do not exist (Phase 3 and
  later).
