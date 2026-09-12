# URL Shortener — agent-generated workload

**This directory is intentionally empty.**

Its contents are produced by the ASES orchestrator at runtime, not written by
hand. That is the locked decision recorded in
[`docs/02`](../../docs/02-ORCHESTRATOR-IMPLEMENTATION-PLAN.md) §0: the workload
is *fully agent-generated from the requirement alone*.

Hand-writing a `.sln`, a `.csproj` or any `.cs` file here would pre-empt the
exact capability the prototype exists to demonstrate.

## What lands here, and when

| Produced by | Node | Output |
|---|---|---|
| Architect agent → `SCAFFOLD` | after Gate 2 | `UrlShortener.sln`, `.csproj` files, pinned `Directory.Packages.props`, frozen interface/DTO signatures |
| Implementer agents | `IMPL_TASK_*` | `src/` — entities, Base62 encoder, EF Core persistence, API surface |
| Test agent | `TEST_GEN` | `tests/` — unit and integration tests |
| Migration pipeline | gated | EF Core migrations, after the generated SQL is classified and human-approved |

Nothing is written here directly. Agents work in an isolated git-backed sandbox
worktree; content is promoted into this directory only after Gate 3
(release approval).

## Expected shape once generated

Recorded as a grading reference only — see
[`docs/03`](../../docs/03-URL-SHORTENER-WORKLOAD-PLAN.md) §3.4. It is **never**
placed in an agent's context. If the agents converge on something materially
different that still satisfies the contract and the conformance oracle, that is
a legitimate outcome and the divergence is recorded, not penalised.

```
UrlShortener.sln
  src/UrlShortener.Api/             endpoints, DI wiring, Problem Details
  src/UrlShortener.Core/            ShortUrl entity, Base62 encoder, abstractions
  src/UrlShortener.Infrastructure/  EF Core DbContext, repository, migrations
  tests/UrlShortener.Tests/         unit + integration
```

## Why the acceptance oracle is not in this directory

The black-box conformance suite lives at
[`workloads/url-shortener/conformance/`](../../workloads/url-shortener/conformance/),
deliberately outside this tree.

The sandbox copies *this* directory for the agents to work in. The oracle must
never be copied with it — the implementation and its tests are both
agent-generated, so an agent that could read the acceptance criteria could
satisfy them without satisfying the requirement. Physical separation is what
keeps the oracle independent.

## Git

Generated content here is git-ignored; only this README and the `.gitkeep`
placeholders are tracked. The reviewed, frozen output of the greenfield run is
committed separately to `workloads/url-shortener/baseline/`, which is the
artifact of record and the input to Stage 2 brownfield work.
