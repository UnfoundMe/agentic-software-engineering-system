# RoslynIndexer — **Stage 2, not yet built**

A .NET 9 console application that will analyse the generated C# workload using
`Microsoft.CodeAnalysis.CSharp` and speak JSON over stdio to the Python
orchestrator.

## Why it is out of process

Roslyn is a .NET library and cannot be hosted inside the Python orchestrator.
The boundary is a versioned, schema-validated JSON contract in both directions,
and the indexer is unit-tested independently in its own CI leg.

## Why it is deferred

Brownfield reasoning is Stage 2 by decision. Stage 1 delivers the greenfield
path and covers both requirement-quality classes (well-defined and ambiguous)
without needing codebase analysis.

The seam is designed now so Stage 2 plugs in rather than retrofits: the
workflow graph depends on the `CodebaseAnalyzer` protocol in
`src/orchestrator/ases/codebase/base.py`, never on Roslyn directly.

## Planned responsibilities

- Symbol table; project and assembly reference graph
- API map — ASP.NET Core attribute routing *and* minimal-API endpoint registration
- Data flow — entity → `DbContext` → migration → endpoint → DTO
- Impact analysis — blast radius from a changed symbol set, ranked by distance

See [`docs/02`](../../docs/02-ORCHESTRATOR-IMPLEMENTATION-PLAN.md) Phase 6.
