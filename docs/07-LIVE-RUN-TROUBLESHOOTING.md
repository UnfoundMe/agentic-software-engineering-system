# Live Run Troubleshooting Log

**Status:** living document - append a new entry every time a real
(`ASES_LLM_MODE=live`) run surfaces something the mocked/replay test suite
never could.
**Purpose:** everything in this file was found by actually running
`ases run greenfield` against a live model and a real `dotnet`/`git`
toolchain, not by reasoning about the code in the abstract. The mocked
end-to-end test (`tests/unit/test_greenfield_full_e2e.py`) proves the
*orchestration* is wired correctly; it cannot catch a real model's token
usage, a real MSBuild defect, or a real human clicking "reject" - only a
live run can, and every issue below is one it found.

Each entry: **symptom** (what you'd actually see), **root cause**, **fix**
(files + the actual mechanism), and **how to recognize this class of issue
again** if something that looks similar happens on a future run.

---

## 1. `ScaffoldAgent` created every project as a plain class library

**Symptom:** never seen live - found by re-reading the code before the
first live attempt, specifically to answer "what would break if this ran
for real."

**Root cause:** `ctx.invoke_tool("dotnet.new", template="classlib", name=project)`
was hardcoded for every project regardless of role. The API project got no
ASP.NET Core host, no `Program.cs`, no `ControllerBase`/`WebApplication` -
nothing an implementation task's controller code could ever reference.

**Fix:** `ScaffoldAgent._template_for(project)` (`agents/scaffold.py`) infers
`webapi` (with `-controllers`, since the default template is minimal-API-only
otherwise) from a trailing `Api`/`Web`/`WebApi` name segment, `classlib`
otherwise. The prompt now tells the model this convention explicitly rather
than assuming it.

**Recognize it again:** a `CS0246`/"type or namespace not found" error on an
ASP.NET Core base type (`ControllerBase`, `WebApplication`, `[ApiController]`)
in a project that should be the API host.

---

## 2. `pinned_packages` was reported but never installed

**Symptom:** never seen live - same code-review pass as #1.

**Root cause:** `SolutionSkeleton.pinned_packages` existed as an artifact
field the model populated, but no tool call ever ran `dotnet add package`.
Any code depending on EF Core/Npgsql/etc. would fail to compile with "the
type doesn't exist," for the same reason as #1.

**Fix:** added `dotnet.add_package` (`kernel/tools/dotnet.py`);
`ScaffoldAgent` installs every pinned package (parsed as `PackageId/Version`)
into every project. Deliberately solution-wide, not per-project -
`SolutionSkeleton` has no per-project package mapping to route from.

**Recognize it again:** `CS0246` on a type that belongs to a NuGet package
named in the design's `key_decisions` (e.g. `DbContext`, `NpgsqlConnection`).

---

## 3. `dotnet ef` had no `--project`/`--startup-project` targeting

**Symptom:** never seen live - found while implementing #1/#2, reasoning
about what a *second* project in the solution would do to the migration
pipeline.

**Root cause:** `dotnet ef migrations add`/`script`/`database update` all
auto-discover a lone project in the working directory. Once `ScaffoldAgent`
started producing genuine multi-project solutions, `dotnet ef` had no way to
guess which project holds the `DbContext` or which is the runnable host.

**Fix:** `kernel/tools/migrations.py`'s `_ef_targeting_args` appends both
flags when given; `agents/migration.py`'s `ef_targets(projects)` derives them
by naming convention (a project ending `Infrastructure` is `--project`, the
last project in declared dependency order is `--startup-project`) - a
heuristic, not a schema field, disclosed as such in the module docstring.
`agents/wiring.py` reuses the same function for the `migration_apply` tool
node.

**Recognize it again:** `dotnet ef` reporting "more than one DbContext was
found" or "specify which project to use."

---

## 4. Rejecting any gate but `gate1` silently stalled the whole run

**Symptom (live, first real run past `gate2`):** the human rejected `gate2`
with a reason; the CLI printed nothing further and the process exited
looking "stuck," not failed or retried.

**Root cause:** `workflows/greenfield.yaml` only ever had one `on_rejected`
edge (`gate1 -> req`, the clarification cycle). `gate2`/`migration_gate`/
`gate3` had no recovery edge at all, so a rejection correctly prevented
everything downstream from ever running - but nothing was wired to try
again either.

**Fix:** added `on_rejected` edges `gate2 -> arch`, `migration_gate ->
migration`, `gate3 -> release`, each bounded by a `cycle_budget: 3` on the
target node (matching `req`'s existing pattern). Added
`ContextRetriever.rejection_reason_of(gate_node_id)` and wired it into
`requirements`, `architect`, `migration`, and `release` so a retry responds
to *why* it was rejected instead of blindly reproducing the same output.

**A bug caught before it shipped, while building this fix:** adding the
second incoming edge to `arch`/`migration`/`release` without also setting
`join: any` broke the *happy path* - `join: all` (the default) requires
every incoming edge to match simultaneously, which is impossible on the
first pass since the `on_rejected` edge's source hasn't run yet. The
mocked e2e test caught this immediately (`COMPLETED` -> `FAILED`) before it
ever reached a live run. Fixed by adding `join: any` to all three nodes,
mirroring `build_domain`/`test_run`'s existing "first arrival or after
retry" shape.

**Disclosed limit:** `migration`/`release` retries can only change what
those two agents themselves decide (a migration's name/rationale, a release
recommendation) - neither writes code, so a rejection whose real cause is
the underlying implementation cannot be fixed by looping back to them.

**Recognize it again:** a run's `state.status` stays `RUNNING` indefinitely
after a rejection with no further output - check `workflows/greenfield.yaml`
for an `on_rejected` edge from that gate before assuming anything else.

---

## 5. A permanently-stuck run reported `RUNNING`, never `FAILED`

**Symptom (live, same run as #4):** `ases runs list` (once built - see #6)
showed the run as `running` forever, even though nothing could ever make
further progress.

**Root cause:** `Scheduler.run()`'s loop only breaks on `not progressed`
without checking *why* nothing progressed - "waiting on a human who might
still answer" and "genuinely nothing can ever happen again" look identical
to that check.

**Fix:** `Scheduler._awaiting_human_decision()` (`kernel/scheduler.py`)
checks whether any node is `AWAITING_APPROVAL` before accepting quiescence.
If nothing is ready, nothing is awaiting a decision, and the run is not
complete, it now emits `RUN_FAILED` with an explicit reason instead of
leaving `status` stale.

**Recognize it again:** `state.status is RunStatus.RUNNING` days after the
last event with no `AWAITING_APPROVAL` node anywhere in `state.nodes`.

---

## 6. `ases runs list` crashed if any one run's log didn't fold

**Symptom (live):** `ases runs list` raised `InvalidTransitionError: node
'gate1' cannot move pending -> succeeded` and printed nothing at all - not
even the runs that were perfectly fine.

**Root cause:** `tests/integration/test_store_postgres.py` connects to
`settings().app_dsn` - the *same* physical `ases` database (and
`control.events` table) the live CLI reads from, not a dedicated test
database. Its synthetic event sequences (written purely to exercise store
mechanics) are not meant to represent a foldable `RunState`, and `_runs_list`
folded every run returned by `list_runs()` in one list comprehension with no
per-run error handling.

**Fix:** `interfaces/cli.py`'s `_fold_safely()` catches
`(InvalidTransitionError, ValueError, NotImplementedError)` per run; a run
that doesn't fold shows as `unreadable` with the error, and every other run
is unaffected.

**Not fixed, and worth knowing:** the integration test suite still writes
into the same database `ases run` uses. This doc doesn't change that -
`ases runs list` is now merely resilient to it.

**Recognize it again:** an `unreadable` row in `ases runs list`'s output is
almost always leftover test data, not a real orchestrator defect - check
whether the run has a sane number of events and a real `req`/`arch`/...
node lifecycle before investigating further.

---

## 7. `arch` (and any agent) could fail with truncated JSON

**Symptom (live):**
```
node.failed (arch): DesignSpec: repair attempt did not fix the schema error.
first attempt: 1 validation error for DesignSpec
  Invalid JSON: EOF while parsing a string at line 1 column 11392
repair attempt: 1 validation error for DesignSpec
  Invalid JSON: EOF while parsing a string at line 1 column 11546
run.failed: node 'arch' failed with no ON_FAILURE recovery edge: ...
```

**Root cause:** current-generation Claude models (`claude-opus-5`,
`claude-sonnet-5`) run adaptive extended thinking **by default** whenever a
request omits the `thinking` parameter - and thinking tokens are billed and
counted against `max_tokens` exactly like any other output.
`providers/anthropic_provider.py` never sets `thinking`. `ArchitectAgent`
requested `max_tokens=4096`; between reasoning and a genuinely verbose
`DesignSpec` (four layers' worth of ADR-style `key_decisions`), the response
ran out of budget mid-JSON-string. The one bounded repair
(`providers/structured.py`'s `_repair_request`) copies `max_tokens`
unchanged, so it failed the identical way one column later - this was never
actually recoverable at the old ceiling.

**Fix:** raised `max_tokens` on every agent's structured-output call -
4096 -> 8192 (`requirements`, `architect`, `scaffold`, `decompose`,
`reviewer`, `docs`, `release`), 8192 -> 16000 (`implementer`, `tester` - real
C# file contents, the largest payloads), 1024 -> 4096 (`migration` - smallest
schema, but thinking alone can exceed 1024 regardless of output size).
Deliberately did not touch `thinking`/`effort` settings - disabling thinking
on Opus 5 has its own documented failure modes (tool calls leaking into
visible text, `<thinking>` tag leakage); raising the ceiling is the safer,
more certain fix.

**Recognize it again:** any `node.failed` payload containing `Invalid JSON:
EOF while parsing a string` - it is a truncation, not a genuinely malformed
response, and the fix is headroom, not a repair-prompt change. Check
`llm.completed`'s `output_tokens` against the node's own `max_tokens` in
`control.events` to confirm.

---

## 8. Nothing was printed between one gate's approval and the next prompt

**Symptom (live, user-reported):** after approving `gate1`/`gate2`, the
terminal went blank while `arch`/`scaffold` were genuinely working (a real
LLM call, real `dotnet` invocations) in the background - indistinguishable
from a hang.

**Fix:** `interfaces/cli.py`'s `_ProgressExecutor` wraps every real
`NodeExecutor`, printing a line when a node starts and another when it
finishes (with elapsed time and, on failure, the actual error). A
`_heartbeat` background task prints "still working" every 15s for anything
still in flight. Deliberately plain lines, not an animated spinner: nodes
run genuinely concurrently (`asyncio.gather`), and two overlapping spinners
on one terminal would fight each other; separate print lines handle
concurrency for free.

**Recognize it again:** this is purely a CLI presentation concern
(`interfaces/cli.py` only) - it has no effect on scheduling, retries, or any
kernel behavior, so it is not a place to look for orchestration bugs.

---

## 9. `dotnet add package` crashed inside NuGet.targets (MSBuild node reuse)

**Symptom (live, run `60ed0166-...`):**
```
node.failed (scaffold): dotnet.add_package failed: 'Microsoft.EntityFrameworkCore.Relational'
-> 'UrlShortener.Application': `dotnet add UrlShortener.Application package
Microsoft.EntityFrameworkCore.Relational --version 8.0.10` exited 1:
...NuGet.targets(237,5): error MSB4018: ... at Newtonsoft.Json.JsonWriter..cctor() ...
Unable to create dependency graph file for project '...UrlShortener.Application.csproj'.
Cannot add package reference.
```
This happened immediately after two prior `dotnet` invocations (`new`,
`sln add`) against the same project moments earlier.

**Root cause:** a documented MSBuild defect - the .NET SDK keeps a worker
process ("node") alive between CLI invocations for speed, and that process's
internal state can become corrupted under rapid sequential invocations
against the same project. `agents/scaffold.py`'s own call pattern (`new` ->
`sln add` -> `add_package`, back to back, per project) is exactly the shape
that triggers it.

**Fix:** `kernel/tools/process.py`'s `default_runner` now sets
`MSBUILDDISABLENODEREUSE=1` on every real subprocess it spawns (harmless for
`git` - the variable means nothing to it), forcing a fresh MSBuild process
per invocation. Verified against a real spawned subprocess that the variable
actually reaches the child and the rest of the environment (`PATH`, etc.) is
still inherited.

**Recognize it again:** an `MSB4018` error, or any crash whose stack trace
runs through `NuGet.targets` / `NuGet.ProjectModel.DependencyGraphSpec.Save`
/ a `Newtonsoft.Json` static initializer, on a `dotnet` call that follows
closely after other `dotnet` calls in the same project. If it recurs despite
the env var, the next step is a bounded retry on this specifically
`idempotent=True` tool (not attempted here - see "Still open" below).

---

## 10. `dotnet new webapi` and `dotnet new classlib` picked different target frameworks

**Symptom (live, run `27a721ce-...`):** `UrlShortener.Api` (the `webapi`
project) could not reference `UrlShortener.Application`/`Infrastructure`
(the `classlib` projects) - a `NU1201`/`NU1603` restore error ("Project
UrlShortener.Application is not compatible with net8.0. Project
UrlShortener.Application supports: net10.0"). Two `repair_api` attempts
rewrote unrelated `.cs` files while the real cause (a solution-wide
framework mismatch, not anything in a `.cs` file) went untouched, and
`build_api`'s `cycle_budget: 2` ran out.

**Root cause:** different `dotnet new` templates default to different
target frameworks on the same SDK - `webapi` defaults to an LTS release
(`net8.0` on this SDK), `classlib` defaults to the SDK's own version
(`net10.0`). `ScaffoldAgent` never pinned one explicitly, so the two
templates silently disagreed within the same solution.

**Fix:** `kernel/tools/dotnet.py`'s `dotnet_new` handler always passes
`-f net10.0` (`TARGET_FRAMEWORK`), regardless of template. This is CLAUDE.md
section 1's own fixed stack choice (".NET 10 / ASP.NET Core"), not something
detected from the installed SDK.

**A related, separate fix from the same investigation:** `dotnet build`'s
`ToolOutcome.error` used to report only the exit code (`` `dotnet build
-warnaserror` exited 1 ``), dropping the actual NuGet/compiler diagnostic
that was sitting right there in `.output`. A repair agent reading
`ContextRetriever.last_error_of` back as its entire picture of what broke
cannot do better than guess from an exit code alone - which is exactly why
the two `repair_api` attempts above edited the wrong thing.
`kernel/tools/process.py`'s `_failure_message` now folds the actual
diagnostic (tail-truncated, since MSBuild's real error summary comes last)
into `.error`, not just the bare exit code.

**Recognize it again:** `NU1201`/`NU1603` between two projects generated in
the same run is a target-framework mismatch, not a real incompatibility -
check that every `dotnet new` call in the run's event log used the same
`-f` value.

---

## 11. A correct repair fix silently landed one directory too deep and never took effect

**Symptom (live, run `95502a6f-...`):** `build_api`/`build_domain` failed
identically, twice in a row, with the same `NU1903`/`NU1510` errors both
times, until `build_api`'s `cycle_budget: 2` ran out and the run halted:
```
UrlShortener.Application.csproj : error NU1903: Package 'SSH.NET' 2023.0.0
has a known high severity vulnerability, https://github.com/advisories/GHSA-q939-rpr3-3284
UrlShortener.Api.csproj : error NU1510: PackageReference
Microsoft.Extensions.Configuration.Abstractions will not be pruned. This
package is automatically available and does not need to be referenced
explicitly. Remove the PackageReference item.
```

**The original failure was real and reasonable, not a bug:** `NU1903` is
.NET's NuGet audit correctly flagging a known-vulnerable transitive
dependency (`SSH.NET` 2023.0.0, pulled in by Testcontainers, used only by
infrastructure tests); `NU1510` is the SDK's package-pruning feature
correctly noting that two packages the model pinned into `UrlShortener.Api`
already ship inside the ASP.NET Core shared framework it targets. Both are
only fatal because every generated `.csproj` sets
`TreatWarningsAsErrors=true` and `dotnet.build` passes `-warnaserror`.

**`repair_api` actually diagnosed and fixed both correctly, on its first
attempt:** it wrote a `Directory.Build.targets` setting
`NuGetAuditMode=direct` (stop auditing transitive-only advisories) plus a
targeted `NoWarn`, and removed the two redundant `PackageReference` items
from `UrlShortener.Api` specifically (via an
`$(MSBuildProjectName)`-conditioned `ItemGroup`, leaving them intact in
`UrlShortener.Infrastructure`, which genuinely needs them). This is
correct, well-scoped MSBuild - the reasoning was right.

**Root cause of why the fix never took effect:** the file was written to
path `"src/url-shortener/Directory.Build.targets"`. `fs.write_file` resolves
paths against `ctx.cwd`, which for every implementer/tester/docs call is
already `<sandbox>/src/url-shortener` (set once in `interfaces/cli.py` so
`dotnet` tool invocations run from the directory the `.sln` lives in). The
write therefore landed at
`<sandbox>/src/url-shortener/src/url-shortener/Directory.Build.targets` -
one level too deep for any project's MSBuild import search (which walks
*up* from each project directory) to ever discover. `fs.write_file` reported
`ok=True` regardless, since nothing about that path was otherwise invalid -
there was no signal anywhere in the event log that the fix had silently
gone nowhere until a human read the sandbox directory by hand and found
`Directory.Build.targets` sitting one level too deep, never at the solution
root the `.sln` and every project folder actually live in.

The proximate cause: `agents/implementer.py`/`tester.py`'s prompts said a
`FileChange.path` is "relative to the sandbox root" - true for every
project-scoped path the model had written successfully so far (e.g.
`"UrlShortener.Api/Program.cs"`, correctly relative to `tool_cwd`), but
false for a *solution-wide* file, where the model reasonably read "sandbox
root" as the actual top of the sandbox tree and re-added the
`src/url-shortener` segment it could see everywhere else in its own context
(project names, the design summary, prior file paths).

**Fix:**
- `agents/implementer.py`, `agents/tester.py`, `agents/docs.py`: prompts now
  say paths are "relative to the solution root - the directory that
  directly contains {projects} and the solution file itself", with an
  explicit "never prefix a path with `src/url-shortener/`" instruction.
  Each bumped `PROMPT_VERSION` (1 -> 2) - wording changes are half the
  cassette key (`providers/prompts/registry.py`'s own docstring), so a
  silent version reuse would have mixed old and new prompts under one
  cassette.
- `kernel/tools/fs.py`: `_write_file`/`_read_file` now refuse (rather than
  silently mis-resolving) any path whose leading segment(s) literally repeat
  `ctx.cwd`'s own trailing segment(s) - the exact doubling shape above -
  via a new `DoublesCwdPrefixError`. This makes a recurrence of this
  mistake (from this model or a future one, regardless of prompt wording)
  surface immediately as a normal tool failure, fed back into the next
  repair attempt through the same `last_error_of` path a compiler error
  already uses - not discovered later by a human reading the sandbox by
  hand.

**Recognize it again:** an implementer/tester/docs-written fix that looks
correct in the event log's `artifact.produced` payload, followed by the
*exact same* build/test failure recurring on the very next attempt with no
new diagnostic content - check whether the fix's file actually exists in
the sandbox at the path the model intended, not just that `node.succeeded`
was emitted for the write.

---

## 12. `ef_targets` would have picked a test project as EF's `--startup-project`

**Symptom:** never seen live - no run has reached `migration_apply` yet
(every run so far halted earlier, at `scaffold`/`build_api`). Found by
re-reading `ef_targets` specifically to ask "what happens once a run
actually gets this far," against two real runs' recorded `SolutionSkeleton`
output.

**Root cause:** `ef_targets` picked `--startup-project` as `projects[-1]`
outright, on the documented assumption that `ScaffoldAgent`'s "in dependency
order" prompt would always put the ASP.NET Core host last. Both real runs'
`SolutionSkeleton.projects` disproved that: run `27a721ce-...` produced
`[Domain, Application, Infrastructure, Api, Tests]` - `Tests` last, not
`Api`, since a test project genuinely depends on everything else it
exercises, making it the *more* correctly dependency-ordered final entry.
`--startup-project UrlShortener.Tests` would have sent `migration_apply`
(`ef.database_update`, no `ON_FAILURE` edge of its own) into a project with
no `Program.cs`/DI composition root and none of `Api`'s connection-string
configuration - failing the run outright the first time it ever reached
this far. Whether this triggers is not fixed by the workflow's shape: the
other real run's skeleton happened to have no `Tests` entry at all, landing
on `Api` last by coincidence, not by any guarantee.

**Fix:** `agents/migration.py`'s `ef_targets` now searches `projects` by the
same `Api`/`Web`/`WebApi` suffix convention `ScaffoldAgent._WEB_HOST_SUFFIXES`
already uses to decide which project gets the `webapi` template, falling
back to `projects[-1]` only when no project name matches.

**Recognize it again:** `dotnet ef` reporting it cannot find a way to build
the startup project, or a migration applying against a project with no
`appsettings.json`/connection string - check `SolutionSkeleton.projects`'
declared order against which project the `--startup-project` flag actually
named.

---

## 13. A secret-scan finding would have failed the whole run instead of reaching the human at gate3

**Symptom:** never seen live - no run has reached `sec_scan` yet. Found by
tracing what actually happens to a `PolicyViolation` finding end to end,
after `agents/release.py`'s own docstring pointed at `sec_scan` as "the
first agent with a genuinely *optional* upstream artifact."

**Root cause:** `security.scan_for_secrets` returned `ok=not findings` - any
match (an AWS key, or the generic `password\s*[:=]\s*"..."` shape, which a
perfectly ordinary `appsettings.Development.json` JWT-signing placeholder
or similar dev-only config value could easily trip) reported the *tool
call itself* as failed. `kernel.scheduler._finish_agent_node` returns as
soon as it emits `NODE_FAILED` for a not-`ok` outcome - *before* the
artifact-emission code that would otherwise turn the finding into the
`PolicyViolation` `agents/wiring.py`'s `_scan_artifact` builds. `sec_scan`
has no `ON_FAILURE` edge in `workflows/greenfield.yaml`, so the very first
finding - false positive or not - would have taken down the entire run with
`RUN_FAILED`, and the `PolicyViolation` artifact would never have been
recorded at all, let alone reached `agents/release.py`/the human at `gate3`
- despite both of those being explicitly, documented-in-code, built to read
exactly this artifact.

**Fix:** `kernel/tools/security.py`'s `_scan_for_secrets` now reports
`ok=True` unconditionally - scanning succeeded whether or not it found
anything; a finding is data for the `PolicyViolation` artifact to carry, not
an operation failure. Detection is unchanged; only the reporting channel
changes, from "crash the run" to "hand it to the review this system already
has for exactly this purpose."

**Recognize it again:** any tool node whose handler sets `ok=False` for a
*finding* rather than a genuine operation failure, feeding into a `kind:
tool` graph node with no `ON_FAILURE` edge, silently discards whatever
artifact `build_artifact` would have produced - check `_finish_agent_node`'s
early `return` on `not outcome.ok` before assuming an artifact this system
builds ever actually gets recorded.

---

## 14. Every real EF Core migration script would have been classified OPAQUE

**Symptom:** never seen live - no run has reached `migration` yet. Found by
checking `classify_migration` against the actual shape `dotnet ef migrations
script` produces (a transaction wrapper plus EF's own history bookkeeping
row), not only the single-statement snippets the existing unit tests used.

**Root cause, two compounding gaps, either one alone enough to trigger this:**
- `_RISKY_PATTERNS`/`_SAFE_PATTERNS`' `CREATE INDEX` patterns never matched
  `CREATE UNIQUE INDEX` - `UNIQUE` sits between the two keywords a plain
  `CREATE\s+INDEX` regex expects adjacent. `CREATE UNIQUE INDEX` is exactly
  the shape EF Core emits for any unique constraint, which a `ShortCode`
  column needs precisely one of.
- No pattern recognized `START TRANSACTION`/`BEGIN TRANSACTION`/`COMMIT` or
  the `INSERT INTO "__EFMigrationsHistory"` bookkeeping row - EF Core's own
  fixed wrapper around *every* migration script, present regardless of what
  the migration itself changes.

`classify_migration` classifies the whole migration as the *worst* of every
statement in it (`worst_of`), and OPAQUE outranks DESTRUCTIVE. A migration
containing any of the above - which is to say, essentially every real EF
Core migration ever produced - fell through to OPAQUE purely from its own
boilerplate, regardless of what it actually changed. OPAQUE is "denied by
policy, no approval path at all" (`agents/migration.py`'s
`OpaqueMigrationRejectedError`, docs/04 section 4.3) - a deliberate,
disclosed stance for genuinely unrecognizable SQL, but this made it the
*universal* outcome instead: `migration` has no `ON_FAILURE` edge, so the
very first real migration this pipeline ever generated would have failed
the entire run, unconditionally.

**Fix:** `kernel/tools/migrations.py`'s `_RISKY_PATTERNS`/`_SAFE_PATTERNS`
now accept an optional `UNIQUE` between `CREATE` and `INDEX` (RISKY without
`CONCURRENTLY`, SAFE with it, exactly like the plain-index case already
worked), and `_SAFE_PATTERNS` gained three new entries for EF Core's fixed
wrapper (`START`/`BEGIN TRANSACTION`, `COMMIT`,
`INSERT INTO "__EFMigrationsHistory"`). A realistic full migration script
(transaction wrapper, `CREATE TABLE`, `CREATE UNIQUE INDEX`, the history
insert) now classifies as RISKY - the correct class for a migration whose
only structurally-significant statement is a blocking index build - not
OPAQUE.

**Recognize it again:** any `migration` node failure whose error names
`OpaqueMigrationRejectedError`/"denied by policy" - read the actual SQL in
the `MigrationPlan` artifact (or the `ef.migrations_script` tool output) and
check which statement `classify_migration` could not recognize; it is very
likely EF Core's own fixed boilerplate rather than anything the migration
itself does, unless a genuinely exotic statement shape (a raw
`migrationBuilder.Sql(...)` block using syntax outside docs/04 section
4.3's table) is actually present.

---

## 15. The net8.0/net10.0 mismatch (#10) recurred - from a different agent this time

**Symptom (live, run `efd88cc9-...`):** the exact same failure shape as #10
- `build_api`/`build_domain` failing twice identically on `NU1201`
(`UrlShortener.Application`/`Infrastructure` "not compatible with net8.0
... supports: net10.0") plus an unrelated `NU1903` vulnerable-package
warning, until `build_api`'s `cycle_budget: 2` ran out - **despite** #10's
own fix (`kernel/tools/dotnet.py`'s `dotnet_new` passing `-f net10.0`)
still being in place and working correctly at `scaffold` time.

**Root cause: `impl_api` reintroduced the mismatch #10 had already
eliminated at the source.** Reading `impl_api`'s actual `CodePatch` for this
run: it "modified" `UrlShortener.Api.csproj` - to add a `ProjectReference`
it believed the focus area ("thin controllers, DI wiring") needed - by
regenerating the *entire* file from scratch, including
`<TargetFramework>net8.0</TargetFramework>`, silently overwriting the
`net10.0` `ScaffoldAgent` had correctly created moments earlier. #10's fix
only ever controlled what `scaffold` produces; nothing stopped a later
agent from clobbering it. The `ProjectReference` the model was trying to
add was never actually necessary: SDK-style project references are
transitive, so `ScaffoldAgent`'s own linear chain
(`Api -> Infrastructure -> Application -> Domain`) already gives `Api`
compile-time access to `Application`'s public types with no direct
reference required - the model solved a problem that did not exist, and in
doing so broke something that did.

**Fix, two layers this time given the repeat:**
- `agents/implementer.py`'s prompt (`PROMPT_VERSION` 2 -> 3) now states
  outright: never write or modify a `.csproj`/`.sln`/`.slnx` file; project
  structure is scaffold's alone, and project references are transitive, so
  the reference this agent kept trying to add was never needed.
- `kernel/tools/fs.py`'s `_write_file` now refuses any `.csproj`/`.sln`/
  `.slnx` path outright, as a hard backstop independent of prompt wording -
  no current or future `fs.write_file` caller has a legitimate reason to
  target one; `ScaffoldAgent` materializes and wires all of them
  exclusively through its own `dotnet.*` tool calls, never through
  `fs.write_file`.

**Recognize it again:** the same `NU1201`/`net8.0` mismatch as #10, but
check *which* project's `.csproj` actually contains `net8.0` this time
before assuming it's a scaffold-time regression - the fix here addressed a
specific agent (`impl_api`) overwriting a specific file; a different agent
finding a different way to touch project structure would need the same
treatment, not a re-application of #10's fix (which was never wrong, only
incomplete against a threat from a different direction).

---

## 16. NuGet package *versions* were the same unenforced-invariant problem as #10/#15, one layer down

**Symptom:** the proximate cause of both #14's `SSH.NET` finding and #15's
`Microsoft.Extensions.Caching.Memory` finding - `NU1903` (a pinned version
with a known CVE) on two separate live runs, plus a related but unseen
`NU1603` risk (a stale `net8.0`-era version pinned into a `net10.0`
solution, silently resolved upward and then rejected as a downgrade warning
under `-warnaserror`). Not found as a standalone crash - found by asking,
after fixing #10/#15's target-framework mismatches, "is target framework
the only thing here that was never a tracked, enforced invariant?" Package
version was the same shape of gap: `scaffold.plan`'s prompt asked the model
for `"PackageId/Version"` and `dotnet.add_package` installed whatever
version it wrote, verbatim, with no check that it was current, unaffected
by a known advisory, or even compatible with the `net10.0` target every
project was just pinned to.

**Root cause:** the same one this whole file keeps finding in different
places - a value only an LLM was choosing, with no deterministic check or
fallback, standing in for something that has an actual right answer once
the rest of the solution is fixed. Here, "the latest stable release
compatible with `net10.0`" is that right answer, and `dotnet add package
<id>` (no `--version`) resolves exactly that - deterministically, and newer
by construction, so less likely to carry an already-known CVE than whatever
version happened to be common in the model's training data (both findings
so far were `8.0.x`-era releases).

**Fix:** `agents/scaffold.py`'s prompt (`PROMPT_VERSION` 1 -> 2) now asks
for a bare package id, explicitly with no version; `ScaffoldAgent._package_id`
(renamed from `_parse_package`) strips a version the model includes anyway
rather than trusting the prompt change alone, and `_materialize_project`
never passes `version=` to `dotnet.add_package` at all - the tool's
`--version` support (used correctly elsewhere, e.g. a future caller pinning
a deliberate, known-good release) stays; only this one caller stops using
it. Mirrors #10/#15's structural fix exactly: stop asking the model for a
value it cannot reliably get right, and let deterministic tooling supply it
instead.

**Recognize it again:** `NU1903` (a specific package/version pair with a
known advisory) or `NU1603`/`NU1605` (a version conflict/downgrade) naming a
package in `SolutionSkeleton.pinned_packages` - check whether the failing
version was ever actually requested via `--version`, or whether NuGet's own
resolution produced it; the former means this fix regressed or a new
`dotnet.add_package` caller reintroduced pinning, the latter is a genuinely
new package-graph conflict this fix does not cover.

---

## 17. `build_domain`/`build_api` were the same redundant, racing whole-solution build

**Symptom (live, run `b5da55c3-...`):** `build_domain` and `build_api` failed
with byte-for-byte identical `NU1510` diagnostics, both naming
`UrlShortener.Api.csproj`, immediately after `impl_domain`/`impl_api`
finished. `repair_domain` and `repair_api` both ran, both wrote a freshly
regenerated `Directory.Build.targets` to fix it (both diagnosed correctly),
`node.succeeded` for both. The retry then failed *again*, this time with two
divergent, seemingly unrelated errors - an `MSB4018` file-handle crash inside
`Microsoft.AspNetCore.Mvc.Testing.targets` for `build_api`, a `CS0234`
"'UseCases' does not exist" for `build_domain` - until `build_api`'s
`cycle_budget: 2` ran out and the run halted.

**Root cause, three compounding facts, not one:**
- `kernel/tools/dotnet.py`'s `dotnet_build` took no project argument - it was
  a bare `dotnet build -warnaserror`, which resolves whatever `.sln` sits in
  `ctx.cwd`. `build_domain` and `build_api` were never "build this project"
  vs. "build that project"; they were the *same command* run twice.
- `agents/wiring.py` gave both graph nodes the exact same `ToolNodeExecutor`
  instance (`executors["dotnet_build"]`), so there was no way for them to
  differ even if the tool had supported scoping.
- `kernel.scheduler.Scheduler._step` dispatches independently-ready nodes as
  genuinely concurrent subprocesses (`asyncio.gather`, by design - see issue
  #8's note on why). `build_domain` and `build_api` become ready in the same
  pass, so two real `dotnet build` processes ran against the identical
  solution directory's `obj`/`bin` trees at the same instant - a distinct
  concurrency hazard from #9's MSBuild node-reuse defect (that one is about
  *sequential* invocations corrupting one reused worker process; this one is
  about two processes racing on the same files at the same time), which is
  what produced the divergent, non-reproducible second-round errors.

**A second race, layered on top:** since both builds failed identically, both
`repair_domain` and `repair_api` ran concurrently too, and both decided the
fix was `Directory.Build.targets` - the same shared, solution-wide file.
`kernel/tools/fs.py`'s `_write_file` was a plain `write_text` with no
read-before-write of any kind, so whichever of the two writes happened to
execute second (asyncio is single-threaded and cooperative, so this is
deterministic *per run* but not predictable in advance) silently discarded
the other repair's entire fix. Nothing in the event log distinguished this
from a normal, correct write - both reported `node.succeeded`.

**Fix, three layers matching the three facts above:**
- `kernel/tools/dotnet.py`: `dotnet.build`/`dotnet.test` now accept an
  optional `project` argument, scoping the command instead of always
  resolving the whole solution.
- `agents/wiring.py`: `build_domain`/`build_api` are now two separate
  `ToolNodeExecutor`s (`dotnet_build_domain`/`dotnet_build_api`), each with
  its own `build_args` deriving its own project from `SolutionSkeleton` by
  the same naming convention `agents/migration.py`'s `ef_targets` already
  uses for the API project, plus a new `_domain_project` for the domain one.
- `kernel/tools/process.py`: `default_runner` now serializes subprocess
  invocations that share a working directory behind a per-cwd `asyncio.Lock`
  - this is the actual fix for the concurrent-MSBuild hazard, general to any
  `dotnet`/`git` tool call now or in the future, not tied to these two nodes
  specifically. Project scoping above reduces how much of the solution two
  nodes redundantly rebuild; the lock is what makes doing so at the same
  time safe regardless.
- `kernel/tools/fs.py`: `_write_file` now refuses to overwrite one of a
  small, named set of shared MSBuild files (`Directory.Build.props`/
  `.targets`) with content that differs from what is already on disk,
  returning the current content in the refusal (`SHARED_FILE_CONFLICT_PREFIX`).
  `agents/implementer.py`'s `run` reacts to exactly that refusal with one
  bounded reconciliation retry - feeding the actual current content back
  through the same `prior_error_section` mechanism already used for a
  `dotnet build` failure - so a losing writer gets a chance to merge instead
  of silently disappearing.
- `kernel/scheduler.py`: `_has_recovery_edge` now also counts an `ALWAYS`
  edge as recovery, not only `ON_FAILURE`. Found while reasoning about the
  fix above: `repair`/`repair_domain`/`repair_api` each have exactly one
  outgoing edge, `condition: always`, back to the node they repair - never
  `on_failure`. Before this, a repair agent that itself failed (which the
  reconciliation retry above can now legitimately do, on a second conflict)
  would have been reported as having no recovery edge and immediately failed
  the whole run - even though the `always` edge had, moments earlier in the
  same call, already rescheduled its target for another attempt. Never
  triggered by any live run so far (no repair agent had failed outright
  before this fix existed to make it possible); caught by reasoning about
  what the new failure path needed before it shipped, the same way issues
  #12-#14 were found.

**Recognize it again:** two `kind: tool` nodes gating different
implementation tasks failing with byte-for-byte identical diagnostics is
always a "these are actually the same build" signal, not two independent
findings - check whether the underlying tool call was ever actually scoped
to a project before treating the two failures as unrelated. A `node.failed`
whose error starts with `shared file conflict:` is this fix's guard working
as intended, not a new defect - it means a solution-wide file was about to be
silently clobbered and was not.

**Verified on the next live run (`237d6873-...`):** `build_api` failed with
a correctly project-scoped `` `dotnet build UrlShortener.Api -warnaserror` ``
(not the whole solution), `build_domain` succeeded cleanly on its own build
with no trace of `build_api`'s failure, and exactly one repair
(`repair_api`) fired - not two racing ones. The fix holds under a real model
and a real toolchain, not just the mocked test suite.

---

## 18. `migration` crashed the entire run on the first live call it ever made

**Symptom (live, run `237d6873-...`, the first run ever to reach
`migration`):**
```
node.failed (migration): Anthropic returned 400: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "output_config.format.schema: For 'object' type, 'additionalProperties' must be explicitly set to false"}, ...}
run.failed: node 'migration' failed with no ON_FAILURE recovery edge: ...
```
This is a request-construction failure, not a model output problem - the
call never got as far as Anthropic generating any text at all, and `migration`
has no `ON_FAILURE` recovery edge (docs/04 section 4.1: it isn't meant to
need one, since its only output is a name and a rationale), so the entire
run ended immediately.

**Root cause:** `providers/anthropic_provider.py`'s `_json_schema_output_config`
sends `schema.model_json_schema()` to Anthropic's raw-schema structured-output
mode as-is, which requires `additionalProperties: false` to be *explicitly*
present on every object-typed node - a schema that merely omits the key
(plain JSON Schema's own default, meaning "permissive") is rejected outright.
Pydantic only emits that key when a model sets `extra="forbid"` -
`contracts.base.ArtifactModel` does, so every other agent's output schema
(`RequirementSpec`, `DesignSpec`, `SolutionSkeleton`, `TaskGraph`, `CodePatch`,
...) has always been fine. `agents/migration.py`'s `_MigrationProposal` is
the *only* LLM output schema in the whole agent roster that is not an
`ArtifactModel` - its own docstring says so explicitly, by design, since it
is "the only thing the LLM actually decides" rather than a tracked artifact -
and it never set `extra="forbid"` either, so its schema silently lacked the
key from day one. This is exactly why no earlier run ever surfaced it: every
one of the sixteen prior fixes in this file happened at `scaffold`/`build_*`
or earlier - `migration` had never actually been reached by a live run
before now (see issue #12's and #14's own "never seen live" notes, both
about this same, previously-untested node).

**Fix, two layers - the same shape as issue #16's "stop trusting every
caller to get a shared property right, fix it once at the boundary too"
pattern:**
- `agents/migration.py`: `_MigrationProposal.model_config` now sets
  `extra="forbid"`, matching every other structured-output schema - the
  correct, direct fix for this specific model.
- `providers/anthropic_provider.py`: `_json_schema_output_config` now
  post-processes the generated schema, filling in
  `additionalProperties: false` on the root and on every `$defs` entry that
  doesn't already have it - a backstop at the LLM boundary for *any* current
  or future output schema that makes the same omission, not reliance on
  every model author remembering one config flag. Only fills in what
  pydantic left unset; never overrides an explicit value.

**Recognize it again:** an Anthropic `400` naming
`output_config.format.schema` and `additionalProperties` is always a
schema-construction defect, never a model/prompt problem - check whether the
`output_schema` passed to that call is an `ArtifactModel` subclass (or
otherwise sets `extra="forbid"`) before assuming the backstop above somehow
didn't apply (it should always apply now; if this recurs, the backstop
itself likely regressed, not a new schema-authoring mistake).

**While auditing the rest of `migration` before a follow-up live run** (not
found live, found by re-reading the pipeline end to end against docs/04
section 4.2 before spending more run budget on it):

- `ef.migrations_add`/`ef.migrations_script` were left at `ToolSpec`'s
  default `timeout_s=60.0`. Both trigger a full design-time build under the
  hood before EF Core can do anything (the same cost `dotnet.build`/
  `dotnet.test` were deliberately given `300.0` for) - a cold build could
  plausibly exceed 60s, and `migration` has no `ON_FAILURE` recovery edge, so
  a timeout here would crash the run exactly like any other failure. Fixed:
  both now use `timeout_s=300.0`, matching the precedent already set.

---

## 19. `migration` will fail deterministically the moment it reaches `ef.migrations_add` - no `DbContext` is ever written

**Not yet seen live** - run `237d6873-...` crashed at the LLM-schema bug
(#18) before ever reaching this tool call, so this is a step ahead: found by
checking docs/04 section 4.2's own prerequisite ("1. ENTITY DESIGN -
Implementer agent writes entities **+ DbContext** (sandbox)") against what
`workflows/greenfield.yaml` actually wires, the same way issues #12-#14 were
found before a live run could hit them.

**The gap:** the workflow hard-codes exactly two implementation positions,
`impl_domain` (entities, explicitly "zero infrastructure deps" per its own
focus text) and `impl_api` (controllers) - both disclosed, from the start,
as stand-ins for `DECOMPOSE`'s real dynamic subgraph (`workflows/greenfield.yaml`'s
own header comment, `agents/decompose.py`'s docstring). Nothing implements
`UrlShortener.Infrastructure`, which is where a `DbContext` actually belongs
(docs/03's reference layout, line ~138: `Infrastructure/ EF Core DbContext,
repository impl, migrations` - `Domain` and `Application` are the two layers
that must never depend on EF Core, per `claude.md` section 9 and
`docs/03` section 3.1a). `agents/migration.py`'s own docstring used to claim
the `DbContext` lives in the domain layer - it does not, and that note has
been corrected in place. `dotnet ef migrations add` needs a `DbContext` it
can discover via the startup project's DI container or a parameterless
constructor to do anything at all; with none written anywhere in the
sandbox, it will fail (something like "Unable to create a 'DbContext'... no
application service provider was found... consider adding an
`IDesignTimeDbContextFactory`"), and `migration` has no `ON_FAILURE`
recovery edge - the run ends there, the same shape of failure as #18, one
step later.

**Not fixed here - a real scope decision, not a bug fix.** The correct fix
is a third implementation position (`impl_infrastructure`, with its own
`build_infrastructure`/`repair_infrastructure` pair, mirroring
`impl_domain`/`impl_api`/`build_domain`/`build_api`/`repair_domain`/
`repair_api` exactly) wired into `workflows/greenfield.yaml` and
`agents/wiring.py`, plus deciding whether `MigrationAgent.build_input`
should read that new node's `CodePatch` too (entities alone, from
`impl_domain`, may still be enough context for the LLM's name/rationale -
the SQL itself comes from the compiled `DbContext`, not from what this
agent reads). This is real topology work, not a defect fix, and is exactly
the kind of change CLAUDE.md section 12 wants an explicit decision (and
ideally an ADR) for before it lands - not something to add silently while
chasing a live-run failure.

**Recognize it again:** `dotnet ef` reporting no `DbContext`/no application
service provider/suggesting `IDesignTimeDbContextFactory` is this gap, not a
new defect - check whether an Infrastructure implementation task has been
wired in yet before assuming anything else changed.

---

## Still open - not fixed here, watch for these next

- **No generic retry for `idempotent=True` tools.** Every fix above that
  touches a transient failure (MSBuild node reuse, #9) is a *root-cause*
  fix, not a retry. If a genuinely transient `dotnet`/`ef` failure recurs
  despite it, the next step is a bounded retry keyed off `ToolSpec.idempotent`
  in `ToolRegistry.invoke` (docs/02 Phase 5, not built) - not a blind retry
  at the subprocess level, which would be unsafe for non-idempotent tools
  (`dotnet.new`, `ef.database_update`).
- **Only `gate1`'s original clarification cycle and the three added in #4
  exist.** A rejection whose real cause is inside `impl_domain`/`impl_api`
  has no recovery edge back to an implementation task - only Phase 7
  (re-planning) would let a rejection invalidate and re-run a specific
  upstream artifact rather than a fixed graph position.
- **The migration pipeline still stops at "generate, classify, gate, apply."**
  No real snapshot/down-test cycle, no verification against a live
  `workload_test` Postgres schema (docs/04 section 4.2 steps 7, 9-11).
- **L2 static validation (`dotnet.format_verify`, `dotnet.list_vulnerable`)
  is registered but never invoked by any graph node.** A live run's
  formatting or known-vulnerable-package issues will not be caught.
