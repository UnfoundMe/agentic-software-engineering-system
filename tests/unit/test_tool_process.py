"""`kernel.tools.process.run_argv` - the shared "run this argv, capture
output, turn it into a `ToolOutcome`" helper every `dotnet`/`git`/migration
tool is built on (see that module's docstring).

Covers the specific defect a live run surfaced: a failed `ToolOutcome.error`
used to be only `` `dotnet build -warnaserror` exited 1 `` - the exit code,
with the actual compiler/NuGet diagnostic (captured fine in `.output`)
silently dropped. `agents/implementer.py`'s `repair*` positions read exactly
this `.error` string back via `ContextRetriever.last_error_of` as their whole
picture of what broke - a repair agent given only an exit code cannot
possibly do better than guessing, and a run's `27a721ce-...` sandbox is the
recorded proof it didn't: two `repair_api` attempts rewrote unrelated `.cs`
files while the actual failure (`UrlShortener.Api.csproj` targeting `net8.0`
against `UrlShortener.Application`/`Infrastructure`'s `net10.0`, a `NU1201`/
`NU1603` restore error) went untouched, and `build_api`'s `cycle_budget: 2`
ran out.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import ToolContext
from ases.kernel.tools.process import default_runner, run_argv


def _ctx(cwd: str = "/sandbox") -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd), run_id="run-1")


class _FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        return self.returncode, self.stdout, self.stderr


async def test_a_successful_command_has_no_error() -> None:
    outcome = await run_argv(_FakeRunner(0, stdout="ok"), ["dotnet", "build"], _ctx())
    assert outcome.ok is True
    assert outcome.error is None


async def test_a_failed_command_with_no_output_falls_back_to_the_bare_exit_summary() -> None:
    outcome = await run_argv(_FakeRunner(1), ["dotnet", "build"], _ctx())
    assert outcome.ok is False
    assert outcome.error == "`dotnet build` exited 1"


async def test_a_failed_command_s_diagnostic_is_folded_into_error_not_just_the_exit_code() -> None:
    """The exact shape of the `27a721ce-...` failure: a NuGet/MSBuild
    diagnostic on stdout, nothing on stderr."""
    diagnostic = (
        "UrlShortener.Api.csproj : error NU1201: Project UrlShortener.Application "
        "is not compatible with net8.0. Project UrlShortener.Application supports: net10.0"
    )
    outcome = await run_argv(
        _FakeRunner(1, stdout=diagnostic), ["dotnet", "build", "-warnaserror"], _ctx()
    )
    assert outcome.ok is False
    assert outcome.error is not None
    assert "exited 1" in outcome.error
    assert "NU1201" in outcome.error
    assert diagnostic in outcome.error


async def test_stderr_diagnostic_is_folded_in_too() -> None:
    outcome = await run_argv(
        _FakeRunner(1, stderr="CS0103: name does not exist"), ["dotnet", "build"], _ctx()
    )
    assert outcome.error is not None
    assert "CS0103: name does not exist" in outcome.error


async def test_both_streams_are_included_when_both_are_non_empty() -> None:
    outcome = await run_argv(
        _FakeRunner(1, stdout="restoring...", stderr="fatal: something"),
        ["dotnet", "build"],
        _ctx(),
    )
    assert outcome.error is not None
    assert "restoring..." in outcome.error
    assert "fatal: something" in outcome.error


async def test_a_very_long_diagnostic_is_tail_truncated_not_dropped() -> None:
    """The tail is kept: MSBuild's actual error summary comes last, after
    however much restore/log noise preceded it."""
    noise = "x" * 10_000
    outcome = await run_argv(
        _FakeRunner(1, stdout=noise + "REAL ERROR AT THE END"), ["dotnet", "build"], _ctx()
    )
    assert outcome.error is not None
    assert "REAL ERROR AT THE END" in outcome.error
    assert len(outcome.error) < len(noise)
    assert "truncated" in outcome.error


async def test_default_runner_disables_msbuild_node_reuse(tmp_path: Path) -> None:
    """Found live (`60ed0166-...`): `dotnet add package` crashed deep inside
    NuGet.targets writing the dependency-graph spec, immediately after two
    prior `dotnet` invocations against the same project - a documented
    MSBuild node-reuse defect (a worker process kept alive between `dotnet`
    CLI calls for speed, corrupted under rapid sequential invocations).
    `MSBUILDDISABLENODEREUSE=1` forces a fresh process every time. Verified
    against a real subprocess (Python itself, always available) rather than
    a fake runner - the point is what actually reaches the child's
    environment, which a fake cannot observe."""
    returncode, stdout, _ = await default_runner(
        [sys.executable, "-c", "import os; print(os.environ.get('MSBUILDDISABLENODEREUSE', ''))"],
        tmp_path,
    )
    assert returncode == 0
    assert stdout.strip() == "1"


async def test_default_runner_still_inherits_the_rest_of_the_environment(tmp_path: Path) -> None:
    """The fix must add the one variable, not replace the environment -
    losing `PATH` would break every real `dotnet`/`git` invocation outright."""
    returncode, stdout, _ = await default_runner(
        [sys.executable, "-c", "import os; print('PATH' in os.environ)"],
        tmp_path,
    )
    assert returncode == 0
    assert stdout.strip() == "True"


async def test_default_runner_serializes_invocations_sharing_a_cwd(tmp_path: Path) -> None:
    """Found live (`b5da55c3-...`): `build_domain`/`build_api` are dispatched
    together by the scheduler (`kernel.scheduler.Scheduler._step`'s genuine
    `asyncio.gather` concurrency) and both ran `dotnet build` against the
    same solution directory at the same instant - two real MSBuild processes
    racing on the same `obj`/`bin` trees, which produced two divergent,
    non-reproducible failures on the retry. Proven here with a real
    subprocess (Python itself) that records its own [start, end) window, so
    true overlap is directly observable rather than inferred from timing."""
    script = "import time; s=time.monotonic(); time.sleep(0.3); print(s, time.monotonic())"

    (_, stdout_a, _), (_, stdout_b, _) = await asyncio.gather(
        default_runner([sys.executable, "-c", script], tmp_path),
        default_runner([sys.executable, "-c", script], tmp_path),
    )

    start_a, end_a = (float(v) for v in stdout_a.split())
    start_b, end_b = (float(v) for v in stdout_b.split())
    assert end_a <= start_b or end_b <= start_a, "the two windows overlapped"


async def test_default_runner_does_not_serialize_across_different_cwds(tmp_path: Path) -> None:
    """The lock is keyed per working directory, not global - two unrelated
    sandboxes' subprocesses must still run concurrently, or every run in the
    process would be serialized against every other run."""
    cwd_a, cwd_b = tmp_path / "a", tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    # A longer sleep and a generous margin below 2x it: under real CPU
    # contention (the full suite spawning many subprocesses at once) a
    # 0.3s/0.5s pair was observed to occasionally cross the threshold even
    # when genuinely concurrent - widened once that flake was seen, rather
    # than left to fail intermittently in CI.
    script = "import time; time.sleep(0.6)"

    started = time.monotonic()
    await asyncio.gather(
        default_runner([sys.executable, "-c", script], cwd_a),
        default_runner([sys.executable, "-c", script], cwd_b),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0  # serialized would take >= 1.2s; concurrent, ~0.6s


async def test_output_still_carries_the_untouched_raw_streams() -> None:
    """`.error` is a human/LLM-readable summary; `.output` remains the exact,
    unmodified stdout/stderr for anything that needs it verbatim."""
    outcome = await run_argv(
        _FakeRunner(1, stdout="a" * 10_000, stderr="b"), ["dotnet", "build"], _ctx()
    )
    assert outcome.output["stdout"] == "a" * 10_000
    assert outcome.output["stderr"] == "b"
