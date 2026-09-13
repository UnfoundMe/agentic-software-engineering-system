"""`sandbox.workspace`: real `git worktree` isolation, exercised for real.

This is the one Phase 3 mechanism deliberately **not** tested behind a fake
subprocess runner (unlike `dotnet.py`/`git.py`): sandbox isolation is the
safety-critical property CLAUDE.md section 3 names directly ("Code changes
occur only inside an isolated sandbox/workspace"), and a test that mocked
away git itself would prove nothing about whether isolation actually holds.
Git is guaranteed present (this project *is* a git repository), and every
worktree this suite creates lives under `tmp_path`, with its branch removed
in a `finally` block - a failing assertion must never leave the real
repository holding a stray `ases/sandbox/*` branch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ases.config import REPO_ROOT
from ases.sandbox.workspace import (
    SandboxError,
    Workspace,
    create_workspace,
    destroy_workspace,
    new_run_id,
)


@pytest.fixture
async def workspace(tmp_path: Path) -> AsyncIterator[Workspace]:
    run_id = f"test-{new_run_id()}"
    ws = await create_workspace(REPO_ROOT, run_id, worktrees_dir=tmp_path)
    try:
        yield ws
    finally:
        if ws.root.exists():
            await destroy_workspace(REPO_ROOT, ws)


async def test_create_workspace_checks_out_a_real_working_tree(workspace: Workspace) -> None:
    assert workspace.root.is_dir()
    assert (workspace.root / "README.md").is_file()
    assert (workspace.root / "pyproject.toml").is_file()


async def test_create_workspace_is_on_a_dedicated_branch(workspace: Workspace) -> None:
    assert workspace.branch.startswith("ases/sandbox/")
    assert workspace.run_id in workspace.branch


async def test_writes_inside_the_workspace_never_touch_the_real_repository(
    workspace: Workspace,
) -> None:
    marker = workspace.root / "sandbox-only-file.txt"
    marker.write_text("this must never appear in the real repo", encoding="utf-8")

    assert marker.exists()
    assert not (REPO_ROOT / "sandbox-only-file.txt").exists()


async def test_creating_a_workspace_at_an_existing_path_raises(tmp_path: Path) -> None:
    run_id = f"test-{new_run_id()}"
    first = await create_workspace(REPO_ROOT, run_id, worktrees_dir=tmp_path)
    try:
        with pytest.raises(SandboxError):
            await create_workspace(REPO_ROOT, run_id, worktrees_dir=tmp_path)
    finally:
        await destroy_workspace(REPO_ROOT, first)


async def test_destroy_workspace_removes_the_directory_entirely(tmp_path: Path) -> None:
    run_id = f"test-{new_run_id()}"
    ws = await create_workspace(REPO_ROOT, run_id, worktrees_dir=tmp_path)
    assert ws.root.exists()

    await destroy_workspace(REPO_ROOT, ws)
    assert not ws.root.exists()


async def test_destroy_workspace_removes_the_branch(tmp_path: Path) -> None:
    run_id = f"test-{new_run_id()}"
    ws = await create_workspace(REPO_ROOT, run_id, worktrees_dir=tmp_path)
    await destroy_workspace(REPO_ROOT, ws)

    proc_returncode, _, _ = await _git(REPO_ROOT, "rev-parse", "--verify", ws.branch)
    assert proc_returncode != 0  # the branch ref no longer resolves


async def test_tool_cwd_round_trips_through_a_real_platform_path(workspace: Workspace) -> None:
    """The exact property `kernel.tools.classification.ToolContext.cwd`
    depends on: converting the real `Workspace.root` to `PurePosixPath` and
    back via `Path(str(...))` must land on the same real directory,
    verified interactively before this was relied upon anywhere - see
    `sandbox/workspace.py`'s module docstring."""
    reconstructed = Path(str(workspace.tool_cwd))
    assert reconstructed == workspace.root
    assert reconstructed.is_dir()  # noqa: ASYNC240 - a single stat() call, not worth anyio.Path here


async def _git(cwd: Path, *args: str) -> tuple[int, str, str]:
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(), stderr.decode()
