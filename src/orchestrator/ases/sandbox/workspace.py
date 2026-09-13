"""Isolated git-backed workspaces. Agents never write to the real repository.

`create_workspace` is `git worktree add` onto a dedicated branch - the
workspace shares the repository's object database (cheap to create) but is a
physically separate directory an agent can be pointed at, and whose entire
contents can be discarded by `destroy_workspace` with the real repository
completely unaffected. This is what makes sandbox rollback "total and free"
(docs/05 section 5): discarding the worktree *is* the rollback, not an
approximation of one.

`Workspace.tool_cwd` is the one place a real filesystem `Path` is converted
to the `PurePosixPath` `kernel.tools.classification.ToolContext.cwd` expects.
The conversion goes through `.as_posix()` deliberately - on Windows,
`PurePosixPath(str(path))` would keep backslashes as literal (non-separator)
characters and silently produce a broken path; `.as_posix()` normalises to
forward slashes first, which round-trips correctly back through
`Path(str(...))` on every platform. Verified interactively before relying on
it, not assumed.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class SandboxError(RuntimeError):
    """A git worktree operation failed. Never swallowed - a sandbox that
    silently failed to isolate would defeat the entire mechanism."""


@dataclass(frozen=True)
class Workspace:
    """One isolated worktree. `root` is real and platform-native; use
    `tool_cwd` when constructing a `ToolContext`."""

    root: Path
    branch: str
    run_id: str

    @property
    def tool_cwd(self) -> PurePosixPath:
        return PurePosixPath(self.root.as_posix())


async def _run_git(repo_root: Path, *args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(repo_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def create_workspace(
    repo_root: Path,
    run_id: str,
    *,
    worktrees_dir: Path | None = None,
    ref: str = "HEAD",
) -> Workspace:
    """Creates a new worktree at `<worktrees_dir>/<run_id>` on a fresh branch
    `ases/sandbox/<run_id>`, checked out from `ref`.

    `run_id` becomes both the directory name and the branch suffix, so two
    concurrent sandboxes for the same run can never collide, and a stray
    leftover worktree is traceable back to the run that created it.
    """
    target_root = worktrees_dir or (repo_root / ".ases-sandboxes")
    target = target_root / run_id
    if target.exists():
        raise SandboxError(f"sandbox path already exists: {target}")
    target_root.mkdir(parents=True, exist_ok=True)

    branch = f"ases/sandbox/{run_id}"
    returncode, _, stderr = await _run_git(
        repo_root, "worktree", "add", "-b", branch, str(target), ref
    )
    if returncode != 0:
        raise SandboxError(f"git worktree add failed: {stderr.strip()}")
    return Workspace(root=target, branch=branch, run_id=run_id)


async def destroy_workspace(repo_root: Path, workspace: Workspace) -> None:
    """Discards the worktree and its branch. This *is* sandbox rollback -
    total and free, per docs/05 section 5 - not an approximation of one."""
    returncode, _, stderr = await _run_git(
        repo_root, "worktree", "remove", "--force", str(workspace.root)
    )
    if returncode != 0:
        raise SandboxError(f"git worktree remove failed: {stderr.strip()}")

    # Best-effort: the branch has no working tree left to reference it, but a
    # failure to delete it is not a sandbox-isolation failure (nothing agent
    # -writable survives), so it is not raised as a `SandboxError`.
    await _run_git(repo_root, "branch", "-D", workspace.branch)


def new_run_id() -> str:
    """A filesystem- and branch-name-safe identifier, distinct from the
    kernel's own `run_id` (a UUID) - callers may pass either as `run_id` to
    `create_workspace`, but tests that need a disposable one call this."""
    return uuid.uuid4().hex
