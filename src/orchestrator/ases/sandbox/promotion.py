"""Promotes agent-generated content from a sandbox worktree into the real,
tracked repository - the step CLAUDE.md section 13 requires ("Promotion from
sandbox must produce a reviewable diff") and `src/url-shortener/README.md`
documents by name ("content is promoted into this directory only after Gate
3 (release approval)").

Deliberately conservative, in two ways:

- It **adds and overwrites** files under `relative_path`; it never deletes
  anything already tracked that this run's sandbox did not touch. A stray
  leftover from a previous run is a problem for `git status` to reveal and a
  human to resolve, not something this function silently cleans up.
- It never runs `git add` or `git commit` itself (CLAUDE.md section 13:
  "Agents must not push or merge without explicit authorization"). The
  result is an uncommitted, reviewable `git diff`/`git status` in the
  caller's own working tree - the "reviewable diff" the rule requires -
  and a human decides whether, and what, to commit.

Callers are expected to gate this on the run having actually reached
`RunStatus.COMPLETED` with the release gate granted - promoting content from
a rejected or failed run would defeat the approval gate this exists
downstream of.

**`bin`/`obj` are never promoted.** `dotnet build`/`dotnet test` (`build_domain`,
`build_api`, `test_run`) populate every project's `bin/` and `obj/` with
compiled DLLs, PDBs and intermediate MSBuild state - regenerable, often
binary, and not excluded by this repository's `.gitignore` (there is no
tracked `src/url-shortener/` build output to have taught it to). Copying
them verbatim would turn "a reviewable diff" into a pile of binary noise a
human is expected to `git add` blindly - exactly what this function's own
docstring says promotion must not produce.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ases.sandbox.workspace import Workspace


class PromotionError(RuntimeError):
    """The sandbox has nothing at `relative_path` to promote."""


#: Directory names never promoted, wherever they appear under `relative_path`
#: - see the module docstring's note on `bin`/`obj`.
_EXCLUDED_DIR_NAMES = frozenset({"bin", "obj"})


def _is_excluded(relative: Path) -> bool:
    return any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1])


def promote(
    workspace: Workspace, repo_root: Path, *, relative_path: str = "src/url-shortener"
) -> tuple[Path, ...]:
    """Copies every file under `<workspace.root>/<relative_path>` into
    `<repo_root>/<relative_path>`, creating destination directories as
    needed and overwriting any file that already exists at that path -
    except anything under a `bin/` or `obj/` directory, which is never
    promoted (see the module docstring).

    Returns the written destination paths, relative to `repo_root`, in
    sorted order - the reviewable change set for the caller to print.
    """
    source_root = workspace.root / relative_path
    if not source_root.is_dir():
        raise PromotionError(f"nothing to promote: {source_root} does not exist in the sandbox")
    destination_root = repo_root / relative_path

    written: list[Path] = []
    for source_file in sorted(source_root.rglob("*")):
        if source_file.is_dir():
            continue
        relative = source_file.relative_to(source_root)
        if _is_excluded(relative):
            continue
        destination_file = destination_root / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        written.append(destination_file.relative_to(repo_root))
    return tuple(written)
