"""`sandbox.promotion.promote` - copying a sandbox worktree's generated
content into the real, tracked repository as a reviewable diff.

`Workspace` is constructed directly (not via `create_workspace`) so this
suite needs no real git worktree - only the plain directory-copy behaviour
is under test here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ases.sandbox.promotion import PromotionError, promote
from ases.sandbox.workspace import Workspace


def _workspace(root: Path) -> Workspace:
    return Workspace(root=root, branch="ases/sandbox/test", run_id="test")


def test_promote_copies_generated_files_into_the_destination(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    repo_root = tmp_path / "repo"
    (sandbox_root / "src/url-shortener/src/UrlShortener.Core").mkdir(parents=True)
    (sandbox_root / "src/url-shortener/src/UrlShortener.Core/ShortUrl.cs").write_text(
        "public class ShortUrl { }"
    )
    (repo_root / "src/url-shortener").mkdir(parents=True)

    written = promote(_workspace(sandbox_root), repo_root)

    assert Path("src/url-shortener/src/UrlShortener.Core/ShortUrl.cs") in written
    destination = repo_root / "src/url-shortener/src/UrlShortener.Core/ShortUrl.cs"
    assert destination.read_text() == "public class ShortUrl { }"


def test_promote_overwrites_an_existing_destination_file(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    repo_root = tmp_path / "repo"
    (sandbox_root / "src/url-shortener").mkdir(parents=True)
    (sandbox_root / "src/url-shortener/README.md").write_text("# Real content")
    (repo_root / "src/url-shortener").mkdir(parents=True)
    (repo_root / "src/url-shortener/README.md").write_text("This directory is intentionally empty.")

    promote(_workspace(sandbox_root), repo_root)

    assert (repo_root / "src/url-shortener/README.md").read_text() == "# Real content"


def test_promote_never_deletes_a_destination_file_the_sandbox_does_not_have(
    tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "sandbox"
    repo_root = tmp_path / "repo"
    (sandbox_root / "src/url-shortener").mkdir(parents=True)
    (sandbox_root / "src/url-shortener/new_file.cs").write_text("new")
    (repo_root / "src/url-shortener").mkdir(parents=True)
    (repo_root / "src/url-shortener/.gitkeep").write_text("")

    promote(_workspace(sandbox_root), repo_root)

    assert (repo_root / "src/url-shortener/.gitkeep").exists()
    assert (repo_root / "src/url-shortener/new_file.cs").exists()


def test_promote_raises_if_the_sandbox_has_nothing_at_the_relative_path(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    repo_root = tmp_path / "repo"

    with pytest.raises(PromotionError):
        promote(_workspace(sandbox_root), repo_root)


def test_promote_defaults_to_the_url_shortener_relative_path(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "src/url-shortener").mkdir(parents=True)
    (sandbox_root / "src/url-shortener/x.cs").write_text("x")
    repo_root = tmp_path / "repo"

    written = promote(_workspace(sandbox_root), repo_root)

    assert written == (Path("src/url-shortener/x.cs"),)


def test_promote_supports_a_custom_relative_path(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "other").mkdir(parents=True)
    (sandbox_root / "other/x.cs").write_text("x")
    repo_root = tmp_path / "repo"

    written = promote(_workspace(sandbox_root), repo_root, relative_path="other")

    assert written == (Path("other/x.cs"),)


def test_promote_never_copies_bin_or_obj_build_output(tmp_path: Path) -> None:
    """`dotnet build`/`dotnet test` populate every project's `bin/`/`obj/`
    with compiled DLLs and intermediate MSBuild state - regenerable, often
    binary, and not something this repo's `.gitignore` excludes for a
    generated `src/url-shortener/` tree. Promoting it would turn the
    "reviewable diff" the module docstring promises into binary noise."""
    sandbox_root = tmp_path / "sandbox"
    repo_root = tmp_path / "repo"
    project = sandbox_root / "src/url-shortener/UrlShortener.Api"
    (project / "bin/Debug/net10.0").mkdir(parents=True)
    (project / "bin/Debug/net10.0/UrlShortener.Api.dll").write_bytes(b"\x00binary")
    (project / "obj/Debug/net10.0").mkdir(parents=True)
    (project / "obj/Debug/net10.0/UrlShortener.Api.AssemblyInfo.cs").write_text("// generated")
    (project / "Program.cs").write_text("// real source")
    (repo_root / "src/url-shortener").mkdir(parents=True)

    written = promote(_workspace(sandbox_root), repo_root)

    assert written == (Path("src/url-shortener/UrlShortener.Api/Program.cs"),)
    assert not (repo_root / "src/url-shortener/UrlShortener.Api/bin").exists()
    assert not (repo_root / "src/url-shortener/UrlShortener.Api/obj").exists()
