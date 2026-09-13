"""`kernel.tools.fs` handlers, exercised directly (not only through the
registry) so the handler's own defence-in-depth path check is proven too."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import ToolContext
from ases.kernel.tools.fs import READ_FILE, WRITE_FILE, SideEffect


def _ctx(cwd: Path) -> ToolContext:
    return ToolContext(cwd=PurePosixPath(cwd.as_posix()), run_id="run-1")


async def test_write_then_read_round_trips(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    write_result = await WRITE_FILE.handler({"path": "a/b.txt", "content": "hello"}, ctx)
    assert write_result.ok is True

    read_result = await READ_FILE.handler({"path": "a/b.txt"}, ctx)
    assert read_result.ok is True
    assert read_result.output["content"] == "hello"


async def test_write_creates_intermediate_directories(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": "deep/nested/dir/file.txt", "content": "x"}, ctx)
    assert result.ok is True
    assert (tmp_path / "deep" / "nested" / "dir" / "file.txt").read_text(encoding="utf-8") == "x"


async def test_reading_a_nonexistent_file_fails_cleanly(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await READ_FILE.handler({"path": "missing.txt"}, ctx)
    assert result.ok is False
    assert "no such file" in (result.error or "")


async def test_write_rejects_a_path_traversal_attempt(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": "../escape.txt", "content": "x"}, ctx)
    assert result.ok is False
    assert "escapes sandbox" in (result.error or "")
    assert not (tmp_path.parent / "escape.txt").exists()


async def test_write_rejects_an_absolute_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    absolute = tmp_path.joinpath("outside.txt").resolve().as_posix()
    result = await WRITE_FILE.handler({"path": absolute, "content": "x"}, ctx)
    assert result.ok is False
    assert "escapes sandbox" in (result.error or "")


async def test_write_rejects_a_windows_backslash_absolute_path(tmp_path: Path) -> None:
    """Regression test for a real, confirmed exploit found while building
    this: `PurePosixPath` never splits on `\\`, so a backslash-form Windows
    path becomes one opaque path segment. When that segment is later passed
    to `pathlib.Path(*parts)` on a Windows host, it is recognised as a full
    absolute path and joining it onto the sandbox root silently discards the
    root - verified interactively (see `classification.py`'s
    `is_safe_relative_path` docstring) before this test was written. Must be
    rejected regardless of host OS, since args arrive as plain strings and
    the same request could be replayed on any platform."""
    outside = tmp_path.parent / "should-not-exist.txt"
    windows_style_absolute = r"C:\Windows\System32\should-not-exist.txt"
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": windows_style_absolute, "content": "x"}, ctx)
    assert result.ok is False
    assert "escapes sandbox" in (result.error or "")
    assert not outside.exists()


async def test_write_rejects_a_bare_colon_path() -> None:
    """Defence in depth: a colon has no meaning in a POSIX relative path and
    signals a drive letter on Windows, so it is rejected outright rather than
    trusted to be harmless in whatever form it takes."""
    from pathlib import Path as _Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        ctx = _ctx(_Path(tmp))
        result = await WRITE_FILE.handler({"path": "weird:name.txt", "content": "x"}, ctx)
        assert result.ok is False
        assert "escapes sandbox" in (result.error or "")


async def test_read_rejects_a_path_traversal_attempt(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("do not read me", encoding="utf-8")
    try:
        ctx = _ctx(tmp_path)
        result = await READ_FILE.handler({"path": "../secret.txt"}, ctx)
        assert result.ok is False
        assert "escapes sandbox" in (result.error or "")
    finally:
        outside.unlink()


def test_write_file_declares_the_current_directory_as_its_only_writable_root() -> None:
    assert WRITE_FILE.writable_paths == (PurePosixPath("."),)
    assert WRITE_FILE.path_args == ("path",)
    assert WRITE_FILE.side_effect is SideEffect.SANDBOX


def test_read_file_declares_no_side_effect() -> None:
    assert READ_FILE.side_effect is SideEffect.NONE
    assert READ_FILE.writable_paths == ()
