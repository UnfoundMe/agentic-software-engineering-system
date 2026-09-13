"""`kernel.tools.fs` handlers, exercised directly (not only through the
registry) so the handler's own defence-in-depth path check is proven too."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ases.kernel.tools.classification import ToolContext
from ases.kernel.tools.fs import READ_FILE, SHARED_FILE_CONFLICT_PREFIX, WRITE_FILE, SideEffect


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


def _nested_ctx(tmp_path: Path, *tail: str) -> ToolContext:
    """A `ToolContext` whose `cwd` ends in `tail` - simulating the real
    `tool_cwd` an implementer/tester/docs agent writes under
    (`.../src/url-shortener`), to reproduce the doubled-prefix bug a live
    run hit."""
    cwd = tmp_path
    for segment in tail:
        cwd = cwd / segment
    cwd.mkdir(parents=True, exist_ok=True)
    return _ctx(cwd)


async def test_write_rejects_a_path_that_repeats_the_full_cwd_tail(tmp_path: Path) -> None:
    """The exact shape of the live-run bug: `cwd` is already
    `.../src/url-shortener`, and the model wrote
    `"src/url-shortener/Directory.Build.targets"` - which resolved one level
    too deep for any project to ever find, silently, since nothing about
    that path was otherwise invalid."""
    ctx = _nested_ctx(tmp_path, "src", "url-shortener")
    result = await WRITE_FILE.handler(
        {"path": "src/url-shortener/Directory.Build.targets", "content": "x"}, ctx
    )
    assert result.ok is False
    assert "leading segment" in (result.error or "")
    assert not (Path(ctx.cwd) / "src" / "url-shortener").exists()


async def test_write_rejects_a_path_that_repeats_only_the_last_cwd_segment(
    tmp_path: Path,
) -> None:
    ctx = _nested_ctx(tmp_path, "src", "url-shortener")
    result = await WRITE_FILE.handler({"path": "url-shortener/x.cs", "content": "x"}, ctx)
    assert result.ok is False
    assert "leading segment" in (result.error or "")


async def test_read_also_rejects_a_doubled_cwd_prefix(tmp_path: Path) -> None:
    ctx = _nested_ctx(tmp_path, "src", "url-shortener")
    result = await READ_FILE.handler({"path": "src/url-shortener/x.cs"}, ctx)
    assert result.ok is False
    assert "leading segment" in (result.error or "")


async def test_write_still_accepts_ordinary_project_scoped_paths(tmp_path: Path) -> None:
    """The guard must not false-positive on the normal case: every path an
    implementer actually writes successfully looks like this."""
    ctx = _nested_ctx(tmp_path, "src", "url-shortener")
    result = await WRITE_FILE.handler(
        {"path": "UrlShortener.Api/Program.cs", "content": "// ok"}, ctx
    )
    assert result.ok is True
    assert (Path(ctx.cwd) / "UrlShortener.Api" / "Program.cs").read_text(
        encoding="utf-8"
    ) == "// ok"


async def test_write_still_accepts_a_solution_wide_file_named_without_the_prefix(
    tmp_path: Path,
) -> None:
    """The corrected prompt wording: a solution-wide fix belongs at the
    solution root directly, with no path prefix at all."""
    ctx = _nested_ctx(tmp_path, "src", "url-shortener")
    result = await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "<Project />"}, ctx
    )
    assert result.ok is True
    assert (Path(ctx.cwd) / "Directory.Build.targets").exists()


async def test_write_refuses_a_csproj_file(tmp_path: Path) -> None:
    """Regression test: two live runs had `impl_api` regenerate
    `UrlShortener.Api.csproj` from scratch to add a `ProjectReference` -
    silently reverting `<TargetFramework>` from `net10.0` back to `net8.0`
    both times, identically failing `build_api` until its `cycle_budget`
    ran out. Project/solution files are `ScaffoldAgent`'s alone."""
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler(
        {"path": "UrlShortener.Api/UrlShortener.Api.csproj", "content": "<Project />"}, ctx
    )
    assert result.ok is False
    assert "project/solution files" in (result.error or "")
    assert not (tmp_path / "UrlShortener.Api" / "UrlShortener.Api.csproj").exists()


async def test_write_refuses_a_sln_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": "UrlShortener.sln", "content": "x"}, ctx)
    assert result.ok is False
    assert "project/solution files" in (result.error or "")


async def test_write_refuses_a_slnx_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": "UrlShortener.slnx", "content": "x"}, ctx)
    assert result.ok is False
    assert "project/solution files" in (result.error or "")


async def test_write_refuses_a_csproj_file_regardless_of_case(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await WRITE_FILE.handler({"path": "UrlShortener.Api/App.CSPROJ", "content": "x"}, ctx)
    assert result.ok is False
    assert "project/solution files" in (result.error or "")


async def test_read_is_not_blocked_from_project_files(tmp_path: Path) -> None:
    """The guard is write-only - an implementer may legitimately want to
    read the existing `.csproj` for context before writing a `.cs` file."""
    (tmp_path / "UrlShortener.Api").mkdir()
    (tmp_path / "UrlShortener.Api" / "UrlShortener.Api.csproj").write_text(
        "<Project />", encoding="utf-8"
    )
    ctx = _ctx(tmp_path)
    result = await READ_FILE.handler({"path": "UrlShortener.Api/UrlShortener.Api.csproj"}, ctx)
    assert result.ok is True
    assert result.output["content"] == "<Project />"


async def test_write_refuses_to_overwrite_a_shared_msbuild_file_with_different_content(
    tmp_path: Path,
) -> None:
    """Regression test for the run `b5da55c3-...` race: `repair_domain` and
    `repair_api` both regenerated `Directory.Build.targets` from scratch in
    the same scheduling pass, and a plain `write_text` let the second one
    silently discard the first repair's entire fix. The guard only fires when
    the content actually differs - see the next test for the no-op case."""
    ctx = _ctx(tmp_path)
    first = await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "<Project><!-- fix A --></Project>"}, ctx
    )
    assert first.ok is True

    second = await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "<Project><!-- fix B --></Project>"}, ctx
    )
    assert second.ok is False
    assert (second.error or "").startswith(SHARED_FILE_CONFLICT_PREFIX)
    assert "fix A" in (second.error or "")  # the current content is handed back for reconciliation
    # the first writer's content must survive untouched
    assert "fix A" in (tmp_path / "Directory.Build.targets").read_text(encoding="utf-8")


async def test_write_treats_an_identical_rewrite_of_a_shared_file_as_a_no_op(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    content = "<Project><!-- fix A --></Project>"
    first = await WRITE_FILE.handler({"path": "Directory.Build.targets", "content": content}, ctx)
    second = await WRITE_FILE.handler({"path": "Directory.Build.targets", "content": content}, ctx)
    assert first.ok is True
    assert second.ok is True


async def test_write_succeeds_with_different_content_when_expected_content_matches(
    tmp_path: Path,
) -> None:
    """The compare-and-swap escape hatch: a caller that actually read the
    file first (`agents/implementer.py` does, via `fs.read_file`, before
    ever asking the model to write) and passes back what it read is making
    an informed, legitimate change - allowed even though the content
    genuinely differs, unlike the conflict case above where nothing was
    read first."""
    ctx = _ctx(tmp_path)
    await WRITE_FILE.handler({"path": "Directory.Build.targets", "content": "A"}, ctx)

    result = await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "B", "expected_content": "A"}, ctx
    )

    assert result.ok is True
    assert (tmp_path / "Directory.Build.targets").read_text(encoding="utf-8") == "B"


async def test_write_still_conflicts_when_expected_content_is_stale(tmp_path: Path) -> None:
    """`expected_content` must match what is *actually* on disk right now,
    not merely be present - a caller reasoning from a stale read (someone
    else wrote in between) is exactly the race this guard exists to catch."""
    ctx = _ctx(tmp_path)
    await WRITE_FILE.handler({"path": "Directory.Build.targets", "content": "A"}, ctx)
    await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "B", "expected_content": "A"}, ctx
    )

    result = await WRITE_FILE.handler(
        {"path": "Directory.Build.targets", "content": "C", "expected_content": "A"}, ctx
    )

    assert result.ok is False
    assert (result.error or "").startswith(SHARED_FILE_CONFLICT_PREFIX)
    assert (tmp_path / "Directory.Build.targets").read_text(encoding="utf-8") == "B"


async def test_write_refuses_to_overwrite_directory_build_props_too(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await WRITE_FILE.handler({"path": "Directory.Build.props", "content": "<Project />"}, ctx)
    result = await WRITE_FILE.handler(
        {"path": "Directory.Build.props", "content": "<Project><X/></Project>"}, ctx
    )
    assert result.ok is False
    assert (result.error or "").startswith(SHARED_FILE_CONFLICT_PREFIX)


async def test_ordinary_files_are_never_subject_to_the_shared_file_guard(tmp_path: Path) -> None:
    """The guard must only ever fire for the two named MSBuild filenames -
    every ordinary per-file implementer write (hundreds of `.cs` files across
    a real run) must keep overwriting freely."""
    ctx = _ctx(tmp_path)
    await WRITE_FILE.handler({"path": "ShortUrl.cs", "content": "v1"}, ctx)
    result = await WRITE_FILE.handler({"path": "ShortUrl.cs", "content": "v2"}, ctx)
    assert result.ok is True
    assert (tmp_path / "ShortUrl.cs").read_text(encoding="utf-8") == "v2"


def test_write_file_declares_the_current_directory_as_its_only_writable_root() -> None:
    assert WRITE_FILE.writable_paths == (PurePosixPath("."),)
    assert WRITE_FILE.path_args == ("path",)
    assert WRITE_FILE.side_effect is SideEffect.SANDBOX


def test_read_file_declares_no_side_effect() -> None:
    assert READ_FILE.side_effect is SideEffect.NONE
    assert READ_FILE.writable_paths == ()
