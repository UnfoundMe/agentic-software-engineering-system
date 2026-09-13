"""`kernel.tools.security`: secret scanning (docs/02 Phase 3; CLAUDE.md
section 6's "Security" tool category)."""

from __future__ import annotations

from pathlib import PurePosixPath

from ases.kernel.tools.classification import SideEffect, ToolContext
from ases.kernel.tools.security import SCAN_FOR_SECRETS, scan_text


def _ctx() -> ToolContext:
    return ToolContext(cwd=PurePosixPath("/sandbox"), run_id="run-1")


def test_clean_content_has_no_findings() -> None:
    assert scan_text("def add(a, b):\n    return a + b\n") == ()


def test_detects_an_aws_access_key_id() -> None:
    findings = scan_text("AWS_KEY = AKIAABCDEFGHIJKLMNOP\n")
    assert any(f.rule == "aws_access_key_id" for f in findings)


def test_detects_an_anthropic_api_key() -> None:
    findings = scan_text('key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"\n')
    assert any(f.rule == "anthropic_api_key" for f in findings)


def test_detects_a_github_token() -> None:
    findings = scan_text("token: ghp_" + "a" * 36)
    assert any(f.rule == "github_token" for f in findings)


def test_detects_a_private_key_block() -> None:
    findings = scan_text("-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n")
    assert any(f.rule == "private_key_block" for f in findings)


def test_detects_a_generic_hardcoded_password_assignment() -> None:
    findings = scan_text('password = "hunter2ButLonger"\n')
    assert any(f.rule == "hardcoded_credential_assignment" for f in findings)


def test_reports_the_correct_line_number() -> None:
    findings = scan_text("line one\nline two\nAKIAABCDEFGHIJKLMNOP\n")
    assert findings[0].line == 3


def test_a_short_quoted_string_that_looks_like_a_word_is_not_flagged() -> None:
    """The credential-assignment pattern requires >= 8 characters inside the
    quotes, so ordinary short config values are not false positives."""
    findings = scan_text('name = "short"\n')
    assert findings == ()


async def test_tool_handler_reports_ok_true_even_when_findings_exist() -> None:
    """`ok=True` regardless of findings: a finding is data for the
    `PolicyViolation` artifact (`agents/wiring.py`'s `_scan_artifact`) to
    carry to `agents/release.py`/`gate3`'s human review, not an operation
    failure. `sec_scan` has no `ON_FAILURE` edge in `workflows/greenfield.yaml`
    - `ok=False` here used to mean `kernel.scheduler._finish_agent_node`
    failed the entire run before that artifact was ever emitted, on the very
    first finding, false positive or not."""
    result = await SCAN_FOR_SECRETS.handler({"content": "AKIAABCDEFGHIJKLMNOP"}, _ctx())
    assert result.ok is True
    assert len(result.output["findings"]) == 1
    assert result.output["findings"][0]["rule"] == "aws_access_key_id"


async def test_tool_handler_reports_ok_true_for_clean_content() -> None:
    result = await SCAN_FOR_SECRETS.handler({"content": "print('hello world')"}, _ctx())
    assert result.ok is True
    assert result.output["findings"] == []


def test_scan_for_secrets_is_side_effect_free() -> None:
    assert SCAN_FOR_SECRETS.side_effect is SideEffect.NONE
    assert SCAN_FOR_SECRETS.idempotent is True
