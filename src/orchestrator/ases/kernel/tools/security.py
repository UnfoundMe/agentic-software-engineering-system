"""Secret scanning (docs/02 Phase 3: "Secret scanning of every generated
artifact"; CLAUDE.md section 6 names "Security" as its own tool category).

Registered as a tool, not a hidden internal check - the same "all side
effects go through registered tools" discipline applies to analysis as to
mutation, and it means a workflow gate can require this tool's result exactly
like it requires `dotnet.build`'s, rather than assuming some other layer
remembered to call it.

Pattern-based, not entropy-based: entropy heuristics on arbitrary generated
code produce enough false positives on legitimate GUIDs, hashes and base64
fixtures to be actively unhelpful, and a missed secret here is not the last
line of defense - `.gitignore`, code review, and the human approval gates
around anything destructive all still apply.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ases.kernel.tools.classification import SideEffect, ToolContext, ToolOutcome, ToolSpec


@dataclass(frozen=True)
class SecretFinding:
    rule: str
    line: int
    excerpt: str


@dataclass(frozen=True)
class _Rule:
    name: str
    pattern: re.Pattern[str]


_RULES: tuple[_Rule, ...] = (
    _Rule("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    _Rule("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    _Rule("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    _Rule("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    _Rule("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    _Rule(
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"),
    ),
    _Rule(
        "hardcoded_credential_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)


def scan_text(content: str) -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        for rule in _RULES:
            match = rule.pattern.search(line)
            if match:
                findings.append(SecretFinding(rule=rule.name, line=line_no, excerpt=match.group(0)))
    return tuple(findings)


async def _scan_for_secrets(args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
    findings = scan_text(str(args["content"]))
    return ToolOutcome(
        ok=not findings,
        output={
            "findings": [{"rule": f.rule, "line": f.line, "excerpt": f.excerpt} for f in findings]
        },
        error=(
            None
            if not findings
            else f"{len(findings)} potential secret(s) found: {', '.join(f.rule for f in findings)}"
        ),
    )


SCAN_FOR_SECRETS = ToolSpec(
    name="security.scan_for_secrets",
    handler=_scan_for_secrets,
    idempotent=True,
    side_effect=SideEffect.NONE,
)
