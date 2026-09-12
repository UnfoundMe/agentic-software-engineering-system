"""Canonical serialisation and content hashing.

Every hash in the system - the event chain and artifact content-addressing -
goes through this module. Two properties matter, and both are load-bearing:

1. **Canonical.** The same logical value must always produce the same bytes,
   on any platform, in any Python process. Sorted keys, no insignificant
   whitespace, UTF-8, and a fixed representation for datetimes and UUIDs.

2. **Stable across versions.** If this function changes, every previously
   recorded chain fails verification and every approval bound to an artifact
   hash is invalidated. `HASH_ALGORITHM_VERSION` exists so that such a change
   is a deliberate, visible migration rather than a silent one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Final
from uuid import UUID

HASH_ALGORITHM_VERSION: Final[str] = "sha256/v1"

GENESIS_HASH: Final[str] = "0" * 64
"""Predecessor hash of the first event in a run. Not a real digest by design -
it is visibly distinguishable from one, so a genesis event cannot be confused
with a chained event whose predecessor was lost."""


def _encode(value: Any) -> Any:
    """Fixed representation for types json cannot serialise natively.

    Deliberately explicit rather than `str(value)`: relying on repr would make
    the hash sensitive to unrelated library changes.
    """
    if isinstance(value, datetime):
        # Naive datetimes are rejected rather than assumed UTC: guessing a
        # timezone would make the hash depend on where the process ran.
        if value.tzinfo is None:
            raise ValueError("refusing to hash a naive datetime; use timezone-aware UTC")
        # Normalised to UTC first: the same instant expressed in two offsets is
        # the same value and must produce the same digest.
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, frozenset | set):
        # Sets have no inherent order; sort so the digest is deterministic.
        return sorted(_encode(v) for v in value)
    raise TypeError(f"no canonical encoding for {type(value).__name__}")


def canonical_json(value: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_encode,
        allow_nan=False,
    )


def canonical_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest(value: Mapping[str, Any] | list[Any]) -> str:
    """Hex sha256 of the canonical encoding."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_text(text: str) -> str:
    """Hex sha256 of raw text.

    Used for file content, where the bytes themselves are the value and
    re-encoding through JSON would be lossy and pointless.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
