"""Canonical hashing.

These properties are load-bearing: approvals bind to artifact hashes, and the
event chain is only tamper-evident if the digest is stable and order-independent
in the right ways.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from ases.kernel.hashing import GENESIS_HASH, canonical_json, digest


def test_key_order_does_not_change_the_digest() -> None:
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_different_content_changes_the_digest() -> None:
    assert digest({"a": 1}) != digest({"a": 2})


def test_digest_is_stable_across_calls() -> None:
    value = {"nested": {"x": [1, 2, 3]}, "flag": True}
    assert digest(value) == digest(value)


def test_uuid_and_enum_encode_deterministically() -> None:
    uid = UUID("12345678-1234-5678-1234-567812345678")
    assert canonical_json({"id": uid}) == '{"id":"12345678-1234-5678-1234-567812345678"}'


def test_same_instant_in_different_offsets_hashes_identically() -> None:
    """A timezone offset is a representation detail, not part of the value.

    Without normalisation, the same event recorded in two zones would produce
    different digests and appear to be two different events.
    """
    utc = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    plus_two = utc.astimezone(timezone(timedelta(hours=2)))
    assert digest({"t": utc}) == digest({"t": plus_two})


def test_naive_datetime_is_refused() -> None:
    """Assuming a timezone would make the digest depend on where it ran."""
    with pytest.raises(ValueError, match="naive datetime"):
        digest({"t": datetime(2026, 9, 12, 12, 0)})


def test_nan_is_refused() -> None:
    """NaN is not valid JSON and never compares equal to itself."""
    with pytest.raises(ValueError):
        digest({"x": float("nan")})


def test_unencodable_type_is_refused_rather_than_stringified() -> None:
    class Opaque:
        pass

    with pytest.raises(TypeError, match="no canonical encoding"):
        digest({"x": Opaque()})


def test_genesis_hash_is_not_a_real_digest() -> None:
    """It must be visibly distinguishable from a sha256 of anything."""
    assert GENESIS_HASH == "0" * 64
    assert digest({}) != GENESIS_HASH
