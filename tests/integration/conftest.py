"""Fixtures for tests that require the live Postgres container.

Assumes `ases db bootstrap` and `ases db upgrade` have already been run
(`scripts/dev-up.sh` does both) - these tests exercise the running store, not
the provisioning process itself; provisioning has its own checks in
`test_database_setup.py`. If the schema is missing, tests here fail with a
clear connection/permission error rather than silently re-provisioning, so a
broken `dev-up` is never masked by a test quietly working around it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from ases.config import settings
from ases.kernel.store.postgres import PostgresEventStore


@pytest_asyncio.fixture
async def pg_store() -> AsyncIterator[PostgresEventStore]:
    store = PostgresEventStore(settings().app_dsn)
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def run_id() -> UUID:
    return uuid4()
