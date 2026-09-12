"""Runtime configuration.

Loaded from the environment (and `.env` when present). Deliberately tolerant of
a missing `.env` so that unit tests - which must not require infrastructure -
can import the package and construct defaults.

Secrets are typed `SecretStr` so they cannot be printed by accident: pydantic
renders them as `**********` in reprs, logs and tracebacks. Reaching the real
value requires an explicit `.get_secret_value()`, which is greppable in review.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
"""Repository root: .../ases/config.py -> ases -> orchestrator -> src -> root."""

CONTROL_SCHEMA: Final[str] = "control"
WORKLOAD_SCHEMA: Final[str] = "workload_test"


class LLMMode(StrEnum):
    """How the agent plane reaches a model.

    `replay` is the default everywhere, including CI: the demo must be
    reproducible offline and without an API key.
    """

    REPLAY = "replay"
    RECORD = "record"
    LIVE = "live"
    MOCK = "mock"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Postgres ---------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ases"

    # Superuser: used by `ases db bootstrap` only, never at runtime.
    postgres_superuser: str = "ases_su"
    postgres_superuser_password: SecretStr = SecretStr("")

    # Least-privilege identities created by bootstrap.
    ases_control_user: str = "ases_control"
    ases_control_password: SecretStr = SecretStr("")
    ases_app_user: str = "ases_app"
    ases_app_password: SecretStr = SecretStr("")
    workload_app_user: str = "workload_app"
    workload_app_password: SecretStr = SecretStr("")

    # --- LLM --------------------------------------------------------------
    ases_llm_mode: LLMMode = LLMMode.REPLAY
    anthropic_api_key: SecretStr = SecretStr("")

    # --- Budgets (enforced at entry gates; exhaustion triggers safe-stop) --
    ases_max_tokens_per_run: int = 2_000_000
    ases_max_usd_per_run: float = 25.0
    ases_max_wallclock_seconds: int = 3600
    ases_max_repair_iterations: int = 3
    ases_max_parallel_nodes: int = 4

    # --- Observability ----------------------------------------------------
    ases_log_level: str = "INFO"
    ases_runs_dir: Path = REPO_ROOT / "runs"

    # --- Dashboard --------------------------------------------------------
    ases_web_host: str = "127.0.0.1"
    ases_web_port: int = 8420

    # ----------------------------------------------------------------------

    def _dsn(self, user: str, password: SecretStr, *, db: str | None = None) -> str:
        return (
            f"postgresql+asyncpg://{user}:{password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{db or self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def superuser_dsn(self) -> str:
        """Bootstrap only. Creating roles and schemas requires elevated rights."""
        return self._dsn(self.postgres_superuser, self.postgres_superuser_password)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def control_dsn(self) -> str:
        """Alembic only. Owns `control`; may run DDL within it."""
        return self._dsn(self.ases_control_user, self.ases_control_password)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def app_dsn(self) -> str:
        """Orchestrator runtime. DML only - no DDL, no UPDATE/DELETE on events.

        This is the identity that makes the append-only guarantee hold at
        runtime: an application bug cannot alter the event log, because the
        connection has no privilege to.
        """
        return self._dsn(self.ases_app_user, self.ases_app_password)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def workload_app_dsn(self) -> str:
        """Agent-generated EF Core migrations. Same database as everything
        else - `workload_test` is a schema, not a separate database - but
        `workload_app`'s `search_path` is set to it by bootstrap, and it has
        no `USAGE` on `control` at all: not a lesser privilege within the
        same namespace, a different namespace entirely."""
        return self._dsn(self.workload_app_user, self.workload_app_password)

    def run_dir(self, run_id: str) -> Path:
        return self.ases_runs_dir / run_id


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings. Cached so `.env` is read once.

    Call `settings.cache_clear()` in tests that need to re-read the environment.
    """
    return Settings()
