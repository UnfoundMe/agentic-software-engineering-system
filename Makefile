# Convenience wrapper. `make` is not installed on the primary dev machine
# (Windows), so the shell scripts in scripts/ are the single source of truth
# and these targets only delegate. CI on Linux uses these targets.

.PHONY: help dev-up dev-down dev-purge install lint typecheck test test-unit test-integration test-invariants check reset-workload

help:
	@echo "dev-up            start Postgres+Redis, bootstrap roles/schemas, migrate"
	@echo "dev-down          stop containers, preserve data"
	@echo "dev-purge         stop containers AND DESTROY the audit log"
	@echo "install           sync Python dependencies"
	@echo "lint              ruff"
	@echo "typecheck         mypy --strict"
	@echo "test              full suite"
	@echo "check             lint + typecheck + test (what CI runs)"
	@echo "reset-workload    drop and recreate workload_test, preserve audit log"

dev-up:
	@bash scripts/dev-up.sh

dev-down:
	@bash scripts/dev-down.sh

dev-purge:
	@bash scripts/dev-down.sh --purge

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration -m integration

test-invariants:
	uv run pytest tests/invariants -m invariant

check: lint typecheck test

reset-workload:
	uv run ases db reset-workload
