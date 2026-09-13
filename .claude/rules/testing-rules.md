# Testing Rules

Every meaningful behavior change requires tests.

Testing hierarchy:

Unit
→ Integration
→ End-to-End

Before changing behavior:
- Inspect existing tests.
- Understand current expected behavior.

After implementation:
- Add or update tests.
- Run the narrowest relevant tests first.
- Run the broader suite before completion.

Never:
- Delete tests merely because they fail.
- Modify tests only to make an incorrect implementation pass.
- Hardcode behavior to satisfy a specific test.
- Claim tests passed without actually running them.

Tests verify requirements.
Tests do not define requirements.

For ASES orchestration, test:
- DAG dependencies
- execution ordering
- parallel execution
- synchronization
- state transitions
- retries
- fallback
- rollback/compensation
- safe-stop
- approval/rejection
- re-planning
- event replay
- artifact lineage