"""Runnable demonstration of the kernel as implemented so far.

    uv run python scripts/demo_kernel.py

No LLM, no database, no network - which is the point. Everything shown here is
the deterministic control plane operating on its own.

Five things are demonstrated:

1. A graph with parallel fan-out, a barrier join and a bounded repair cycle
   validates; the same graph without a cycle budget is refused.
2. A dynamic subgraph is admitted only after re-validation.
3. A realistic run is written to an append-only hash-chained log.
4. Run state is folded from that log - including a human approval bound to an
   artifact hash, and its revocation when the upstream artifact changes.
5. Editing the log on disk is detected.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

from ases.kernel.events import Actor, EventType, UnsealedEvent
from ases.kernel.graph import (
    Edge,
    EdgeCondition,
    GraphError,
    JoinPolicy,
    NodeKind,
    NodeSpec,
    WorkflowGraph,
)
from ases.kernel.state import fold
from ases.kernel.store.jsonl import JsonlEventStore

DEMO_DIR = Path(__file__).resolve().parent.parent / "runs" / "_demo"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def agent_node(node_id: str, **kw: object) -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.AGENT, handler=node_id, **kw)  # type: ignore[arg-type]


def build_graph() -> WorkflowGraph:
    """A miniature of the greenfield workflow.

    REQ -> GATE1 -> SCAFFOLD -> DECOMPOSE -> (IMPL_A || IMPL_B) -> BARRIER
        -> TEST <-> REPAIR (bounded) -> DONE
    """
    return WorkflowGraph(
        name="demo-greenfield",
        entry=("req",),
        nodes=(
            # cycle_budget bounds the clarification cycle: a human may send the
            # requirement back at most 3 times before the run safe-stops.
            agent_node("req", cycle_budget=3),
            NodeSpec(id="gate1", kind=NodeKind.GATE, requires_approval=True),
            agent_node("scaffold"),
            agent_node("decompose"),
            NodeSpec(id="barrier", kind=NodeKind.BARRIER, join=JoinPolicy.ALL),
            agent_node("test", cycle_budget=2),
            agent_node("repair"),
            NodeSpec(id="done", kind=NodeKind.TERMINAL),
        ),
        edges=(
            Edge(source="req", target="gate1"),
            Edge(source="gate1", target="scaffold"),
            # A rejected gate sends the requirement back - the clarification cycle.
            Edge(source="gate1", target="req", condition=EdgeCondition.ON_REJECTED),
            Edge(source="scaffold", target="decompose"),
            Edge(source="decompose", target="barrier"),
            Edge(source="barrier", target="test"),
            Edge(source="test", target="repair", condition=EdgeCondition.ON_FAILURE),
            Edge(source="repair", target="test", condition=EdgeCondition.ALWAYS),
            Edge(source="test", target="done"),
        ),
    )


def demo_graph_validation() -> WorkflowGraph:
    rule("1. Graph validation")
    graph = build_graph()
    graph.validate_graph()
    print(f"  OK    {graph.name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"        entry={graph.entry}  terminals={graph.terminal_nodes}")
    print("        'req' has a backward edge from gate1, yet is still the entry point")
    print("        - which is why entry is declared rather than inferred.")

    # The same graph with the repair loop's budget removed must be refused.
    unbounded = WorkflowGraph(
        name=graph.name,
        entry=graph.entry,
        nodes=tuple(
            agent_node("test") if n.id == "test" else n  # drop cycle_budget
            for n in graph.nodes
        ),
        edges=graph.edges,
    )
    try:
        unbounded.validate_graph()
    except GraphError as exc:
        print(f"  REFUSED  {exc}")
    return graph


def demo_subgraph_admission(graph: WorkflowGraph) -> None:
    rule("2. Dynamic subgraph admission")
    expanded = graph.with_subgraph(
        nodes=[agent_node("impl_a"), agent_node("impl_b")],
        edges=[
            Edge(source="decompose", target="impl_a"),
            Edge(source="decompose", target="impl_b"),
            Edge(source="impl_a", target="barrier"),
            Edge(source="impl_b", target="barrier"),
        ],
    )
    print(f"  ADMITTED  decomposer proposed 2 tasks -> {len(expanded.nodes)} nodes")
    print(f"            'barrier' now joins {len(expanded.predecessors('barrier'))} edges (ALL)")

    # The realistic decomposer failure: a task wired to a node that does not
    # exist, because the model invented a plausible-sounding name.
    try:
        graph.with_subgraph(
            nodes=[agent_node("impl_c")],
            edges=[
                Edge(source="decompose", target="impl_c"),
                Edge(source="impl_c", target="integration_barrier"),
            ],
        )
    except GraphError as exc:
        print(f"  REFUSED   {exc}")
    print("            The proposal never executes: it is checked at admission.")


async def demo_run(store: JsonlEventStore) -> None:
    rule("3. A run, written to an append-only hash-chained log")
    run_id = uuid4()

    def ev(
        event_type: EventType,
        *,
        node: str | None = None,
        who: Actor | None = None,
        **p: object,
    ) -> UnsealedEvent:
        return UnsealedEvent(
            run_id=run_id,
            type=event_type,
            actor=who or Actor.kernel(),
            node_id=node,
            payload=p,
        )

    await store.append_all(
        [
            ev(EventType.RUN_CREATED, workflow="demo-greenfield"),
            ev(EventType.RUN_STARTED),
            # --- requirement analysis, first attempt
            ev(EventType.NODE_READY, node="req"),
            ev(EventType.NODE_STARTED, node="req"),
            ev(
                EventType.LLM_COMPLETED,
                node="req",
                model="claude-opus-5",
                input_tokens=4200,
                output_tokens=1100,
                usd=0.31,
            ),
            ev(
                EventType.ARTIFACT_PRODUCED,
                node="req",
                artifact_hash="sha-req-v1",
                kind="RequirementSpec",
            ),
            ev(EventType.NODE_SUCCEEDED, node="req"),
            # --- human gate: rejected, then re-run (the clarification cycle)
            ev(EventType.NODE_READY, node="gate1"),
            ev(EventType.NODE_STARTED, node="gate1"),
            ev(EventType.APPROVAL_REQUESTED, node="gate1", artifact_hash="sha-req-v1"),
            ev(
                EventType.APPROVAL_REJECTED,
                node="gate1",
                who=Actor.human("alice"),
                artifact_hash="sha-req-v1",
                reason="'daily' is undefined: calendar day or rolling 24h?",
            ),
            ev(EventType.NODE_READY, node="gate1"),
        ]
    )

    # --- the requirement is amended; the old approval must not survive it
    await store.append_all(
        [
            ev(EventType.NODE_MARKED_STALE, node="req"),
            ev(EventType.NODE_READY, node="req"),
            ev(EventType.NODE_STARTED, node="req"),
            ev(
                EventType.ARTIFACT_PRODUCED,
                node="req",
                artifact_hash="sha-req-v2",
                kind="RequirementSpec",
                inputs=["sha-req-v1"],
            ),
            ev(EventType.NODE_SUCCEEDED, node="req"),
            ev(
                EventType.APPROVAL_REVOKED,
                node="gate1",
                reason="upstream artifact changed sha-req-v1 -> sha-req-v2",
            ),
            ev(EventType.NODE_STARTED, node="gate1"),
            ev(EventType.APPROVAL_REQUESTED, node="gate1", artifact_hash="sha-req-v2"),
            ev(
                EventType.APPROVAL_GRANTED,
                node="gate1",
                who=Actor.human("alice"),
                artifact_hash="sha-req-v2",
                reason="rolling 24h, sender timezone",
            ),
            # --- a policy guardrail fires
            ev(
                EventType.POLICY_VIOLATION,
                node="scaffold",
                rule="no_raw_sql_migration",
                severity="high",
            ),
        ]
    )

    events = await store.read_all(run_id)
    print(f"  {len(events)} events written to runs/_demo/{run_id}/events.jsonl")
    print(f"  first: seq={events[0].seq} prev_hash={events[0].prev_hash[:16]}... (genesis)")
    print(f"  last:  seq={events[-1].seq} hash={events[-1].hash[:16]}...")

    rule("4. State folded from the log")
    state = fold(run_id, events)
    print(f"  run status       : {state.status}")
    print(f"  workflow         : {state.workflow}")
    print(f"  tokens / cost    : {state.usage.total_tokens} tokens, ${state.usage.usd:.2f}")
    print(f"  policy violations: {state.policy_violations}")
    print("  nodes:")
    for node_id, node in state.nodes.items():
        produced = f"  produced={list(node.produced)}" if node.produced else ""
        print(f"    {node_id:<10} {node.status}{produced}")
    print("  approvals:")
    for approval in state.approvals.values():
        flag = "GRANTED" if approval.granted else ("REVOKED" if approval.revoked else "OPEN")
        print(f"    {approval.node_id:<10} {flag:<8} bound to {approval.artifact_hash}")
        if approval.revoked_reason:
            print(f"               reason: {approval.revoked_reason}")
    print("\n  Note the approval granted against sha-req-v1 did not survive the")
    print("  requirement being amended. Inheriting it would mean approving")
    print("  content nobody reviewed.")

    verification = await store.verify_chain(run_id)
    print(f"\n  chain verification: ok={verification.ok} ({verification.events_checked} events)")

    rule("5. Tamper detection")
    path = DEMO_DIR / str(run_id) / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if "approval.rejected" in line)
    record = json.loads(lines[target])
    record["payload"]["reason"] = "looked fine to me"
    lines[target] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f'  edited seq {target + 1} on disk: rejection reason -> "looked fine to me"')

    after = await store.verify_chain(run_id)
    print(f"  chain verification: ok={after.ok}")
    for problem in after.problems:
        print(f"    seq {problem.seq}: {problem.reason}")
    print("\n  The edit is detectable because the reason is inside the hashed body.")


async def main() -> None:
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    store = JsonlEventStore(DEMO_DIR, fsync=False)

    graph = demo_graph_validation()
    demo_subgraph_admission(graph)
    await demo_run(store)
    print(f"\nArtifacts left in {DEMO_DIR.relative_to(Path.cwd())} for inspection.\n")


if __name__ == "__main__":
    asyncio.run(main())
