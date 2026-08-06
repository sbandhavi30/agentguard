import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from agentguard.adapters.langgraph import DurableGraph
from agentguard.stores.memory import InMemoryStore
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta


def make_mock_graph(return_value=None):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=return_value or {"output": "done"})
    return graph


@pytest.mark.asyncio
async def test_ainvoke_checkpoints_state():
    store = InMemoryStore()
    graph = make_mock_graph({"output": "result"})
    dg = DurableGraph(graph=graph, store=store)
    await dg.ainvoke({"task": "test"}, run_id="run-lg1")
    metas = await store.list("run-lg1")
    assert len(metas) >= 1
    assert metas[0].framework == "langgraph"


@pytest.mark.asyncio
async def test_ainvoke_returns_graph_result():
    store = InMemoryStore()
    graph = make_mock_graph({"output": "my result"})
    dg = DurableGraph(graph=graph, store=store)
    result = await dg.ainvoke({"task": "test"}, run_id="run-lg2")
    assert result == {"output": "my result"}


@pytest.mark.asyncio
async def test_resume_restores_and_reinvokes():
    store = InMemoryStore()
    state = json.dumps({"input": {"task": "resumed task"}, "step": 3}).encode("utf-8")
    meta = CheckpointMeta(
        run_id="run-lg3", step=3, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langgraph",
    )
    await store.save("run-lg3", 3, state, meta)

    graph = make_mock_graph({"output": "resumed"})
    dg = DurableGraph(graph=graph, store=store)
    result = await dg.resume("run-lg3")
    assert result == {"output": "resumed"}
    graph.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    dg = DurableGraph(graph=MagicMock(), store=store)
    with pytest.raises(RestoreError):
        await dg.resume("missing-lg-run")
