import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from agentguard.adapters.langchain import DurableExecutor
from agentguard.stores.memory import InMemoryStore
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta


def make_mock_executor(return_value=None):
    ex = MagicMock()
    ex.ainvoke = AsyncMock(return_value=return_value or {"output": "done"})
    return ex


@pytest.mark.asyncio
async def test_ainvoke_checkpoints_state():
    store = InMemoryStore()
    ex = make_mock_executor({"output": "lc result"})
    de = DurableExecutor(executor=ex, store=store)
    await de.ainvoke({"input": "test"}, run_id="run-lc1")
    metas = await store.list("run-lc1")
    assert len(metas) >= 1
    assert metas[0].framework == "langchain"


@pytest.mark.asyncio
async def test_ainvoke_returns_executor_result():
    store = InMemoryStore()
    ex = make_mock_executor({"output": "lc output"})
    de = DurableExecutor(executor=ex, store=store)
    result = await de.ainvoke({"input": "test"}, run_id="run-lc2")
    assert result == {"output": "lc output"}


@pytest.mark.asyncio
async def test_resume_restores_and_reinvokes():
    store = InMemoryStore()
    state = json.dumps({"input": {"input": "resumed"}, "step": 2}).encode("utf-8")
    meta = CheckpointMeta(
        run_id="run-lc3", step=2, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langchain",
    )
    await store.save("run-lc3", 2, state, meta)
    ex = make_mock_executor({"output": "resumed lc"})
    de = DurableExecutor(executor=ex, store=store)
    result = await de.resume("run-lc3")
    assert result == {"output": "resumed lc"}


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    de = DurableExecutor(executor=MagicMock(), store=store)
    with pytest.raises(RestoreError):
        await de.resume("missing-lc-run")
