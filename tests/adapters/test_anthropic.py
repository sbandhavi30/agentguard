import pickle
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from agentguard.adapters.anthropic import DurableAgentLoop
from agentguard.stores.memory import InMemoryStore
from agentguard.core.triggers import TriggerPolicy
from agentguard.core.exceptions import RestoreError


def make_mock_client(responses):
    """Return an Anthropic client mock that yields responses in sequence."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


def end_response(text="done"):
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [MagicMock(type="text", text=text)]
    r.usage = MagicMock(input_tokens=100, output_tokens=50)
    return r


def tool_response(tool_name="list_files"):
    r = MagicMock()
    r.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = "tool_abc"
    block.input = {}
    r.content = [block]
    r.usage = MagicMock(input_tokens=200, output_tokens=80)
    return r


@pytest.mark.asyncio
async def test_run_checkpoints_after_tool_call():
    store = InMemoryStore()
    client = make_mock_client([tool_response("list_files"), end_response()])
    loop = DurableAgentLoop(client=client, store=store)

    async def fake_execute(tool_name, tool_input):
        return [{"type": "tool_result", "tool_use_id": "tool_abc", "content": "[]"}]

    await loop.run(
        messages=[{"role": "user", "content": "list files"}],
        run_id="run-a1",
        model="claude-sonnet-4-6",
        tools=[],
        tool_executor=fake_execute,
    )
    metas = await store.list("run-a1")
    assert len(metas) >= 1


@pytest.mark.asyncio
async def test_resume_restores_from_checkpoint():
    store = InMemoryStore()
    # Pre-seed a checkpoint with serialized messages
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    state = pickle.dumps({"messages": messages, "step": 2})
    from agentguard.core.types import CheckpointMeta
    meta = CheckpointMeta(
        run_id="run-a2", step=2, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    await store.save("run-a2", 2, state, meta)

    client = make_mock_client([end_response("resumed ok")])
    loop = DurableAgentLoop(client=client, store=store)

    result = await loop.resume(
        run_id="run-a2",
        model="claude-sonnet-4-6",
        tools=[],
    )
    assert result is not None


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    client = MagicMock()
    loop = DurableAgentLoop(client=client, store=store)
    with pytest.raises(RestoreError):
        await loop.resume(run_id="missing-run", model="claude-sonnet-4-6", tools=[])
