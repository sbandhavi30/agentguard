import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta
from agentguard.stores.memory import InMemoryStore


def make_meta(step=1):
    return CheckpointMeta(
        run_id="run-1", step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )


@pytest.mark.asyncio
async def test_checkpoint_saves_to_store():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))
    restored = await store.load_latest("run-1")
    assert restored is not None
    assert restored.state == b"state"


@pytest.mark.asyncio
async def test_checkpoint_failure_does_not_raise():
    store = InMemoryStore()
    store.save = AsyncMock(side_effect=Exception("disk full"))
    engine = CheckpointEngine(store=store)
    # Must NOT raise — agent continues
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))


@pytest.mark.asyncio
async def test_checkpoint_failure_calls_callback():
    store = InMemoryStore()
    store.save = AsyncMock(side_effect=Exception("disk full"))
    callback = MagicMock()
    engine = CheckpointEngine(store=store, on_checkpoint_failure=callback)
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))
    callback.assert_called_once()
    args = callback.call_args[0]
    assert args[0] == "run-1"
    assert args[1] == 1


@pytest.mark.asyncio
async def test_restore_returns_latest():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    await engine.checkpoint("run-1", 3, b"state-3", make_meta(3))
    restored = await engine.restore("run-1")
    assert restored.step == 3
    assert restored.state == b"state-3"


@pytest.mark.asyncio
async def test_restore_raises_when_no_checkpoint():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    with pytest.raises(RestoreError, match="run_id=missing-run"):
        await engine.restore("missing-run")


@pytest.mark.asyncio
async def test_restore_raises_on_empty_state():
    store = InMemoryStore()
    # Manually seed corrupted checkpoint
    meta = CheckpointMeta(
        run_id="run-1", step=1, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    await store.save("run-1", 1, b"", meta)
    engine = CheckpointEngine(store=store)
    with pytest.raises(RestoreError, match="Corrupted"):
        await engine.restore("run-1")
