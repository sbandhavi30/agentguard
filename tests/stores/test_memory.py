import pytest
from datetime import datetime, timezone
from agentguard.stores.memory import InMemoryStore
from agentguard.core.types import CheckpointMeta, RestoredState


def make_meta(run_id="run-1", step=1, trigger="tool_call", framework="anthropic"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger=trigger,
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.mark.asyncio
async def test_save_and_load_latest():
    store = InMemoryStore()
    meta = make_meta(step=5)
    state = b"serialized-state"
    await store.save("run-1", 5, state, meta)
    restored = await store.load_latest("run-1")
    assert restored is not None
    assert restored.step == 5
    assert restored.state == state


@pytest.mark.asyncio
async def test_load_latest_returns_none_when_empty():
    store = InMemoryStore()
    result = await store.load_latest("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_load_latest_returns_highest_step():
    store = InMemoryStore()
    for step in [1, 3, 2]:
        await store.save("run-1", step, f"state-{step}".encode(), make_meta(step=step))
    restored = await store.load_latest("run-1")
    assert restored.step == 3


@pytest.mark.asyncio
async def test_list_returns_all_meta_desc():
    store = InMemoryStore()
    for step in [1, 2, 3]:
        await store.save("run-1", step, b"s", make_meta(step=step))
    metas = await store.list("run-1")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_run():
    store = InMemoryStore()
    await store.save("run-1", 1, b"s", make_meta())
    await store.delete("run-1")
    assert await store.load_latest("run-1") is None
    assert await store.list("run-1") == []


@pytest.mark.asyncio
async def test_prune_keeps_last_n():
    store = InMemoryStore()
    for step in range(10):
        await store.save("run-1", step, f"s{step}".encode(), make_meta(step=step))
    await store.prune("run-1", keep_last=3)
    metas = await store.list("run-1")
    assert len(metas) == 3
    assert metas[0].step == 9
