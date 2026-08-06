import pytest
from datetime import datetime, timezone
from testcontainers.community.redis import RedisContainer
from agentguard.stores.redis import RedisStore
from agentguard.core.types import CheckpointMeta


def make_meta(run_id="run-1", step=1, framework="langgraph"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture(scope="module")
def redis_url():
    with RedisContainer() as container:
        yield f"redis://{container.get_container_host_ip()}:{container.get_exposed_port(6379)}"


@pytest.fixture
async def store(redis_url):
    s = RedisStore(url=redis_url)
    yield s
    await s.aclose()


@pytest.mark.asyncio
async def test_save_and_load_latest(store):
    await store.save("run-r1", 3, b"state-3", make_meta(step=3))
    restored = await store.load_latest("run-r1")
    assert restored is not None
    assert restored.step == 3
    assert restored.state == b"state-3"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(store):
    result = await store.load_latest("run-nonexistent-xyz")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(store):
    for step in [1, 2, 3]:
        await store.save("run-r2", step, b"s", make_meta(run_id="run-r2", step=step))
    metas = await store.list("run-r2")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_clears_all_keys(store):
    await store.save("run-r3", 1, b"s", make_meta(run_id="run-r3"))
    await store.delete("run-r3")
    assert await store.load_latest("run-r3") is None


@pytest.mark.asyncio
async def test_prune_keeps_last_n(store):
    for step in range(6):
        await store.save("run-r4", step, f"s{step}".encode(), make_meta(run_id="run-r4", step=step))
    await store.prune("run-r4", keep_last=2)
    metas = await store.list("run-r4")
    assert len(metas) == 2
    assert metas[0].step == 5
