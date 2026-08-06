import pytest
from datetime import datetime, timezone
from testcontainers.community.postgres import PostgresContainer
from agentguard.stores.postgres import PostgresStore
from agentguard.core.types import CheckpointMeta


def make_meta(run_id="run-1", step=1, framework="langchain"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture(scope="module")
def pg_dsn():
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url().replace("psycopg2", "asyncpg").replace("+asyncpg", "")


@pytest.fixture
async def store(pg_dsn):
    s = PostgresStore(dsn=pg_dsn)
    await s.initialize()
    yield s
    await s.aclose()


@pytest.mark.asyncio
async def test_save_and_load_latest(store):
    await store.save("run-p1", 5, b"pg-state", make_meta(step=5))
    restored = await store.load_latest("run-p1")
    assert restored.step == 5
    assert restored.state == b"pg-state"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(store):
    result = await store.load_latest("run-pg-empty")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(store):
    for step in [1, 2, 3]:
        await store.save("run-p2", step, b"s", make_meta(run_id="run-p2", step=step))
    metas = await store.list("run-p2")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_rows(store):
    await store.save("run-p3", 1, b"s", make_meta(run_id="run-p3"))
    await store.delete("run-p3")
    assert await store.load_latest("run-p3") is None


@pytest.mark.asyncio
async def test_prune_keeps_last_n(store):
    for step in range(5):
        await store.save("run-p4", step, f"s{step}".encode(), make_meta(run_id="run-p4", step=step))
    await store.prune("run-p4", keep_last=2)
    metas = await store.list("run-p4")
    assert len(metas) == 2
    assert metas[0].step == 4
