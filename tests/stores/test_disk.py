import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from agentguard.stores.disk import DiskStore
from agentguard.core.types import CheckpointMeta, RestoredState


def make_meta(run_id="run-1", step=1, framework="anthropic"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture
def tmp_store(tmp_path):
    return DiskStore(base_dir=tmp_path)


@pytest.mark.asyncio
async def test_save_creates_files(tmp_store, tmp_path):
    await tmp_store.save("run-1", 1, b"state", make_meta())
    assert (tmp_path / "run-1" / "step_00001.bin").exists()
    assert (tmp_path / "run-1" / "meta.jsonl").exists()


@pytest.mark.asyncio
async def test_save_and_load_latest(tmp_store):
    await tmp_store.save("run-1", 1, b"state-1", make_meta(step=1))
    await tmp_store.save("run-1", 2, b"state-2", make_meta(step=2))
    restored = await tmp_store.load_latest("run-1")
    assert restored.step == 2
    assert restored.state == b"state-2"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(tmp_store):
    result = await tmp_store.load_latest("no-such-run")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(tmp_store):
    for step in [1, 2, 3]:
        await tmp_store.save("run-1", step, b"s", make_meta(step=step))
    metas = await tmp_store.list("run-1")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_directory(tmp_store, tmp_path):
    await tmp_store.save("run-1", 1, b"s", make_meta())
    await tmp_store.delete("run-1")
    assert not (tmp_path / "run-1").exists()


@pytest.mark.asyncio
async def test_prune_keeps_last_n(tmp_store, tmp_path):
    for step in range(5):
        await tmp_store.save("run-1", step, f"s{step}".encode(), make_meta(step=step))
    await tmp_store.prune("run-1", keep_last=2)
    metas = await tmp_store.list("run-1")
    assert len(metas) == 2
    assert metas[0].step == 4
