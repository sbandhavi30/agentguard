import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentguard.cli import app
from agentguard.core.types import CheckpointMeta
from agentguard.stores.disk import DiskStore

runner = CliRunner()


async def seed_store(base_dir: Path, run_id: str, steps: int) -> None:
    store = DiskStore(base_dir=base_dir)
    for step in range(steps):
        meta = CheckpointMeta(
            run_id=run_id,
            step=step,
            trigger="tool_call",
            timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
            framework="anthropic",
        )
        await store.save(run_id, step, f"state-{step}".encode(), meta)


def test_list_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli1", 3))
    result = runner.invoke(app, ["list", "run-cli1", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "run-cli1" in result.output
    assert "tool_call" in result.output


def test_list_empty_run(tmp_path):
    result = runner.invoke(app, ["list", "nonexistent-run", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "No checkpoints" in result.output


def test_inspect_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli2", 3))
    result = runner.invoke(
        app, ["inspect", "run-cli2", "--step", "1", "--store", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "step" in result.output.lower()


def test_inspect_missing_step(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli2b", 2))
    result = runner.invoke(
        app, ["inspect", "run-cli2b", "--step", "99", "--store", str(tmp_path)]
    )
    assert result.exit_code == 1


def test_resume_command_prints_snippet(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli3", 2))
    result = runner.invoke(app, ["resume", "run-cli3", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "DurableAgentLoop" in result.output or "resume" in result.output.lower()


def test_resume_no_checkpoints(tmp_path):
    result = runner.invoke(app, ["resume", "nonexistent-run", "--store", str(tmp_path)])
    assert result.exit_code == 1
    assert "No checkpoints" in result.output


def test_prune_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli4", 5))
    result = runner.invoke(
        app, ["prune", "run-cli4", "--keep", "2", "--store", str(tmp_path)]
    )
    assert result.exit_code == 0
    store = DiskStore(base_dir=tmp_path)
    metas = asyncio.run(store.list("run-cli4"))
    assert len(metas) == 2


def test_delete_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli5", 2))
    result = runner.invoke(app, ["delete", "run-cli5", "--store", str(tmp_path)])
    assert result.exit_code == 0
    store = DiskStore(base_dir=tmp_path)
    metas = asyncio.run(store.list("run-cli5"))
    assert metas == []


def test_stats_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli6", 4))
    result = runner.invoke(app, ["stats", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "total" in result.output.lower()


def test_stats_empty_store(tmp_path):
    empty = tmp_path / "empty_store"
    empty.mkdir()
    result = runner.invoke(app, ["stats", "--store", str(empty)])
    assert result.exit_code == 0
    assert "total" in result.output.lower()


def test_stats_no_store(tmp_path):
    missing = tmp_path / "missing_store"
    result = runner.invoke(app, ["stats", "--store", str(missing)])
    assert result.exit_code == 0
    assert "No checkpoint store" in result.output
