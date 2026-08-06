import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState

_SAFE_RUN_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _validate_run_id(run_id: str) -> None:
    """Reject run_ids that could escape the store root via path traversal."""
    if not _SAFE_RUN_ID.match(run_id):
        raise ValueError(f"Invalid run_id {run_id!r}: only [a-zA-Z0-9_-] allowed")


class DiskStore(StorageBackend):
    def __init__(self, base_dir: str | Path = ".agentguard") -> None:
        self._base = Path(base_dir)

    def _run_dir(self, run_id: str) -> Path:
        return self._base / run_id

    def _meta_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "meta.jsonl"

    def _step_path(self, run_id: str, step: int) -> Path:
        return self._run_dir(run_id) / f"step_{step:05d}.bin"

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        _validate_run_id(run_id)
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._step_path(run_id, step).write_bytes(state)
        with self._meta_path(run_id).open("a") as f:
            f.write(json.dumps({
                "run_id": meta.run_id,
                "step": meta.step,
                "trigger": meta.trigger,
                "timestamp": meta.timestamp.isoformat(),
                "framework": meta.framework,
                "token_count": meta.token_count,
                "cost_usd": meta.cost_usd,
            }) + "\n")

    async def load_latest(self, run_id: str) -> RestoredState | None:
        _validate_run_id(run_id)
        metas = await self.list(run_id)
        if not metas:
            return None
        latest = metas[0]
        state = self._step_path(run_id, latest.step).read_bytes()
        return RestoredState(step=latest.step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        _validate_run_id(run_id)
        meta_path = self._meta_path(run_id)
        if not meta_path.exists():
            return []
        metas = []
        seen_steps: set[int] = set()
        for line in meta_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            step = d["step"]
            if step in seen_steps:
                continue
            seen_steps.add(step)
            metas.append(CheckpointMeta(
                run_id=d["run_id"],
                step=step,
                trigger=d["trigger"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                framework=d["framework"],
                token_count=d.get("token_count", 0),
                cost_usd=d.get("cost_usd", 0.0),
            ))
        return sorted(metas, key=lambda m: m.step, reverse=True)

    async def delete(self, run_id: str) -> None:
        _validate_run_id(run_id)
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        _validate_run_id(run_id)
        metas = await self.list(run_id)
        for meta in metas[keep_last:]:
            step_file = self._step_path(run_id, meta.step)
            if step_file.exists():
                step_file.unlink()
        kept_steps = {m.step for m in metas[:keep_last]}
        kept_metas = [m for m in metas if m.step in kept_steps]
        meta_path = self._meta_path(run_id)
        with meta_path.open("w") as f:
            for m in sorted(kept_metas, key=lambda x: x.step):
                f.write(json.dumps({
                    "run_id": m.run_id,
                    "step": m.step,
                    "trigger": m.trigger,
                    "timestamp": m.timestamp.isoformat(),
                    "framework": m.framework,
                    "token_count": m.token_count,
                    "cost_usd": m.cost_usd,
                }) + "\n")
