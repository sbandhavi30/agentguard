from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState


class InMemoryStore(StorageBackend):
    def __init__(self) -> None:
        # {run_id: {step: (state_bytes, CheckpointMeta)}}
        self._data: dict[str, dict[int, tuple[bytes, CheckpointMeta]]] = {}

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        if run_id not in self._data:
            self._data[run_id] = {}
        self._data[run_id][step] = (state, meta)

    async def load_latest(self, run_id: str) -> RestoredState | None:
        if run_id not in self._data or not self._data[run_id]:
            return None
        latest_step = max(self._data[run_id].keys())
        state, _ = self._data[run_id][latest_step]
        return RestoredState(step=latest_step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        if run_id not in self._data:
            return []
        return [
            meta
            for _, meta in sorted(
                self._data[run_id].values(), key=lambda x: x[1].step, reverse=True
            )
        ]

    async def delete(self, run_id: str) -> None:
        self._data.pop(run_id, None)

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        if run_id not in self._data:
            return
        steps = sorted(self._data[run_id].keys(), reverse=True)
        for step in steps[keep_last:]:
            del self._data[run_id][step]
