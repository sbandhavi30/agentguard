from abc import ABC, abstractmethod
from agentguard.core.types import CheckpointMeta, RestoredState


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None: ...

    @abstractmethod
    async def load_latest(self, run_id: str) -> RestoredState | None: ...

    @abstractmethod
    async def list(self, run_id: str) -> list[CheckpointMeta]: ...

    @abstractmethod
    async def delete(self, run_id: str) -> None: ...

    @abstractmethod
    async def prune(self, run_id: str, keep_last: int = 3) -> None: ...
