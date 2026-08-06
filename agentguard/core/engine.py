import logging
from collections.abc import Callable
from agentguard.core.exceptions import RestoreError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerPolicy
from agentguard.core.types import CheckpointMeta, RestoredState

logger = logging.getLogger(__name__)


class CheckpointEngine:
    def __init__(
        self,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or TriggerPolicy()
        self.on_checkpoint_failure = on_checkpoint_failure

    async def checkpoint(
        self, run_id: str, step: int, state: bytes, meta: CheckpointMeta
    ) -> None:
        try:
            await self.store.save(run_id, step, state, meta)
        except Exception as exc:
            logger.warning(
                "AgentGuard checkpoint failed: run=%s step=%d error=%s",
                run_id, step, exc,
            )
            if self.on_checkpoint_failure:
                self.on_checkpoint_failure(run_id, step, exc)

    async def restore(self, run_id: str) -> RestoredState:
        result = await self.store.load_latest(run_id)
        if result is None:
            raise RestoreError(f"No checkpoint found for run_id={run_id}")
        if not result.state:
            raise RestoreError(
                f"Corrupted checkpoint: run={run_id} step={result.step}"
            )
        return result
