import json
from collections.abc import Callable
from datetime import datetime, timezone
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta


class DurableGraph:
    def __init__(
        self,
        graph,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        framework_name: str = "langgraph",
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self._graph = graph
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = framework_name

    def _serialize(self, input_dict: dict, step: int) -> bytes:
        return json.dumps({"input": input_dict, "step": step}).encode("utf-8")

    def _deserialize(self, state: bytes) -> dict:
        try:
            return json.loads(state.decode("utf-8"))
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize LangGraph state: {exc}") from exc

    async def ainvoke(self, input_dict: dict, run_id: str, **kwargs):
        step = 0
        trigger_meta = TriggerMeta(tool_name=None, step=step)
        _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)

        state_bytes = self._serialize(input_dict, step)
        meta = CheckpointMeta(
            run_id=run_id, step=step, trigger=trigger_reason,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, state_bytes, meta)

        result = await self._graph.ainvoke(input_dict, **kwargs)

        step += 1
        try:
            final_state_bytes = self._serialize({"result": result}, step)
        except (TypeError, ValueError):
            final_state_bytes = self._serialize(
                {"result": None, "_serialization_failed": True}, step
            )
        final_meta = CheckpointMeta(
            run_id=run_id, step=step, trigger="completed",
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, final_state_bytes, final_meta)
        return result

    async def resume(self, run_id: str, **kwargs):
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        input_dict = state.get("input", state)
        return await self.ainvoke(input_dict, run_id=run_id, **kwargs)
