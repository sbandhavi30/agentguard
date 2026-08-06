import json
import logging
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta

logger = logging.getLogger(__name__)

_TOTAL_BUDGET = 200_000  # Claude default context window


class DurableAgentLoop:
    def __init__(
        self,
        client,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        framework_name: str = "anthropic",
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self._client = client
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = framework_name

    def _serialize(self, messages: list, step: int) -> bytes:
        return json.dumps({"messages": messages, "step": step}).encode("utf-8")

    def _deserialize(self, state: bytes) -> dict:
        try:
            return json.loads(state.decode("utf-8"))
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize Anthropic state: {exc}") from exc

    async def _make_meta(self, run_id: str, step: int, token_count: int, trigger: str) -> CheckpointMeta:
        return CheckpointMeta(
            run_id=run_id,
            step=step,
            trigger=trigger,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
            token_count=token_count,
        )

    async def run(
        self,
        messages: list,
        run_id: str,
        model: str,
        tools: list,
        tool_executor: Callable[[str, dict], Awaitable[list]] | None = None,
        **kwargs,
    ):
        step = 0
        current_messages = list(messages)

        while True:
            response = await self._client.messages.create(
                model=model, messages=current_messages, tools=tools, **kwargs
            )
            token_count = getattr(response.usage, "input_tokens", 0)

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                for block in tool_blocks:
                    trigger_meta = TriggerMeta(
                        tool_name=block.name,
                        token_count=token_count,
                        total_budget=_TOTAL_BUDGET,
                        step=step,
                    )
                    _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)
                    state_bytes = self._serialize(current_messages, step)
                    meta = await self._make_meta(run_id, step, token_count, trigger_reason)
                    await self._engine.checkpoint(run_id, step, state_bytes, meta)

                    if tool_executor:
                        results = await tool_executor(block.name, block.input)
                        content = [
                            b.model_dump() if hasattr(b, "model_dump") else dict(vars(b))
                            for b in response.content
                        ]
                        current_messages = current_messages + [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": results},
                        ]
                step += 1

            else:
                return response

    async def resume(self, run_id: str, model: str, tools: list, **kwargs):
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        messages = state["messages"]
        return await self.run(
            messages=messages,
            run_id=run_id,
            model=model,
            tools=tools,
            **kwargs,
        )
