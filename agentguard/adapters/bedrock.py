"""
AgentGuard adapter for AWS Bedrock Converse API.

Bedrock's Converse API uses a different message/tool format than Anthropic's:
  - messages: list of {"role": "user"|"assistant", "content": [{"text": ...}|{"toolUse": ...}|{"toolResult": ...}]}
  - stop_reason: "tool_use" → "toolUse", "end_turn" → "endTurn"
  - tool blocks: response["output"]["message"]["content"] list of {"toolUse": {"toolUseId", "name", "input"}}
  - tool results: {"toolResult": {"toolUseId": ..., "content": [{"text": ...}]}}

Supports any model available via Bedrock Converse:
  - anthropic.claude-3-5-haiku-20241022-v1:0
  - anthropic.claude-3-5-sonnet-20241022-v2:0
  - amazon.nova-pro-v1:0
  - amazon.nova-lite-v1:0
  - meta.llama3-70b-instruct-v1:0
  - mistral.mistral-large-2402-v1:0
  - ... (any Converse-compatible model)

Install:  pip install "agentguard[bedrock]"
Requires: AWS credentials configured (aws configure / env vars / IAM role)
"""

import json
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone

from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import AgentLoopDetectedError, DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta

_TOTAL_BUDGET = 200_000


class DurableBedrockLoop:
    """
    Durable agent loop for AWS Bedrock Converse API.

    Usage:
        import boto3
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
        store = DiskStore(".agentguard")
        loop = DurableBedrockLoop(client=bedrock, store=store, max_steps=10)

        result = await loop.run(
            messages=[{"role": "user", "content": [{"text": "What is the weather in NYC?"}]}],
            run_id="run-001",
            model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
            tools=[{
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
                }
            }],
            tool_executor=my_tool_executor,
        )

    tool_executor signature:
        async def my_executor(tool_name: str, tool_input: dict) -> str:
            # return a plain string — adapter wraps it in Bedrock toolResult format
            return "result text"
    """

    def __init__(
        self,
        client,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = "bedrock"

    def _serialize(self, messages: list, step: int) -> bytes:
        return json.dumps({"messages": messages, "step": step}).encode("utf-8")

    def _deserialize(self, state: bytes) -> dict:
        try:
            return json.loads(state.decode("utf-8"))
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize Bedrock state: {exc}") from exc

    def _make_meta(self, run_id: str, step: int, token_count: int, trigger: str) -> CheckpointMeta:
        return CheckpointMeta(
            run_id=run_id,
            step=step,
            trigger=trigger,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
            token_count=token_count,
        )

    def _call_converse(self, model_id: str, messages: list, tools: list, **kwargs) -> dict:
        """Synchronous Bedrock converse call (boto3 is sync-only)."""
        params = {"modelId": model_id, "messages": messages}
        if tools:
            params["toolConfig"] = {"tools": tools}
        params.update(kwargs)
        return self._client.converse(**params)

    async def run(
        self,
        messages: list,
        run_id: str,
        model_id: str,
        tools: list,
        tool_executor: Callable[[str, dict], Awaitable[str]] | None = None,
        **kwargs,
    ):
        """
        Run the agent loop until end_turn or loop detection.

        Returns the final Bedrock converse response dict.
        """
        import asyncio
        step = 0
        current_messages = list(messages)

        while True:
            # boto3 is sync — run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._call_converse(model_id, current_messages, tools, **kwargs)
            )

            stop_reason = response.get("stopReason", "")
            usage = response.get("usage", {})
            token_count = usage.get("inputTokens", 0)
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            if stop_reason == "tool_use":
                tool_blocks = [b["toolUse"] for b in content_blocks if "toolUse" in b]

                trigger_meta = TriggerMeta(
                    tool_name=tool_blocks[0]["name"] if tool_blocks else None,
                    token_count=token_count,
                    total_budget=_TOTAL_BUDGET,
                    step=step,
                )
                _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)
                state_bytes = self._serialize(current_messages, step)
                meta = self._make_meta(run_id, step, token_count, trigger_reason)
                await self._engine.checkpoint(run_id, step, state_bytes, meta)

                if self._max_steps is not None and step >= self._max_steps:
                    raise AgentLoopDetectedError(run_id=run_id, step=step)

                if not tool_executor:
                    return response

                # Execute tools, build Bedrock-format tool results
                tool_results = []
                for tb in tool_blocks:
                    result_text = await tool_executor(tb["name"], tb["input"])
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tb["toolUseId"],
                            "content": [{"text": str(result_text)}],
                        }
                    })

                # Bedrock message format: assistant turn + user turn with tool results
                current_messages = current_messages + [
                    {"role": "assistant", "content": content_blocks},
                    {"role": "user", "content": tool_results},
                ]
                step += 1

            else:
                # endTurn, max_tokens, stop_sequence, etc.
                return response

    async def resume(self, run_id: str, model_id: str, tools: list, **kwargs):
        """Resume from the latest checkpoint."""
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        return await self.run(
            messages=state["messages"],
            run_id=run_id,
            model_id=model_id,
            tools=tools,
            **kwargs,
        )
