import json
from datetime import datetime
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState

try:
    import redis.asyncio as aioredis
except ImportError as exc:
    raise ImportError("Install agentguard[redis] to use RedisStore") from exc


class RedisStore(StorageBackend):
    def __init__(self, url: str, ttl_seconds: int = 86400 * 7) -> None:
        self._client = aioredis.from_url(url, decode_responses=False)
        self._ttl = ttl_seconds

    def _state_key(self, run_id: str, step: int) -> str:
        return f"agentguard:{run_id}:state:{step}"

    def _meta_key(self, run_id: str, step: int) -> str:
        return f"agentguard:{run_id}:meta:{step}"

    def _steps_key(self, run_id: str) -> str:
        return f"agentguard:{run_id}:steps"

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        pipe = self._client.pipeline()
        pipe.setex(self._state_key(run_id, step), self._ttl, state)
        pipe.setex(
            self._meta_key(run_id, step),
            self._ttl,
            json.dumps({
                "run_id": meta.run_id,
                "step": meta.step,
                "trigger": meta.trigger,
                "timestamp": meta.timestamp.isoformat(),
                "framework": meta.framework,
                "token_count": meta.token_count,
                "cost_usd": meta.cost_usd,
            }).encode(),
        )
        pipe.zadd(self._steps_key(run_id), {str(step): step})
        pipe.expire(self._steps_key(run_id), self._ttl)
        await pipe.execute()

    async def load_latest(self, run_id: str) -> RestoredState | None:
        steps = await self._client.zrange(self._steps_key(run_id), -1, -1)
        if not steps:
            return None
        latest_step = int(steps[0])
        state = await self._client.get(self._state_key(run_id, latest_step))
        if state is None:
            return None
        return RestoredState(step=latest_step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -1, desc=True)
        metas = []
        for step_bytes in steps:
            step = int(step_bytes)
            raw = await self._client.get(self._meta_key(run_id, step))
            if raw is None:
                continue
            d = json.loads(raw)
            metas.append(CheckpointMeta(
                run_id=d["run_id"],
                step=d["step"],
                trigger=d["trigger"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                framework=d["framework"],
                token_count=d.get("token_count", 0),
                cost_usd=d.get("cost_usd", 0.0),
            ))
        return metas

    async def delete(self, run_id: str) -> None:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -1)
        pipe = self._client.pipeline()
        for step_bytes in steps:
            step = int(step_bytes)
            pipe.delete(self._state_key(run_id, step))
            pipe.delete(self._meta_key(run_id, step))
        pipe.delete(self._steps_key(run_id))
        await pipe.execute()

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -(keep_last + 1))
        if not steps:
            return
        pipe = self._client.pipeline()
        for step_bytes in steps:
            step = int(step_bytes)
            pipe.delete(self._state_key(run_id, step))
            pipe.delete(self._meta_key(run_id, step))
            pipe.zrem(self._steps_key(run_id), str(step))
        await pipe.execute()

    async def aclose(self) -> None:
        await self._client.aclose()
