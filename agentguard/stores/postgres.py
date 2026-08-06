from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState

try:
    import asyncpg
except ImportError as exc:
    raise ImportError("Install agentguard[postgres] to use PostgresStore") from exc

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agentguard_checkpoints (
    run_id      TEXT          NOT NULL,
    step        INTEGER       NOT NULL,
    trigger     TEXT          NOT NULL,
    timestamp   TIMESTAMPTZ   NOT NULL DEFAULT now(),
    framework   TEXT          NOT NULL,
    state       BYTEA         NOT NULL,
    token_count INTEGER       DEFAULT 0,
    cost_usd    NUMERIC(10,6) DEFAULT 0,
    PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_agentguard_run_latest
    ON agentguard_checkpoints(run_id, step DESC);
"""


class PostgresStore(StorageBackend):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)

    async def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Call await store.initialize() before use")
        return self._pool

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agentguard_checkpoints
                    (run_id, step, trigger, timestamp, framework, state, token_count, cost_usd)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (run_id, step) DO UPDATE
                    SET state       = EXCLUDED.state,
                        trigger     = EXCLUDED.trigger,
                        timestamp   = EXCLUDED.timestamp,
                        token_count = EXCLUDED.token_count,
                        cost_usd    = EXCLUDED.cost_usd
                """,
                run_id, step, meta.trigger, meta.timestamp, meta.framework,
                state, meta.token_count, meta.cost_usd,
            )

    async def load_latest(self, run_id: str) -> RestoredState | None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT step, state FROM agentguard_checkpoints "
                "WHERE run_id = $1 ORDER BY step DESC LIMIT 1",
                run_id,
            )
        if row is None:
            return None
        return RestoredState(step=row["step"], state=bytes(row["state"]))

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT run_id, step, trigger, timestamp, framework, token_count, cost_usd "
                "FROM agentguard_checkpoints WHERE run_id = $1 ORDER BY step DESC",
                run_id,
            )
        return [
            CheckpointMeta(
                run_id=r["run_id"],
                step=r["step"],
                trigger=r["trigger"],
                timestamp=r["timestamp"],
                framework=r["framework"],
                token_count=r["token_count"] or 0,
                cost_usd=float(r["cost_usd"] or 0),
            )
            for r in rows
        ]

    async def delete(self, run_id: str) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM agentguard_checkpoints WHERE run_id = $1",
                run_id,
            )

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM agentguard_checkpoints
                WHERE run_id = $1
                  AND step NOT IN (
                      SELECT step FROM agentguard_checkpoints
                      WHERE run_id = $1
                      ORDER BY step DESC LIMIT $2
                  )
                """,
                run_id, keep_last,
            )

    async def aclose(self) -> None:
        if self._pool:
            await self._pool.close()
