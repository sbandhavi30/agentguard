# AgentGuard

Durable execution and checkpointing for AI agents. AgentGuard sits between your agent framework and its storage — saving state after every tool call, on token pressure, and before destructive actions — so agents can survive crashes and resume from where they left off.

```
Your agent code
      │
  AgentGuard          ← wraps the run loop; checkpoints state
      │
Framework (Anthropic SDK / LangGraph / LangChain)
      │
Storage (Disk / Redis / Postgres)
```

## Why

AI agents fail mid-run. A network blip, an OOM, a context-window overflow — the agent dies and restarts from zero. AgentGuard checkpoints state after each tool call so you can resume from the last safe point instead of replaying the whole run.

Key properties:

- **Fail-safe checkpoints** — storage errors are swallowed; the agent never crashes because a checkpoint failed
- **Fail-loud restores** — missing or corrupted checkpoints always raise `RestoreError`; no silent data loss
- **Smart triggers** — checkpoint on every tool call, on token pressure (configurable threshold), or when a destructive tool (`delete_*`, `drop_*`, …) is about to run
- **Byte-opaque core** — the engine never inspects state; adapters own serialization
- **Zero core dependencies** — `core/` and `stores/disk.py` use stdlib only

## Install

```bash
# Core + disk store (zero extra deps)
pip install agentguard

# With CLI
pip install "agentguard[cli]"

# Pick your framework
pip install "agentguard[anthropic]"
pip install "agentguard[langgraph]"
pip install "agentguard[langchain]"

# Pick your storage backend
pip install "agentguard[redis]"
pip install "agentguard[postgres]"

# Everything
pip install "agentguard[all]"
```

## Quickstart

### Anthropic SDK

```python
from agentguard.adapters.anthropic import DurableAgentLoop
from agentguard.stores.disk import DiskStore
import anthropic

client = anthropic.AsyncAnthropic()
store = DiskStore(".agentguard")
loop = DurableAgentLoop(client=client, store=store)

# Run — checkpoints after every tool call
result = await loop.run(
    messages=[{"role": "user", "content": "Research quantum computing trends"}],
    run_id="run-001",
    model="claude-opus-4-5",
    tools=[...],
    tool_executor=my_tool_executor,
)

# If it crashes mid-run, resume from last checkpoint
result = await loop.resume("run-001", model="claude-opus-4-5", tools=[...])
```

### LangGraph

```python
from agentguard.adapters.langgraph import DurableGraph
from agentguard.stores.disk import DiskStore

store = DiskStore(".agentguard")
dg = DurableGraph(graph=my_compiled_graph, store=store)

result = await dg.ainvoke({"question": "..."}, run_id="run-002")

# Resume
result = await dg.resume("run-002")
```

### LangChain

```python
from agentguard.adapters.langchain import DurableExecutor
from agentguard.stores.disk import DiskStore

store = DiskStore(".agentguard")
de = DurableExecutor(executor=my_agent_executor, store=store)

result = await de.ainvoke({"input": "..."}, run_id="run-003")

# Resume
result = await de.resume("run-003")
```

## Storage Backends

### Disk (default, zero deps)

```python
from agentguard.stores.disk import DiskStore
store = DiskStore(base_dir=".agentguard")
```

Stores each run in its own directory: `step_00001.bin` (state bytes) + `meta.jsonl` (append-only metadata log).

### Redis

```python
from agentguard.stores.redis import RedisStore
store = RedisStore(url="redis://localhost:6379", ttl_seconds=604800)  # 7-day TTL
await store.aclose()
```

### Postgres

```python
from agentguard.stores.postgres import PostgresStore
store = PostgresStore(dsn="postgresql://user:pass@localhost/db")
await store.initialize()   # creates table if not exists
await store.aclose()
```

## Trigger Policy

Control when checkpoints fire:

```python
from agentguard.core.triggers import TriggerPolicy

policy = TriggerPolicy(
    # Checkpoint before any tool matching these patterns
    destructive_tools=["delete_*", "drop_*", "rm_*", "overwrite_*"],

    # Checkpoint when token usage crosses this fraction of the context budget
    token_pressure_threshold=0.8,   # default: 0.8 (80%)
)

loop = DurableAgentLoop(client=client, store=store, policy=policy)
```

Every tool call also triggers a checkpoint by default (trigger reason `"tool_call"`).

## Checkpoint Failure Callback

```python
def on_checkpoint_failure(run_id: str, step: int, exc: Exception) -> None:
    logger.warning(f"checkpoint failed run={run_id} step={step}: {exc}")
    metrics.increment("agentguard.checkpoint.failure")

loop = DurableAgentLoop(
    client=client,
    store=store,
    on_checkpoint_failure=on_checkpoint_failure,
)
```

The agent continues running even if the callback is set and a checkpoint fails. The callback is for observability only.

## CLI

```bash
# Requires: pip install "agentguard[cli]"

agentguard list <run_id>              # list all checkpoints
agentguard inspect <run_id> --step N  # show checkpoint metadata as JSON
agentguard resume <run_id>            # print resume code snippet (auto-detects framework)
agentguard prune <run_id> --keep 3    # keep last N checkpoints
agentguard delete <run_id>            # delete all checkpoints for a run
agentguard stats                      # aggregate stats across all runs
```

Default store location: `~/.agentguard/`. Override with `--store /path/to/store`.

Example:

```
$ agentguard list run-001 --store .agentguard
Checkpoints for run_id=run-001:
step     trigger          timestamp                        tokens   cost
2        tool_call        2026-08-05T14:22:11+00:00        12400    $0.0124
1        token_pressure   2026-08-05T14:21:55+00:00        162000   $0.1620
0        tool_call        2026-08-05T14:21:30+00:00        8200     $0.0082

$ agentguard resume run-001 --store .agentguard
Detected framework: anthropic
Latest checkpoint: step 2 (2026-08-05T14:22:11+00:00)

Resume with:

from agentguard.adapters.anthropic import DurableAgentLoop
loop = DurableAgentLoop(client=client, store=store)
result = await loop.resume("run-001", model=model, tools=tools)
```

## Error Handling

```python
from agentguard.core.exceptions import (
    AgentGuardError,          # base
    CheckpointWriteError,     # storage failed to save (never raised to agent)
    RestoreError,             # no checkpoint found or corrupted bytes
    DeserializationError,     # adapter failed to deserialize state
    BackendConnectionError,   # backend unreachable
)
```

Rule: **checkpoint failures are swallowed** (agent keeps running). **Restore failures always raise** (no silent data loss on resume).

## Architecture

```
agentguard/
├── core/
│   ├── types.py        # CheckpointMeta, RestoredState (frozen dataclasses)
│   ├── exceptions.py   # AgentGuardError hierarchy
│   ├── store.py        # StorageBackend ABC
│   ├── engine.py       # CheckpointEngine (fail-safe write, fail-loud restore)
│   └── triggers.py     # TriggerPolicy (wildcard matching + token pressure)
├── stores/
│   ├── memory.py       # InMemoryStore (testing)
│   ├── disk.py         # DiskStore (meta.jsonl + step bins)
│   ├── redis.py        # RedisStore (sorted set + TTL)
│   └── postgres.py     # PostgresStore (BYTEA + compound PK)
├── adapters/
│   ├── anthropic.py    # DurableAgentLoop
│   ├── langgraph.py    # DurableGraph
│   └── langchain.py    # DurableExecutor
└── cli.py              # typer CLI
```

`core/` has zero runtime dependencies. Each adapter and store pulls only what it needs via optional extras.

## Testing

```bash
pip install -e ".[dev]"

# Core + adapters + CLI (no external services needed)
pytest tests/core/ tests/stores/test_memory.py tests/stores/test_disk.py tests/adapters/ tests/test_cli.py -v

# Redis (requires Docker)
pytest tests/stores/test_redis.py -v

# Postgres (requires Docker)
pytest tests/stores/test_postgres.py -v
```

Redis and Postgres tests use [testcontainers](https://testcontainers-python.readthedocs.io/) — Docker must be running.

## License

MIT
