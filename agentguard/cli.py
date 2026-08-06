import asyncio
import concurrent.futures
import json
from pathlib import Path

import typer

from agentguard.stores.disk import DiskStore

app = typer.Typer(help="AgentGuard — inspect and manage agent checkpoints")


def _run_async(coro):
    """Run a coroutine, even if called from within a running event loop.

    asyncio.run() raises RuntimeError when an event loop is already running
    (e.g. during pytest-asyncio tests). To keep CLI commands safe in all
    contexts, offload to a dedicated thread that owns its own event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Running inside an existing event loop (e.g. pytest-asyncio): use a
        # background thread with its own event loop so asyncio.run() works.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, coro)
            return fut.result()
    else:
        return asyncio.run(coro)

_RESUME_SNIPPETS = {
    "anthropic": (
        "from agentguard.adapters.anthropic import DurableAgentLoop\n"
        "loop = DurableAgentLoop(client=client, store=store)\n"
        'result = await loop.resume("{run_id}", model=model, tools=tools)'
    ),
    "langgraph": (
        "from agentguard.adapters.langgraph import DurableGraph\n"
        "app = DurableGraph(graph=graph, store=store)\n"
        'result = await app.resume("{run_id}")'
    ),
    "langchain": (
        "from agentguard.adapters.langchain import DurableExecutor\n"
        "chain = DurableExecutor(executor=executor, store=store)\n"
        'result = await chain.resume("{run_id}")'
    ),
}


def _get_store(store_path: str) -> DiskStore:
    return DiskStore(base_dir=Path(store_path))


@app.command(name="list")
def list_checkpoints(
    run_id: str,
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """List all checkpoints for a run."""
    s = _get_store(store)
    metas = _run_async(s.list(run_id))
    if not metas:
        typer.echo(f"No checkpoints found for run_id={run_id}")
        return
    typer.echo(f"Checkpoints for run_id={run_id}:")
    typer.echo(f"{'step':<8} {'trigger':<22} {'timestamp':<30} {'tokens':<8} {'cost'}")
    for m in metas:
        typer.echo(
            f"{m.step:<8} {m.trigger:<22} {m.timestamp.isoformat():<30} "
            f"{m.token_count:<8} ${m.cost_usd:.4f}"
        )


@app.command()
def inspect(
    run_id: str,
    step: int = typer.Option(..., help="Step number to inspect"),
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Inspect metadata for a specific checkpoint."""
    s = _get_store(store)
    metas = _run_async(s.list(run_id))
    match = next((m for m in metas if m.step == step), None)
    if match is None:
        typer.echo(f"No checkpoint at step={step} for run_id={run_id}")
        raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {
                "run_id": match.run_id,
                "step": match.step,
                "trigger": match.trigger,
                "timestamp": match.timestamp.isoformat(),
                "framework": match.framework,
                "token_count": match.token_count,
                "cost_usd": match.cost_usd,
            },
            indent=2,
        )
    )


@app.command()
def resume(
    run_id: str,
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Print resume code snippet for a run (auto-detects framework)."""
    s = _get_store(store)
    metas = _run_async(s.list(run_id))
    if not metas:
        typer.echo(f"No checkpoints found for run_id={run_id}")
        raise typer.Exit(code=1)
    # list() returns descending by step; metas[0] is latest
    latest = metas[0]
    template = _RESUME_SNIPPETS.get(
        latest.framework,
        "# Unknown framework: {framework}\nawait adapter.resume('{run_id}')",
    )
    # Substitute {framework} before {run_id} so a run_id that literally contains
    # "{framework}" cannot collide with the framework placeholder.
    snippet = template.replace("{framework}", latest.framework).replace("{run_id}", run_id)
    typer.echo(f"Detected framework: {latest.framework}")
    typer.echo(f"Latest checkpoint: step {latest.step} ({latest.timestamp.isoformat()})")
    typer.echo("\nResume with:\n")
    typer.echo(snippet)


@app.command()
def prune(
    run_id: str,
    keep: int = typer.Option(3, help="Number of checkpoints to keep"),
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Prune old checkpoints, keeping the last N."""
    s = _get_store(store)
    _run_async(s.prune(run_id, keep_last=keep))
    typer.echo(f"Pruned checkpoints for {run_id}, kept last {keep}.")


@app.command()
def delete(
    run_id: str,
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Delete all checkpoints for a run."""
    s = _get_store(store)
    _run_async(s.delete(run_id))
    typer.echo(f"Deleted all checkpoints for run_id={run_id}")


async def _stats_async(store: str) -> None:
    """Async implementation of stats — gathers all list() calls without nested _run_async()."""
    base = Path(store)
    if not base.exists():
        typer.echo("No checkpoint store found.")
        return
    s = _get_store(store)
    run_dirs = [d for d in base.iterdir() if d.is_dir()]
    all_metas = await asyncio.gather(*(s.list(d.name) for d in run_dirs))
    total_runs = len(run_dirs)
    total_checkpoints = 0
    total_cost = 0.0
    triggers: dict[str, int] = {}
    for metas in all_metas:
        total_checkpoints += len(metas)
        for m in metas:
            total_cost += m.cost_usd
            triggers[m.trigger] = triggers.get(m.trigger, 0) + 1
    typer.echo(f"Total runs:        {total_runs}")
    typer.echo(f"Total checkpoints: {total_checkpoints}")
    typer.echo(f"Total cost:        ${total_cost:.4f}")
    if triggers:
        most_common = max(triggers, key=lambda k: triggers[k])
        typer.echo(f"Most common trigger: {most_common} ({triggers[most_common]})")


@app.command()
def stats(
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Show aggregate stats across all runs."""
    _run_async(_stats_async(store))


_MAX_STEPS_GUIDE = [
    {
        "use_case": "Simple Q&A / single lookup",
        "recommended": 3,
        "reasoning": "One tool call to fetch, one to return. >3 = stuck.",
    },
    {
        "use_case": "Research / web search loop",
        "recommended": 10,
        "reasoning": "Multiple searches + synthesis. Real agents finish in 4-6.",
    },
    {
        "use_case": "Code generation + test loop",
        "recommended": 15,
        "reasoning": "Write → run tests → fix → re-run. Each iteration = 2-3 steps.",
    },
    {
        "use_case": "Multi-step data pipeline",
        "recommended": 20,
        "reasoning": "Fetch → transform → validate → load, with retries.",
    },
    {
        "use_case": "Autonomous agent (open-ended)",
        "recommended": 50,
        "reasoning": "Complex tasks with many sub-tasks. Monitor via checkpoints.",
    },
    {
        "use_case": "Destructive / DB operations",
        "recommended": 5,
        "reasoning": "Should be short and surgical. Loop = something went wrong.",
    },
]


@app.command(name="max-steps")
def max_steps_guide(
    use_case: str = typer.Argument(
        None,
        help="Filter by keyword (e.g. 'research', 'code', 'db'). Shows all if omitted.",
    ),
) -> None:
    """Show recommended max_steps values for common agent use cases.

    When an agent exceeds max_steps, AgentLoopDetectedError is raised AFTER
    checkpointing — state is always saved before the raise so you can resume
    with a corrective instruction injected into the conversation.
    """
    rows = _MAX_STEPS_GUIDE
    if use_case:
        kw = use_case.lower()
        rows = [r for r in rows if kw in r["use_case"].lower() or kw in r["reasoning"].lower()]
        if not rows:
            typer.echo(f"No match for '{use_case}'. Showing all:\n")
            rows = _MAX_STEPS_GUIDE

    typer.echo(f"\n{'Use case':<35} {'max_steps':>10}   Reasoning")
    typer.echo("─" * 78)
    for r in rows:
        typer.echo(f"{r['use_case']:<35} {r['recommended']:>10}   {r['reasoning']}")
    typer.echo()
    typer.echo("Rule of thumb:")
    typer.echo("  Start conservative (5-10). Raise only if agent legitimately needs more steps.")
    typer.echo("  Check what triggered checkpoints:  agentguard list <run_id>")
    typer.echo("  If token_pressure fires before loop detection → context too large, lower max_steps.")
    typer.echo("  Combine with TriggerPolicy(destructive_tools=[...]) for safety on destructive ops.")
