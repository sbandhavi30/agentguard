import asyncio
import json
from pathlib import Path

import typer

from agentguard.stores.disk import DiskStore

app = typer.Typer(help="AgentGuard — inspect and manage agent checkpoints")

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


@app.command()
def list(
    run_id: str,
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """List all checkpoints for a run."""
    s = _get_store(store)
    metas = asyncio.run(s.list(run_id))
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
    metas = asyncio.run(s.list(run_id))
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
    metas = asyncio.run(s.list(run_id))
    if not metas:
        typer.echo(f"No checkpoints found for run_id={run_id}")
        raise typer.Exit(code=1)
    latest = metas[0]
    snippet = _RESUME_SNIPPETS.get(
        latest.framework,
        f"# Unknown framework: {latest.framework}\nawait adapter.resume('{run_id}')",
    ).format(run_id=run_id)
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
    asyncio.run(s.prune(run_id, keep_last=keep))
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
    asyncio.run(s.delete(run_id))
    typer.echo(f"Deleted all checkpoints for run_id={run_id}")


@app.command()
def stats(
    store: str = typer.Option(
        str(Path.home() / ".agentguard"), help="Path to disk store"
    ),
) -> None:
    """Show aggregate stats across all runs."""
    base = Path(store)
    if not base.exists():
        typer.echo("No checkpoint store found.")
        return
    total_runs = 0
    total_checkpoints = 0
    total_cost = 0.0
    triggers: dict[str, int] = {}
    s = _get_store(store)
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        total_runs += 1
        metas = asyncio.run(s.list(run_dir.name))
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
