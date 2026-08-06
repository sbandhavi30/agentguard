# Multi-Agent Orchestration: LangChain + Bedrock + AgentGuard

## Architecture

```
User query
    ↓
Supervisor (Claude Sonnet 4.5 via Bedrock)
    ├── ResearchAgent (Claude Haiku) — search_web, retrieve_from_kb
    ├── CodeAgent     (Claude Haiku) — generate_code, run_code  
    └── DataAgent     (Claude Haiku) — analyze_data, create_visualization
    ↓
Supervisor synthesises → final answer

AgentGuard (DurableGraph) wraps the LangGraph StateGraph:
    - checkpoints before + after graph execution
    - resume from last checkpoint on crash
    - CLI: agentguard list/inspect/resume/stats
```

## Setup

### 1. AWS credentials
```bash
aws configure
# needs: bedrock:InvokeModel permission
```

### 2. Enable models in Bedrock console (Model access)
- `Anthropic Claude 3.5 Haiku`   → anthropic.claude-3-5-haiku-20241022-v1:0
- `Anthropic Claude Sonnet 4.5`  → us.anthropic.claude-sonnet-4-5-20250929-v1:0

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python agents.py
```

## Inspect checkpoints

```bash
# After running
agentguard stats   --store /tmp/agentguard-multiagent-lc/q1
agentguard list    multi-research-001 --store /tmp/agentguard-multiagent-lc/q1
agentguard inspect multi-research-001 --step 0 --store /tmp/agentguard-multiagent-lc/q1
agentguard resume  multi-research-001 --store /tmp/agentguard-multiagent-lc/q1
```

## What AgentGuard adds

| Without AgentGuard | With AgentGuard |
|---|---|
| Graph crashes → restart from scratch | Crash → resume from last checkpoint |
| No visibility into intermediate state | Full per-step metadata on disk |
| No loop protection | max_steps triggers AgentLoopDetectedError |
| No audit trail | meta.jsonl: step, trigger, tokens, timestamp |
