"""
Multi-agent orchestration with LangChain + Bedrock + AgentGuard.

Architecture (from AWS docs supervisor/worker pattern):

  User query
      ↓
  Supervisor (Claude Sonnet on Bedrock)
      ├── routes to → ResearchAgent  (Claude Haiku) — web search, knowledge retrieval
      ├── routes to → CodeAgent      (Claude Haiku) — code generation, execution
      └── routes to → DataAgent      (Claude Haiku) — data analysis, summarisation
      ↓
  Supervisor synthesises → final answer

AgentGuard wraps the LangGraph graph via DurableGraph:
  - checkpoints after every node execution
  - detects loops (supervisor routing same agent repeatedly)
  - on crash: resume entire graph from last checkpoint
  - CLI: agentguard list/inspect/resume any run

Deploy:
  pip install -r requirements.txt
  aws configure
  python agents.py
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Literal

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

STORE_DIR = Path("/tmp/agentguard-multiagent-lc")
STORE_DIR.mkdir(exist_ok=True)
SEP = "═" * 64
SUB = "─" * 64

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Supervisor uses a more capable model; workers use cheap/fast Haiku
SUPERVISOR_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
WORKER_MODEL     = "anthropic.claude-3-5-haiku-20241022-v1:0"

# ── AWS setup ────────────────────────────────────────────────────

print("Connecting to AWS...")
try:
    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    print(f"  Account : {identity['Account']}")
    print(f"  Identity: {identity['Arn'].split('/')[-1]}")
    print(f"  Region  : {REGION}\n")
except Exception as e:
    print(f"ERROR: AWS credentials not configured: {e}")
    raise SystemExit(1)

# ── AgentGuard ───────────────────────────────────────────────────

from agentguard.adapters.langgraph import DurableGraph
from agentguard.core.exceptions import AgentLoopDetectedError
from agentguard.stores.disk import DiskStore

# ════════════════════════════════════════════════════════════════
# State
# ════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: str
    task_complete: bool
    research_result: str
    code_result: str
    data_result: str

# ════════════════════════════════════════════════════════════════
# Tools — each worker has its own tool set
# ════════════════════════════════════════════════════════════════

# Research agent tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    # Real implementation would call Tavily/SerpAPI/Bing
    # For demo: return realistic-looking results
    results = {
        "AgentGuard": "AgentGuard v0.1.0 released on GitHub (sbandhavi30/agentguard). Provides durable checkpointing for AI agents via Anthropic, LangGraph, LangChain, and Bedrock adapters.",
        "LangGraph": "LangGraph 0.2.x supports multi-agent supervisor patterns via StateGraph. AWS blog post: 'Build multi-agent systems with LangGraph and Amazon Bedrock' published 2024.",
        "Bedrock": "AWS Bedrock added support for Claude 3.5 Sonnet v2 and Amazon Nova models in late 2024. Converse API now supports cross-region inference.",
        "default": f"Search results for '{query}': Found 5 relevant articles. Top result: comprehensive overview published 2024-Q4.",
    }
    for key in results:
        if key.lower() in query.lower():
            return results[key]
    return results["default"]

@tool
def retrieve_from_kb(topic: str) -> str:
    """Retrieve relevant documentation from the internal knowledge base."""
    kb = {
        "checkpoint": "Checkpoints are saved after every tool call. State = JSON bytes. Trigger reasons: tool_call, token_pressure, destructive_action.",
        "bedrock": "Bedrock Converse API supports Claude, Nova, Llama, Mistral. DurableBedrockLoop wraps boto3 client. boto3 is sync — runs in thread pool.",
        "langgraph": "DurableGraph wraps LangGraph compiled graph. Checkpoints before and after ainvoke(). Resume restores input dict and re-invokes.",
        "default": f"Knowledge base entry for '{topic}': Relevant documentation found covering architecture, configuration, and usage patterns.",
    }
    for key in kb:
        if key.lower() in topic.lower():
            return kb[key]
    return kb["default"]

# Code agent tools
@tool
def generate_code(spec: str, language: str = "python") -> str:
    """Generate code based on a specification."""
    return f"""# Generated {language} code for: {spec}

async def run_with_agentguard(query: str, run_id: str):
    from agentguard.adapters.bedrock import DurableBedrockLoop
    from agentguard.stores.disk import DiskStore
    import boto3

    store = DiskStore(".agentguard")
    loop = DurableBedrockLoop(
        client=boto3.client("bedrock-runtime"),
        store=store,
        max_steps=10,
    )
    return await loop.run(
        messages=[{{"role": "user", "content": [{{"text": query}}]}}],
        run_id=run_id,
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        tools=[],
    )
"""

@tool
def run_code(code: str) -> str:
    """Execute code in a sandbox and return the output."""
    # Real implementation would use E2B or Docker sandbox
    return f"Code executed successfully. Output: [sandbox execution result for {len(code)} chars of code]"

# Data agent tools
@tool
def analyze_data(data: str, analysis_type: str = "summary") -> str:
    """Perform data analysis and return insights."""
    return (
        f"Analysis ({analysis_type}) complete:\n"
        f"  - Input size: {len(data)} characters\n"
        f"  - Key insight 1: Pattern detected in first quartile\n"
        f"  - Key insight 2: Anomaly at position ~{len(data)//2}\n"
        f"  - Recommendation: Further investigation needed on outliers"
    )

@tool
def create_visualization(data: str, chart_type: str = "bar") -> str:
    """Create a data visualization and return a description."""
    return f"Created {chart_type} chart from data. Chart saved to /tmp/chart_{hash(data) % 9999}.png"

# ════════════════════════════════════════════════════════════════
# Worker agents
# ════════════════════════════════════════════════════════════════

def make_worker(name: str, system_prompt: str, tools: list):
    """Build a worker: LLM with bound tools + a node function."""
    llm = ChatBedrockConverse(
        model=WORKER_MODEL,
        region_name=REGION,
        temperature=0,
        max_tokens=1024,
    ).bind_tools(tools)

    tool_map = {t.name: t for t in tools}

    async def node(state: AgentState) -> dict:
        print(f"\n  [{name}] thinking...")
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        response = await llm.ainvoke(messages)

        result_messages = [response]

        # Execute any tool calls
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    print(f"  [{name}] calling tool: {tc['name']}({tc['args']})")
                    output = tool_fn.invoke(tc["args"])
                    result_messages.append(
                        ToolMessage(content=str(output), tool_call_id=tc["id"])
                    )

            # Final synthesis after tool results
            followup = await llm.ainvoke(
                [SystemMessage(content=system_prompt)] + state["messages"] + result_messages
            )
            result_messages.append(followup)

        final = result_messages[-1]
        content = final.content if isinstance(final.content, str) else str(final.content)
        print(f"  [{name}] done: {content[:120]}...")

        # Store result by agent name
        key = f"{name.lower().replace(' ', '_')}_result"
        return {
            "messages": result_messages,
            "active_agent": "",
            key: content,
        }

    return node

research_node = make_worker(
    "ResearchAgent",
    "You are a research specialist. Use search_web and retrieve_from_kb to gather accurate information. Always cite your sources.",
    [search_web, retrieve_from_kb],
)

code_node = make_worker(
    "CodeAgent",
    "You are a software engineer. Generate clean, working code. Use generate_code to create implementations and run_code to test them.",
    [generate_code, run_code],
)

data_node = make_worker(
    "DataAgent",
    "You are a data analyst. Use analyze_data to find insights and create_visualization to present findings clearly.",
    [analyze_data, create_visualization],
)

# ════════════════════════════════════════════════════════════════
# Supervisor
# ════════════════════════════════════════════════════════════════

SUPERVISOR_PROMPT = """You are an orchestrator managing three specialist agents:
- ResearchAgent: web search, knowledge retrieval, fact-finding
- CodeAgent: code generation, execution, technical implementation
- DataAgent: data analysis, visualizations, statistical insights

For each user request:
1. Decide which agent(s) to call and in what order
2. Route to the right agent by responding with JSON: {"next": "ResearchAgent"|"CodeAgent"|"DataAgent"|"FINISH"}
3. After all necessary agents have responded, synthesize their outputs into a final answer and respond with {"next": "FINISH"}

Be efficient — only route to agents that are actually needed for the task."""

supervisor_llm = ChatBedrockConverse(
    model=SUPERVISOR_MODEL,
    region_name=REGION,
    temperature=0,
    max_tokens=512,
)

async def supervisor_node(state: AgentState) -> dict:
    print(f"\n  [Supervisor] routing...")
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = await supervisor_llm.ainvoke(messages)

    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse routing decision
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^}]*"next"[^}]*\}', content)
        if json_match:
            decision = json.loads(json_match.group())
            next_agent = decision.get("next", "FINISH")
        else:
            next_agent = "FINISH"
    except Exception:
        next_agent = "FINISH"

    print(f"  [Supervisor] → {next_agent}")
    return {
        "messages": [response],
        "active_agent": next_agent,
        "task_complete": next_agent == "FINISH",
    }

# ════════════════════════════════════════════════════════════════
# Routing logic
# ════════════════════════════════════════════════════════════════

def route_supervisor(state: AgentState) -> Literal["research_agent", "code_agent", "data_agent", "__end__"]:
    agent = state.get("active_agent", "FINISH")
    mapping = {
        "ResearchAgent": "research_agent",
        "CodeAgent":     "code_agent",
        "DataAgent":     "data_agent",
        "FINISH":        "__end__",
    }
    return mapping.get(agent, "__end__")

# ════════════════════════════════════════════════════════════════
# Build graph
# ════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor",      supervisor_node)
    builder.add_node("research_agent",  research_node)
    builder.add_node("code_agent",      code_node)
    builder.add_node("data_agent",      data_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research_agent": "research_agent",
            "code_agent":     "code_agent",
            "data_agent":     "data_agent",
            "__end__":        END,
        },
    )
    # All workers report back to supervisor after finishing
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("code_agent",     "supervisor")
    builder.add_edge("data_agent",     "supervisor")

    return builder.compile()

# ════════════════════════════════════════════════════════════════
# Run scenarios
# ════════════════════════════════════════════════════════════════

async def run_query(query: str, run_id: str, store_subdir: str):
    graph = build_graph()
    store = DiskStore(base_dir=STORE_DIR / store_subdir)
    dg = DurableGraph(graph=graph, store=store)

    print(f"\nQuery   : {query}")
    print(f"Run ID  : {run_id}")
    print(f"Store   : {STORE_DIR / store_subdir}\n")

    result = await dg.ainvoke(
        {"messages": [HumanMessage(content=query)], "active_agent": "", "task_complete": False,
         "research_result": "", "code_result": "", "data_result": ""},
        run_id=run_id,
    )

    # Print final supervisor message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and "{" not in content[:10]:  # skip pure routing JSON
                print(f"\nFinal answer:\n{content}")
                break

    metas = await store.list(run_id)
    print(f"\n{SUB}")
    print(f"AgentGuard checkpoints: {len(metas)}")
    for m in reversed(metas):
        print(f"  step={m.step}  trigger={m.trigger}  framework={m.framework}")

    print(f"\nCLI inspection:")
    print(f"  agentguard list    {run_id} --store {STORE_DIR / store_subdir}")
    print(f"  agentguard inspect {run_id} --step 0 --store {STORE_DIR / store_subdir}")
    print(f"  agentguard resume  {run_id} --store {STORE_DIR / store_subdir}")

    return result


async def main():
    print(SEP)
    print("Multi-Agent Orchestration: LangChain + Bedrock + AgentGuard")
    print(f"Supervisor: {SUPERVISOR_MODEL}")
    print(f"Workers   : {WORKER_MODEL}")
    print(SEP)

    # Query 1: Research only
    print(f"\n{SEP}")
    print("QUERY 1: Research task → routes to ResearchAgent only")
    print(SEP)
    await run_query(
        query="What is AgentGuard and how does it integrate with AWS Bedrock?",
        run_id="multi-research-001",
        store_subdir="q1",
    )

    # Query 2: Code + Research
    print(f"\n{SEP}")
    print("QUERY 2: Code task → Supervisor routes Research then Code")
    print(SEP)
    await run_query(
        query="Show me how to use AgentGuard with LangGraph and generate working Python code for it.",
        run_id="multi-code-001",
        store_subdir="q2",
    )

    # Query 3: All three agents
    print(f"\n{SEP}")
    print("QUERY 3: Complex task → all three agents involved")
    print(SEP)
    await run_query(
        query=(
            "Research the latest AI agent frameworks, write code to benchmark them, "
            "and analyze which performs best on cost vs latency."
        ),
        run_id="multi-all-001",
        store_subdir="q3",
    )

    print(f"\n{SEP}")
    print("All queries complete.")
    print(f"\nFull CLI inspection:")
    print(f"  agentguard stats --store {STORE_DIR}/q1")
    print(f"  agentguard stats --store {STORE_DIR}/q2")
    print(f"  agentguard stats --store {STORE_DIR}/q3")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
