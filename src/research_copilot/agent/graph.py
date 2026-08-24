from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from research_copilot.agent.state import AgentState
from research_copilot.agent.tools import TOOLS
from research_copilot.llm import get_chat_model

SYSTEM_PROMPT = (
    "You are a financial research assistant for European small- and mid-cap "
    "equities. Answer only using facts you can support with the "
    "`search_filings` or `get_stock_price` tools. Always cite the source "
    "document for any claim drawn from filings. If a search does not return "
    "relevant information after at most two attempts, stop searching and "
    "tell the user the information is not available in the ingested "
    "documents — do not keep retrying with reworded queries. Never guess or "
    "state a fact you cannot support with a tool result."
)

# The system prompt above asks the model to stop after ~2 searches, but that
# is only a suggestion the model doesn't reliably follow (observed 3-11
# tool-call rounds on the identical question across runs). Enforce a real
# ceiling in code: once it's hit, invoke the model *without* tools bound so
# it structurally cannot emit another tool_call and must synthesize a final
# answer from whatever was already retrieved.
MAX_TOOL_CALLS = 3


def build_agent_graph() -> CompiledStateGraph:
    llm_with_tools = get_chat_model().bind_tools(TOOLS)
    llm_without_tools = get_chat_model()

    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        tool_call_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        if tool_call_count >= MAX_TOOL_CALLS:
            messages = [
                *messages,
                SystemMessage(
                    content=(
                        "You have reached the search limit. Answer now using only "
                        "what you already found, or state clearly that the "
                        "information is not available in the ingested documents. "
                        "Do not call any more tools."
                    )
                ),
            ]
            return {"messages": [llm_without_tools.invoke(messages)]}

        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
