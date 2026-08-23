from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from research_copilot.agent.state import AgentState
from research_copilot.agent.tools import TOOLS
from research_copilot.llm import get_chat_model

SYSTEM_PROMPT = (
    "You are a financial research assistant for European small- and mid-cap "
    "equities and convertible bonds. Answer only using facts you can support "
    "with the `search_filings` or `get_stock_price` tools. Always cite the "
    "source document for any claim drawn from filings. If the ingested "
    "documents do not contain the answer, say so explicitly instead of "
    "guessing."
)


def build_agent_graph() -> CompiledStateGraph:
    llm_with_tools = get_chat_model().bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
