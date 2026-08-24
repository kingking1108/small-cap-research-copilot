from collections.abc import Callable

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from research_copilot.agent.state import AgentState
from research_copilot.agent.tools import TOOLS
from research_copilot.llm import get_chat_model
from research_copilot.report.schema import ResearchReport

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

REPORT_PROMPT = (
    "Based on the preceding conversation, write a structured research "
    "report. Every entry in key_facts must be a specific, verifiable claim "
    "with the exact source document filename it came from, taken only from "
    "the tool results above — never invent a fact or a source. Do not add "
    "claims, context, or inferences that are not explicitly stated in the "
    "conversation above (e.g. do not guess an industry or sector from the "
    "company's name). List anything the conversation could not establish "
    "under open_questions instead of guessing."
)

# The system prompt above asks the model to stop after ~2 searches, but that
# is only a suggestion the model doesn't reliably follow (observed 3-11
# tool-call rounds on the identical question across runs). Enforce a real
# ceiling in code: once it's hit, invoke the model *without* tools bound so
# it structurally cannot emit another tool_call and must synthesize a final
# answer from whatever was already retrieved.
MAX_TOOL_CALLS = 3


def _build_call_model() -> Callable[[AgentState], dict]:
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

    return call_model


def build_agent_graph() -> CompiledStateGraph:
    """Conversational agent: answers in plain, cited prose. Used by `ask`
    and the eval harness."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _build_call_model())
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_report_graph() -> CompiledStateGraph:
    """Same agent/tools research loop as build_agent_graph, but once the
    agent is done gathering information, a final node synthesizes a
    structured, citation-checked ResearchReport instead of ending on plain
    prose. Used by `research-copilot report`."""
    structured_llm = get_chat_model().with_structured_output(ResearchReport)

    def generate_report(state: AgentState) -> dict:
        messages = [SystemMessage(content=REPORT_PROMPT), *state["messages"]]
        report = structured_llm.invoke(messages)
        return {"report": report}

    graph = StateGraph(AgentState)
    graph.add_node("agent", _build_call_model())
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("generate_report", generate_report)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "generate_report"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("generate_report", END)
    return graph.compile()
