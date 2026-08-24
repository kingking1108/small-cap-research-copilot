from collections.abc import Callable

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from research_copilot.agent.state import AgentState
from research_copilot.agent.tools import TOOLS
from research_copilot.llm import get_chat_model
from research_copilot.report.schema import ResearchReport

LANGUAGE_INSTRUCTION = (
    "Always answer in German (Deutsch), regardless of the language the "
    "question was asked in. Company names, tickers, and direct quotes from "
    "source documents may stay in their original language."
)

SYSTEM_PROMPT = (
    "You are a financial research assistant for European small- and mid-cap "
    "equities. Answer only using facts you can support with the "
    "`search_filings` or `get_stock_price` tools. Always cite the source "
    "document for any claim drawn from filings, in the exact form "
    "'(Quelle: <Dateiname>, S. <Seite>)', copying the filename and page "
    "number verbatim from the `[Source: ...]` tag in the tool result that "
    "supports the claim — never invent your own citation marker or format. "
    "If a search does not return relevant information after at most two "
    "attempts, stop searching and tell the user the information is not "
    "available in the ingested documents — do not keep retrying with "
    "reworded queries. Never guess or state a fact you cannot support with "
    f"a tool result. {LANGUAGE_INSTRUCTION}"
)

REPORT_PROMPT = (
    "Based on the preceding conversation, write a structured research "
    "report. Every entry in key_facts must be a specific, verifiable claim, "
    "taken only from the tool results above, never invented. Each tool "
    "result tags its excerpts with `[Source: <Dateiname>, S. <Seite>]` — put "
    "ONLY the bare filename (e.g. 'report.pdf') into the `source` field and "
    "ONLY the bare page number (e.g. 12) into the `page` field, both copied "
    "verbatim from that tag; `source` must never itself contain the words "
    "'Source:', brackets, or a page number — that belongs in `page` alone. "
    "Some tools (e.g. get_stock_price) tag their result with no page at "
    "all, e.g. `[Source: Yahoo Finance (NA9.DE)]` — for those, leave "
    "`page` unset/null; never write 0 or invent a page number that isn't "
    "in the tag. Each claim must state exactly "
    "ONE number for exactly ONE period, written as a full, self-contained "
    "sentence naming both the metric and the period/date it applies to "
    "(e.g. 'Der Umsatz betrug 2025 363,6 Mio. EUR.'). Tool results "
    "sometimes contain raw table excerpts where a row's columns (e.g. "
    "several years of the same metric) run together as a bare sequence of "
    "numbers with no clear separators — never copy such a sequence, or "
    "more than one number, into a single claim; if you cannot confidently "
    "tell which single number belongs to which single period, leave that "
    "table out of key_facts entirely rather than guessing or dumping the "
    "raw row. Do not add "
    "claims, context, or inferences that are not explicitly stated in the "
    "conversation above. In particular, never infer what a company does — "
    "its industry, sector, or line of business — from words in its own "
    "name; a company's name is not a reliable description of its actual "
    "business (e.g. a company named 'Amadeus Fire' is not necessarily in "
    "fire safety just because 'Fire' is in the name) — only state what the "
    "retrieved tool results explicitly say the company does, and if they "
    "don't describe its business, leave that out entirely rather than "
    "guessing from the name. List anything the conversation could not "
    "establish under open_questions instead of guessing. "
    f"{LANGUAGE_INSTRUCTION} This applies to the summary, key_facts claims, "
    "and open_questions fields alike."
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
                        f"Do not call any more tools. {LANGUAGE_INSTRUCTION}"
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
