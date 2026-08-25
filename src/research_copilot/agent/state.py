from typing import Annotated, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from research_copilot.report.schema import ResearchReport


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # NotRequired: `ask`/`eval` never set this key at all (only
    # build_report_graph's generate_report node ever populates it) - plain
    # TypedDicts don't enforce required keys at runtime, so this went
    # unnoticed until search_filings started validating the full injected
    # state against this schema (see agent/tools.py::search_filings).
    report: NotRequired[ResearchReport | None]
