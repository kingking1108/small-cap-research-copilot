from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from research_copilot.report.schema import ResearchReport


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    report: ResearchReport | None
