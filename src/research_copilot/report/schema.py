from pydantic import BaseModel, Field


class SourcedClaim(BaseModel):
    claim: str
    source: str


class ResearchReport(BaseModel):
    """Structured output for the Day 5 report-generation step.

    TODO (Day 5): wire this into the agent graph, e.g. via a final node that
    calls `llm.with_structured_output(ResearchReport)` on the conversation.
    """

    company: str
    summary: str
    key_facts: list[SourcedClaim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
