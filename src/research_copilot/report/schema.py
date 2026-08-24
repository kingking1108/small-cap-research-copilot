from pydantic import BaseModel, Field


class SourcedClaim(BaseModel):
    claim: str
    source: str
    page: int | None = None


class ResearchReport(BaseModel):
    """Structured, citation-checked report output for `research-copilot
    report` (see `agent/graph.py::build_report_graph`)."""

    company: str
    summary: str
    key_facts: list[SourcedClaim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
