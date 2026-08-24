import re

from pydantic import BaseModel, Field, field_validator

# The model occasionally copies the whole `[Source: file.pdf, S. 12]` tag
# into `source` instead of just the filename, despite the prompt asking for
# the bare filename - strip the wrapper defensively rather than trusting
# prompt compliance (the hosted model isn't fully deterministic even at
# temperature=0, see README).
_SOURCE_WRAPPER_RE = re.compile(r",\s*S\.\s*\d+\s*$", re.IGNORECASE)


def _strip_source_wrapper(value: str) -> str:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text.lower().startswith("source:"):
        text = text[len("source:") :].strip()
    return _SOURCE_WRAPPER_RE.sub("", text).strip()


class SourcedClaim(BaseModel):
    claim: str
    source: str
    page: int | None = None

    @field_validator("source")
    @classmethod
    def _clean_source(cls, value: str) -> str:
        return _strip_source_wrapper(value)


class ResearchReport(BaseModel):
    """Structured, citation-checked report output for `research-copilot
    report` (see `agent/graph.py::build_report_graph`)."""

    company: str
    summary: str
    key_facts: list[SourcedClaim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
