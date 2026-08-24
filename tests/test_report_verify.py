from langchain_core.messages import ToolMessage

from research_copilot.report.schema import ResearchReport, SourcedClaim
from research_copilot.report.verify import find_unverified_claims


def _tool_message(source: str, text: str = "some retrieved text") -> ToolMessage:
    return ToolMessage(content=f"[Source: {source}]\n{text}", tool_call_id="1")


def test_find_unverified_claims_flags_source_never_retrieved() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[
            SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf"),
            SourcedClaim(claim="Made up fact", source="a_document_never_seen.pdf"),
        ],
    )
    messages = [_tool_message("example_ag_2025.pdf")]

    unverified = find_unverified_claims(report, messages)

    assert len(unverified) == 1
    assert unverified[0].claim == "Made up fact"


def test_find_unverified_claims_accepts_all_when_sources_match() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf")],
    )
    messages = [_tool_message("example_ag_2025.pdf")]

    assert find_unverified_claims(report, messages) == []


def test_find_unverified_claims_flags_everything_if_no_tools_were_called() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf")],
    )

    assert len(find_unverified_claims(report, messages=[])) == 1
