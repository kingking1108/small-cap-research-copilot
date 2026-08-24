from langchain_core.messages import ToolMessage

from research_copilot.report.schema import ResearchReport, SourcedClaim
from research_copilot.report.verify import find_unverified_claims


def _tool_message(source: str, page: int = 1, text: str = "some retrieved text") -> ToolMessage:
    return ToolMessage(content=f"[Source: {source}, S. {page}]\n{text}", tool_call_id="1")


def _pageless_tool_message(source: str, text: str = "some retrieved text") -> ToolMessage:
    # get_stock_price's citation tag, unlike search_filings', never carries
    # a page number.
    return ToolMessage(content=f"[Source: {source}]\n{text}", tool_call_id="1")


def test_find_unverified_claims_flags_source_never_retrieved() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[
            SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf", page=1),
            SourcedClaim(claim="Made up fact", source="a_document_never_seen.pdf", page=1),
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
        key_facts=[SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf", page=1)],
    )
    messages = [_tool_message("example_ag_2025.pdf", page=1)]

    assert find_unverified_claims(report, messages) == []


def test_find_unverified_claims_flags_everything_if_no_tools_were_called() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf", page=1)],
    )

    assert len(find_unverified_claims(report, messages=[])) == 1


def test_find_unverified_claims_flags_wrong_page_of_a_seen_document() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[
            SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf", page=99)
        ],
    )
    messages = [_tool_message("example_ag_2025.pdf", page=1)]

    unverified = find_unverified_claims(report, messages)

    assert len(unverified) == 1
    assert unverified[0].claim == "Revenue was 100 EUR"


def test_find_unverified_claims_accepts_claim_without_page() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[SourcedClaim(claim="Revenue was 100 EUR", source="example_ag_2025.pdf")],
    )
    messages = [_tool_message("example_ag_2025.pdf", page=1)]

    assert find_unverified_claims(report, messages) == []


def test_find_unverified_claims_accepts_pageless_source_like_stock_price() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[
            SourcedClaim(claim="Last close was 78.15 EUR", source="Yahoo Finance (NA9.DE)")
        ],
    )
    messages = [_pageless_tool_message("Yahoo Finance (NA9.DE)")]

    assert find_unverified_claims(report, messages) == []


def test_find_unverified_claims_flags_invented_page_on_pageless_source() -> None:
    report = ResearchReport(
        company="Example AG",
        summary="...",
        key_facts=[
            SourcedClaim(
                claim="Last close was 78.15 EUR", source="Yahoo Finance (NA9.DE)", page=1
            )
        ],
    )
    messages = [_pageless_tool_message("Yahoo Finance (NA9.DE)")]

    unverified = find_unverified_claims(report, messages)

    assert len(unverified) == 1
