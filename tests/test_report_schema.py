from research_copilot.report.schema import SourcedClaim


def test_source_strips_full_citation_tag() -> None:
    claim = SourcedClaim(claim="x", source="[Source: report.pdf, S. 12]", page=12)

    assert claim.source == "report.pdf"


def test_source_strips_bare_source_prefix() -> None:
    claim = SourcedClaim(claim="x", source="Source: report.pdf")

    assert claim.source == "report.pdf"


def test_source_leaves_bare_filename_unchanged() -> None:
    claim = SourcedClaim(claim="x", source="report.pdf")

    assert claim.source == "report.pdf"


def test_page_zero_is_normalized_to_none() -> None:
    claim = SourcedClaim(claim="x", source="Yahoo Finance (NA9.DE)", page=0)

    assert claim.page is None


def test_page_none_stays_none() -> None:
    claim = SourcedClaim(claim="x", source="Yahoo Finance (NA9.DE)")

    assert claim.page is None


def test_valid_page_is_unchanged() -> None:
    claim = SourcedClaim(claim="x", source="report.pdf", page=12)

    assert claim.page == 12
