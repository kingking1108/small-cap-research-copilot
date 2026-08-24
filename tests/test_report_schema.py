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
