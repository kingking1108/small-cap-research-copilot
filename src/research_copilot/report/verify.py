import re
import unicodedata

from langchain_core.messages import ToolMessage

from research_copilot.report.schema import ResearchReport, SourcedClaim

# search_filings tags its excerpts `[Source: file.pdf, S. 12]`, but
# get_stock_price's tag has no page at all - `[Source: Yahoo Finance
# (NA9.DE)]` - so the page group must be optional, not just its digits
# optional, or every stock-price citation would silently vanish from
# known_sources and get flagged as unverified regardless of correctness.
_SOURCE_TAG = re.compile(r"\[Source: ([^,\]]+)(?:, S\. (\d+))?\]")


def _normalize(text: str) -> str:
    # Filenames with umlauts can reach us as NFD (macOS filesystem) or NFC
    # (typed text, LLM output) - visually identical, but `==`/`in` fail
    # across forms unless both sides are normalized the same way first.
    return unicodedata.normalize("NFC", text)


def known_sources(messages: list) -> set[tuple[str, int | None]]:
    """Every (source, page) pair the agent's tool calls actually surfaced in
    this conversation, extracted from each tool result's `[Source: ...]`
    tag (see agent/tools.py). `page` is None for tools whose tag never
    carries one (e.g. get_stock_price)."""
    pairs: set[tuple[str, int | None]] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            for source, page in _SOURCE_TAG.findall(str(message.content)):
                pairs.add((_normalize(source), int(page) if page else None))
    return pairs


def find_unverified_claims(report: ResearchReport, messages: list) -> list[SourcedClaim]:
    """Key facts whose cited source doesn't match anything the agent
    actually retrieved - i.e. the model named a document (or a page of a
    document) it never saw. Case-insensitive substring match on the
    filename in both directions since the model sometimes shortens or
    reformats it slightly; if the claim also cites a page, that page must
    match one actually retrieved for the matched source, catching a claim
    that names a real, seen document but a page it never saw."""
    valid = known_sources(messages)
    if not valid:
        return list(report.key_facts)

    def is_verified(claim: SourcedClaim) -> bool:
        cited = _normalize(claim.source).lower()
        matches = [
            (source, page)
            for source, page in valid
            if cited in source.lower() or source.lower() in cited
        ]
        if not matches:
            return False
        if claim.page is None:
            return True
        return any(page == claim.page for _, page in matches)

    return [claim for claim in report.key_facts if not is_verified(claim)]
