import re
import unicodedata

from langchain_core.messages import ToolMessage

from research_copilot.report.schema import ResearchReport, SourcedClaim

_SOURCE_TAG = re.compile(r"\[Source: ([^\]]+)\]")


def _normalize(text: str) -> str:
    # Filenames with umlauts can reach us as NFD (macOS filesystem) or NFC
    # (typed text, LLM output) - visually identical, but `==`/`in` fail
    # across forms unless both sides are normalized the same way first.
    return unicodedata.normalize("NFC", text)


def known_sources(messages: list) -> set[str]:
    """Every filename the agent's tool calls actually surfaced in this
    conversation, extracted from the `[Source: ...]` tags search_filings
    attaches to each retrieved chunk (see agent/tools.py)."""
    sources: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            sources.update(_SOURCE_TAG.findall(str(message.content)))
    return {_normalize(source) for source in sources}


def find_unverified_claims(report: ResearchReport, messages: list) -> list[SourcedClaim]:
    """Key facts whose cited source doesn't match anything the agent
    actually retrieved - i.e. the model named a document it never saw.
    Case-insensitive substring match in both directions since the model
    sometimes shortens or reformats the filename slightly."""
    valid = known_sources(messages)
    if not valid:
        return list(report.key_facts)

    def is_verified(claim: SourcedClaim) -> bool:
        cited = _normalize(claim.source).lower()
        return any(cited in source.lower() or source.lower() in cited for source in valid)

    return [claim for claim in report.key_facts if not is_verified(claim)]
