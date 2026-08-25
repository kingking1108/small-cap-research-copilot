import re
from dataclasses import dataclass
from typing import Annotated

import yfinance as yf
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from research_copilot.agent.state import AgentState
from research_copilot.config import get_settings
from research_copilot.retrieval.vectorstore import get_known_companies, get_vectorstore


@dataclass(frozen=True)
class WatchlistCompany:
    # A fragment actually present in the ingested PDF's filename stem (e.g.
    # "Annual_report_nagarro_2025_de"), since an LLM-supplied name like
    # "SUSS MicroTec" or "Deutsche Beteiligungs" won't substring-match the
    # stem directly.
    filename_hint: str
    # LLMs reliably guess the wrong Yahoo Finance suffix for European
    # small/mid caps (e.g. "NGR.DE" for Nagarro, which doesn't exist - the
    # real ticker is "NA9.DE"), so it's looked up here instead of trusted.
    ticker: str


# Single source of truth for watchlist companies, keyed by every name/spelling
# an LLM might use to refer to them. Kept as one table (rather than separate
# name->hint and name->ticker dicts) so adding a company can't update one
# lookup and silently forget the other.
WATCHLIST: dict[str, WatchlistCompany] = {
    "nagarro": WatchlistCompany(filename_hint="nagarro", ticker="NA9.DE"),
    "amadeus fire": WatchlistCompany(filename_hint="amadeus", ticker="AAD.DE"),
    "hypoport": WatchlistCompany(filename_hint="hypoport", ticker="HYQ.DE"),
    "suss microtec": WatchlistCompany(filename_hint="suss", ticker="SMHN.DE"),
    "süss microtec": WatchlistCompany(filename_hint="suss", ticker="SMHN.DE"),
    "deutsche beteiligungs": WatchlistCompany(filename_hint="dbag", ticker="DBAN.DE"),
    "dbag": WatchlistCompany(filename_hint="dbag", ticker="DBAN.DE"),
}


def _resolve_company(company: str, known_companies: list[str]) -> str | None:
    normalized = company.strip().lower()
    if not normalized or not known_companies:
        return None
    for known in known_companies:
        known_lower = known.lower()
        if normalized in known_lower or known_lower in normalized:
            return known
    for name, entry in WATCHLIST.items():
        if name in normalized:
            for known in known_companies:
                if entry.filename_hint in known.lower():
                    return known
    return None


_CHUNK_SEPARATOR = "\n\n---\n\n"
_CHUNK_TAG_PREFIX_RE = re.compile(r"^\[Source:[^\]]*\]\n")
# Reworded follow-up queries in the same MAX_TOOL_CALLS-bounded conversation
# often re-rank the same top chunks the agent already saw (and already got
# nothing useful out of) rather than surfacing new ones - padding out the
# context budget with repeats of what's already there. Overfetch a bit past
# retrieval_k so there's enough headroom to drop repeats and still return a
# full page of *new* results.
_DEDUP_OVERFETCH_MULTIPLIER = 2


def _previously_seen_chunks(state: AgentState) -> set[str]:
    """Raw `page_content` of every filings chunk already surfaced by a
    `search_filings` call earlier in this conversation, recovered from past
    ToolMessages by stripping the `[Source: ...]` tag this same tool writes
    (see the return below) - not just source+page, since one page can split
    into several distinct chunks."""
    seen: set[str] = set()
    for message in state.get("messages", []):
        if not isinstance(message, ToolMessage):
            continue
        for block in str(message.content).split(_CHUNK_SEPARATOR):
            match = _CHUNK_TAG_PREFIX_RE.match(block)
            if match:
                seen.add(block[match.end() :])
    return seen


@tool
def search_filings(
    query: str,
    company: str | None = None,
    state: Annotated[AgentState, InjectedState] = None,  # type: ignore[assignment]
) -> str:
    """Search ingested company filings and reports for passages relevant to
    the query. Returns the top matching excerpts, each tagged with its
    source document so answers can be cited. Optionally pass `company` (a
    company name, e.g. 'Nagarro' or 'DBAG') to scope the search to that one
    company's filings and avoid results from other companies in the corpus."""
    settings = get_settings()
    vectorstore = get_vectorstore()
    company_filter = None
    if company:
        resolved = _resolve_company(company, get_known_companies())
        # Unresolved name: fall back to an unfiltered search rather than
        # applying a filter guaranteed to match nothing.
        if resolved is not None:
            company_filter = {"company": resolved}

    seen = _previously_seen_chunks(state) if state else set()
    fetch_k = settings.retrieval_k * _DEDUP_OVERFETCH_MULTIPLIER if seen else settings.retrieval_k
    docs = vectorstore.similarity_search(query, k=fetch_k, filter=company_filter)
    if seen:
        docs = [doc for doc in docs if doc.page_content not in seen]
    docs = docs[: settings.retrieval_k]

    if not docs:
        if seen:
            return (
                "All matching passages for this query were already retrieved "
                "earlier in this conversation - try a different query or "
                "conclude no further information is available."
            )
        return "No matching passages found in the ingested documents."
    return _CHUNK_SEPARATOR.join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"S. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _resolve_ticker(query: str) -> str:
    # Resolve our watchlist companies by name instead of trusting the
    # model's ticker knowledge; anything outside the watchlist still falls
    # through to using the model's input as a literal ticker.
    normalized = query.strip().lower()
    for name, entry in WATCHLIST.items():
        if name in normalized:
            return entry.ticker
    return query


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price and 5-day change (%) for a stock. For
    any of the watchlist companies (Nagarro, Amadeus Fire, Hypoport, SUSS
    MicroTec, Deutsche Beteiligungs/DBAG), always pass the company name
    itself (e.g. 'Nagarro') - never guess a ticker for them yourself, your
    own guesses for these are unreliable (their real Yahoo Finance tickers
    use non-obvious suffixes) and the correct one is looked up
    automatically from the name. Only pass a literal ticker with exchange
    suffix (e.g. 'SAP.DE') for a company outside the watchlist."""
    resolved = _resolve_ticker(ticker)
    source = f"Yahoo Finance ({resolved})"
    history = yf.Ticker(resolved).history(period="5d")
    if history.empty:
        return f"[Source: {source}]\nNo price data found for ticker '{resolved}'."
    latest = history["Close"].iloc[-1]
    change_pct = (history["Close"].iloc[-1] / history["Close"].iloc[0] - 1) * 100
    return (
        f"[Source: {source}]\n"
        f"{resolved}: last close {latest:.2f}, 5-day change {change_pct:+.2f}%"
    )


TOOLS = [search_filings, get_stock_price]
