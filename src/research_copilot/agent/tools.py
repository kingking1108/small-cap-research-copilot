from dataclasses import dataclass

import yfinance as yf
from langchain_core.tools import tool

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


@tool
def search_filings(query: str, company: str | None = None) -> str:
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
    docs = vectorstore.similarity_search(query, k=settings.retrieval_k, filter=company_filter)
    if not docs:
        return "No matching passages found in the ingested documents."
    return "\n\n---\n\n".join(
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
