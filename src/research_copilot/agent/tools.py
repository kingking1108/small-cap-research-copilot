import yfinance as yf
from langchain_core.tools import tool

from research_copilot.config import get_settings
from research_copilot.retrieval.vectorstore import get_known_companies, get_vectorstore

# `company` metadata is a raw filename stem (e.g. "Annual_report_nagarro_2025_de"),
# not a clean name, so an LLM-supplied name like "SUSS MicroTec" or "Deutsche
# Beteiligungs" won't substring-match the stem directly. These hints give the
# resolver a fragment that is actually present in the corresponding filename.
COMPANY_NAME_HINTS: dict[str, str] = {
    "nagarro": "nagarro",
    "amadeus fire": "amadeus",
    "hypoport": "hypoport",
    "suss microtec": "suss",
    "süss microtec": "suss",
    "deutsche beteiligungs": "dbag",
    "dbag": "dbag",
}


def _resolve_company(company: str, known_companies: list[str]) -> str | None:
    normalized = company.strip().lower()
    if not normalized or not known_companies:
        return None
    for known in known_companies:
        known_lower = known.lower()
        if normalized in known_lower or known_lower in normalized:
            return known
    for name, hint in COMPANY_NAME_HINTS.items():
        if name in normalized:
            for known in known_companies:
                if hint in known.lower():
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


# LLMs reliably guess the wrong Yahoo Finance suffix for European small/mid
# caps (e.g. "NGR.DE" for Nagarro, which doesn't exist - the real ticker is
# "NA9.DE"). Resolve our watchlist companies by name instead of trusting the
# model's ticker knowledge; anything outside the watchlist still falls
# through to using the model's input as a literal ticker.
WATCHLIST_TICKERS: dict[str, str] = {
    "nagarro": "NA9.DE",
    "amadeus fire": "AAD.DE",
    "hypoport": "HYQ.DE",
    "suss microtec": "SMHN.DE",
    "süss microtec": "SMHN.DE",
    "deutsche beteiligungs": "DBAN.DE",
    "dbag": "DBAN.DE",
}


def _resolve_ticker(query: str) -> str:
    normalized = query.strip().lower()
    for name, ticker in WATCHLIST_TICKERS.items():
        if name in normalized:
            return ticker
    return query


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price and 5-day change (%) for a stock. Pass
    either a Yahoo Finance ticker with exchange suffix (e.g. 'SAP.DE') or
    just the company name (e.g. 'Nagarro') for watchlist companies, whose
    correct ticker is looked up automatically rather than guessed."""
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
