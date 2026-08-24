import yfinance as yf
from langchain_core.tools import tool

from research_copilot.config import get_settings
from research_copilot.retrieval.vectorstore import get_vectorstore


@tool
def search_filings(query: str) -> str:
    """Search ingested company filings and reports for passages relevant to
    the query. Returns the top matching excerpts, each tagged with its
    source document so answers can be cited."""
    settings = get_settings()
    docs = get_vectorstore().similarity_search(query, k=settings.retrieval_k)
    if not docs:
        return "No matching passages found in the ingested documents."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
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
    history = yf.Ticker(resolved).history(period="5d")
    if history.empty:
        return f"No price data found for ticker '{resolved}'."
    latest = history["Close"].iloc[-1]
    change_pct = (history["Close"].iloc[-1] / history["Close"].iloc[0] - 1) * 100
    return f"{resolved}: last close {latest:.2f}, 5-day change {change_pct:+.2f}%"


TOOLS = [search_filings, get_stock_price]
