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


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price and 5-day change (%) for a stock ticker,
    e.g. 'SAP.DE'. Use Yahoo Finance ticker symbols including the exchange
    suffix for European listings."""
    history = yf.Ticker(ticker).history(period="5d")
    if history.empty:
        return f"No price data found for ticker '{ticker}'."
    latest = history["Close"].iloc[-1]
    change_pct = (history["Close"].iloc[-1] / history["Close"].iloc[0] - 1) * 100
    return f"{ticker}: last close {latest:.2f}, 5-day change {change_pct:+.2f}%"


TOOLS = [search_filings, get_stock_price]
