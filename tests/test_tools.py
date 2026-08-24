from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.documents import Document

from research_copilot.agent.tools import get_stock_price, search_filings


@patch("research_copilot.agent.tools.get_vectorstore")
def test_search_filings_includes_page_number_in_source_tag(mock_get_vectorstore: MagicMock) -> None:
    mock_get_vectorstore.return_value.similarity_search.return_value = [
        Document(
            page_content="Umsatz stieg um 10%.",
            metadata={"source": "report.pdf", "page": 12},
        ),
    ]

    result = search_filings.invoke({"query": "Umsatz"})

    assert "report.pdf" in result
    assert "12" in result


@patch("research_copilot.agent.tools.yf.Ticker")
def test_get_stock_price_formats_change(mock_ticker: MagicMock) -> None:
    history = pd.DataFrame({"Close": [100.0, 102.0, 101.0, 103.0, 105.0]})
    mock_ticker.return_value.history.return_value = history

    result = get_stock_price.invoke({"ticker": "TEST.DE"})

    assert "TEST.DE" in result
    assert "+5.00%" in result


@patch("research_copilot.agent.tools.yf.Ticker")
def test_get_stock_price_handles_missing_data(mock_ticker: MagicMock) -> None:
    mock_ticker.return_value.history.return_value = pd.DataFrame()

    result = get_stock_price.invoke({"ticker": "UNKNOWN"})

    assert "No price data" in result
