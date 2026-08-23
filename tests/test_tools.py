from unittest.mock import MagicMock, patch

import pandas as pd

from research_copilot.agent.tools import get_stock_price


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
