from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.documents import Document

from research_copilot.agent.tools import _resolve_company, get_stock_price, search_filings


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


KNOWN_COMPANIES = [
    "2026_03_30_SUSS_Geschäftsbericht 2025",
    "260310_DBAG_Geschäftsbericht_2025_web",
    "Amadeus-Fire_AG_Geschaeftsbericht_2025_DE",
    "Annual_report_nagarro_2025_de",
    "Hypoport-2025-Q4-Annual-report-English",
]


def test_resolve_company_direct_substring_match() -> None:
    assert _resolve_company("Nagarro", KNOWN_COMPANIES) == "Annual_report_nagarro_2025_de"
    assert _resolve_company("DBAG", KNOWN_COMPANIES) == "260310_DBAG_Geschäftsbericht_2025_web"
    assert (
        _resolve_company("Hypoport", KNOWN_COMPANIES)
        == "Hypoport-2025-Q4-Annual-report-English"
    )


def test_resolve_company_falls_back_to_name_hints() -> None:
    assert (
        _resolve_company("SUSS MicroTec", KNOWN_COMPANIES)
        == "2026_03_30_SUSS_Geschäftsbericht 2025"
    )
    assert (
        _resolve_company("Deutsche Beteiligungs", KNOWN_COMPANIES)
        == "260310_DBAG_Geschäftsbericht_2025_web"
    )
    assert (
        _resolve_company("Amadeus Fire", KNOWN_COMPANIES)
        == "Amadeus-Fire_AG_Geschaeftsbericht_2025_DE"
    )


def test_resolve_company_returns_none_when_unresolvable() -> None:
    assert _resolve_company("Totally Unknown Corp", KNOWN_COMPANIES) is None
    assert _resolve_company("anything", []) is None


@patch("research_copilot.agent.tools.get_known_companies")
@patch("research_copilot.agent.tools.get_vectorstore")
def test_search_filings_without_company_searches_unfiltered(
    mock_get_vectorstore: MagicMock, mock_get_known_companies: MagicMock
) -> None:
    store = MagicMock()
    store.similarity_search.return_value = [
        Document(page_content="text", metadata={"source": "doc.pdf"})
    ]
    mock_get_vectorstore.return_value = store

    result = search_filings.invoke({"query": "revenue"})

    store.similarity_search.assert_called_once_with("revenue", k=12, filter=None)
    mock_get_known_companies.assert_not_called()
    assert "[Source: doc.pdf, S. ?]" in result
    assert "text" in result


@patch("research_copilot.agent.tools.get_known_companies")
@patch("research_copilot.agent.tools.get_vectorstore")
def test_search_filings_resolves_company_to_stored_metadata_value(
    mock_get_vectorstore: MagicMock, mock_get_known_companies: MagicMock
) -> None:
    store = MagicMock()
    store.similarity_search.return_value = []
    mock_get_vectorstore.return_value = store
    mock_get_known_companies.return_value = KNOWN_COMPANIES

    search_filings.invoke({"query": "revenue", "company": "Nagarro"})

    store.similarity_search.assert_called_once_with(
        "revenue", k=12, filter={"company": "Annual_report_nagarro_2025_de"}
    )


@patch("research_copilot.agent.tools.get_known_companies")
@patch("research_copilot.agent.tools.get_vectorstore")
def test_search_filings_falls_back_unfiltered_when_company_unresolved(
    mock_get_vectorstore: MagicMock, mock_get_known_companies: MagicMock
) -> None:
    store = MagicMock()
    store.similarity_search.return_value = []
    mock_get_vectorstore.return_value = store
    mock_get_known_companies.return_value = KNOWN_COMPANIES

    search_filings.invoke({"query": "revenue", "company": "Some Unrelated Company"})

    store.similarity_search.assert_called_once_with("revenue", k=12, filter=None)


@patch("research_copilot.agent.tools.get_vectorstore")
def test_search_filings_handles_no_matches(mock_get_vectorstore: MagicMock) -> None:
    store = MagicMock()
    store.similarity_search.return_value = []
    mock_get_vectorstore.return_value = store

    result = search_filings.invoke({"query": "nonexistent"})

    assert "No matching passages found" in result
