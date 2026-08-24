import unicodedata
from pathlib import Path
from unittest.mock import MagicMock

from pypdf import PdfWriter

from research_copilot.ingestion.loaders import (
    _extract_page_content,
    _looks_like_table,
    _table_to_markdown,
    load_pdf,
    load_watchlist,
)


def _write_blank_pdf(path: Path, num_pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)


def test_load_pdf_normalizes_source_to_nfc(tmp_path: Path) -> None:
    nfd_name = unicodedata.normalize("NFD", "Bericht_Übersicht.pdf")
    nfc_name = unicodedata.normalize("NFC", nfd_name)
    assert nfd_name != nfc_name  # sanity check that the two forms differ as strings

    pdf_path = tmp_path / nfd_name
    _write_blank_pdf(pdf_path)

    docs = load_pdf(pdf_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == nfc_name
    assert docs[0].metadata["company"] == Path(nfc_name).stem
    assert docs[0].metadata["page"] == 1


def test_load_pdf_returns_one_document_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "multi_page.pdf"
    _write_blank_pdf(pdf_path, num_pages=3)

    docs = load_pdf(pdf_path)

    assert [doc.metadata["page"] for doc in docs] == [1, 2, 3]
    assert all(doc.metadata["source"] == "multi_page.pdf" for doc in docs)


def test_load_watchlist_only_picks_up_pdfs(tmp_path: Path) -> None:
    _write_blank_pdf(tmp_path / "company_a.pdf")
    _write_blank_pdf(tmp_path / "company_b.pdf")
    (tmp_path / "notes.txt").write_text("not a pdf")

    docs = load_watchlist(tmp_path)

    sources = {doc.metadata["source"] for doc in docs}
    assert sources == {"company_a.pdf", "company_b.pdf"}


def test_table_to_markdown_renders_header_and_rows() -> None:
    rows = [["Jahr", "Umsatz"], ["2025", "363.576"], ["2024", "436.906"]]

    assert _table_to_markdown(rows) == (
        "| Jahr | Umsatz |\n| --- | --- |\n| 2025 | 363.576 |\n| 2024 | 436.906 |"
    )


def test_table_to_markdown_handles_none_cells() -> None:
    assert _table_to_markdown([["A", None], ["1", "2"]]) == "| A |  |\n| --- | --- |\n| 1 | 2 |"


def test_table_to_markdown_handles_empty_input() -> None:
    assert _table_to_markdown([]) == ""


def test_looks_like_table_accepts_numbers_heavy_rows() -> None:
    rows = [
        ["Jahr", "Umsatz", "EBITDA"],
        ["2025", "363.576", "43.059"],
        ["2024", "436.906", "85.040"],
    ]

    assert _looks_like_table(rows) is True


def test_looks_like_table_rejects_wrapped_prose() -> None:
    # A management-bio paragraph the text-based table strategy split into
    # "columns" purely because of coincidental word-gap alignment - no
    # numeric cells, so it should not be treated as a real table.
    rows = [
        ["Robert von Wülfing", "Monika Wiederhold", "Dennis Gerlitzki"],
        ["Vorstandsvorsitzender (CEO)", "Chief Operating Officer (COO)", "Chief Operating"],
    ]

    assert _looks_like_table(rows) is False


def test_looks_like_table_rejects_too_few_cells() -> None:
    assert _looks_like_table([["2025", "363.576"]]) is False


def test_extract_page_content_without_tables_returns_plain_text() -> None:
    page = MagicMock()
    page.find_tables.return_value = []
    page.extract_text.return_value = "Plain prose text."

    assert _extract_page_content(page) == "Plain prose text."


def test_extract_page_content_ignores_false_positive_prose_tables() -> None:
    page = MagicMock()
    table = MagicMock()
    table.extract.return_value = [
        ["Robert von Wülfing", "Monika Wiederhold"],
        ["Vorstandsvorsitzender", "Chief Operating Officer"],
    ]
    page.find_tables.return_value = [table]
    page.extract_text.return_value = "Plain prose text."

    assert _extract_page_content(page) == "Plain prose text."


def test_extract_page_content_renders_tables_as_markdown_separately_from_prose() -> None:
    page = MagicMock()
    table = MagicMock()
    table.bbox = (0, 0, 100, 100)
    table.extract.return_value = [
        ["Jahr", "Umsatz", "EBITDA"],
        ["2025", "363.576", "43.059"],
        ["2024", "436.906", "85.040"],
    ]
    page.find_tables.return_value = [table]
    page.filter.return_value.extract_text.return_value = "Prose outside the table."

    content = _extract_page_content(page)

    assert "Prose outside the table." in content
    assert "| Jahr | Umsatz | EBITDA |" in content
    assert "| 2025 | 363.576 | 43.059 |" in content
