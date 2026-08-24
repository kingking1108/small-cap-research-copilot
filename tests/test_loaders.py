import unicodedata
from pathlib import Path

from pypdf import PdfWriter

from research_copilot.ingestion.loaders import load_pdf, load_watchlist


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
