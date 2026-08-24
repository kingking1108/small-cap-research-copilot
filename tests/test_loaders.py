import unicodedata
from pathlib import Path

from pypdf import PdfWriter

from research_copilot.ingestion.loaders import load_pdf, load_watchlist


def _write_blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)


def test_load_pdf_normalizes_source_to_nfc(tmp_path: Path) -> None:
    nfd_name = unicodedata.normalize("NFD", "Bericht_Übersicht.pdf")
    nfc_name = unicodedata.normalize("NFC", nfd_name)
    assert nfd_name != nfc_name  # sanity check that the two forms differ as strings

    pdf_path = tmp_path / nfd_name
    _write_blank_pdf(pdf_path)

    doc = load_pdf(pdf_path)

    assert doc.metadata["source"] == nfc_name
    assert doc.metadata["company"] == Path(nfc_name).stem


def test_load_watchlist_only_picks_up_pdfs(tmp_path: Path) -> None:
    _write_blank_pdf(tmp_path / "company_a.pdf")
    _write_blank_pdf(tmp_path / "company_b.pdf")
    (tmp_path / "notes.txt").write_text("not a pdf")

    docs = load_watchlist(tmp_path)

    sources = {doc.metadata["source"] for doc in docs}
    assert sources == {"company_a.pdf", "company_b.pdf"}
