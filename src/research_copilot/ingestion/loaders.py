import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: Path) -> list[Document]:
    """Extract raw text from a single PDF filing into one Document per page.

    `metadata["page"]` is 1-indexed to match how a human would cite "page 12"
    of a PDF. Filename (without extension) is used as the company identifier
    — name watchlist PDFs accordingly, e.g. `data/raw/example_ag.pdf`.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Corporate report PDFs are often owner-password-protected (restricts
        # copy/print) but readable with an empty user password.
        reader.decrypt("")
    # macOS normalizes filenames with umlauts to NFD (decomposed: "a" +
    # combining diaeresis) at the filesystem level, while text typed
    # elsewhere (e.g. eval/golden_set.jsonl) defaults to NFC (precomposed
    # "ä"). Visually identical, but `==` fails between them - normalize to
    # NFC here so every downstream string comparison against this source
    # name just works.
    source = unicodedata.normalize("NFC", path.name)
    company = unicodedata.normalize("NFC", path.stem)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": source, "company": company, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def load_watchlist(raw_dir: Path) -> list[Document]:
    return [page_doc for path in sorted(raw_dir.glob("*.pdf")) for page_doc in load_pdf(path)]
