import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: Path) -> Document:
    """Extract raw text from a single PDF filing into a Document.

    Filename (without extension) is used as the company identifier — name
    watchlist PDFs accordingly, e.g. `data/raw/example_ag.pdf`.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Corporate report PDFs are often owner-password-protected (restricts
        # copy/print) but readable with an empty user password.
        reader.decrypt("")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # macOS normalizes filenames with umlauts to NFD (decomposed: "a" +
    # combining diaeresis) at the filesystem level, while text typed
    # elsewhere (e.g. eval/golden_set.jsonl) defaults to NFC (precomposed
    # "ä"). Visually identical, but `==` fails between them - normalize to
    # NFC here so every downstream string comparison against this source
    # name just works.
    source = unicodedata.normalize("NFC", path.name)
    return Document(page_content=text, metadata={"source": source, "company": path.stem})


def load_watchlist(raw_dir: Path) -> list[Document]:
    return [load_pdf(path) for path in sorted(raw_dir.glob("*.pdf"))]
