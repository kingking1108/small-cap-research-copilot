import re
import unicodedata
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

# Most of these reports lay out tables purely with whitespace (no ruling
# lines), so pdfplumber's default line-based table detection sees no grid at
# all and drops most cells. The text-based strategy groups by aligned text
# instead - but it then also groups ordinary wrapped prose into fake
# "tables" wherever line breaks happen to align, so every candidate is
# filtered by _looks_like_table below before being treated as one.
_TABLE_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}

_NUMERIC_CELL_RE = re.compile(r"^-?[\d.,]+\s*%?(\s*pp)?$", re.IGNORECASE)
_MIN_NUMERIC_FRACTION = 0.25


def _looks_like_table(rows: list[list[str | None]]) -> bool:
    """Real financial tables here are numbers-heavy (years, amounts,
    percentages); text-strategy false positives on prose are not - a
    management-bio paragraph split into "columns" by coincidental word
    gaps has ~0% numeric cells, a real metrics table has 50%+."""
    cells = [(cell or "").strip() for row in rows for cell in row if (cell or "").strip()]
    if len(cells) < 6:
        return False
    numeric = sum(1 for cell in cells if _NUMERIC_CELL_RE.match(cell))
    return numeric / len(cells) >= _MIN_NUMERIC_FRACTION


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_page_content(page) -> str:
    """A page's prose plus any tables rendered as clean Markdown, instead of
    pypdf-style raw text extraction where a table's columns run together as
    an unstructured sequence of numbers with no indication of which value
    belongs to which row/column - the direct cause of unusable, multi-number
    "facts" in generated reports. Tables are excluded from the prose extract
    (rather than left in *and* re-rendered below) to avoid every number
    appearing twice and bloating/diluting the chunk."""
    candidates = page.find_tables(table_settings=_TABLE_SETTINGS)
    tables = [table for table in candidates if _looks_like_table(table.extract() or [])]
    if not tables:
        return page.extract_text() or ""

    table_bboxes = [table.bbox for table in tables]

    def outside_tables(obj: dict) -> bool:
        v_mid = (obj["top"] + obj["bottom"]) / 2
        h_mid = (obj["x0"] + obj["x1"]) / 2
        return not any(
            x0 <= h_mid <= x1 and top <= v_mid <= bottom for x0, top, x1, bottom in table_bboxes
        )

    prose = page.filter(outside_tables).extract_text() or ""
    markdown_tables = [_table_to_markdown(table.extract() or []) for table in tables]
    return "\n\n".join(part for part in [prose, *markdown_tables] if part)


def load_pdf(path: Path) -> list[Document]:
    """Extract each page of a PDF filing into its own Document, with tables
    rendered as Markdown (see `_extract_page_content`).

    `metadata["page"]` is 1-indexed to match how a human would cite "page 12"
    of a PDF. Filename (without extension) is used as the company identifier
    — name watchlist PDFs accordingly, e.g. `data/raw/example_ag.pdf`.
    """
    # macOS normalizes filenames with umlauts to NFD (decomposed: "a" +
    # combining diaeresis) at the filesystem level, while text typed
    # elsewhere (e.g. eval/golden_set.jsonl) defaults to NFC (precomposed
    # "ä"). Visually identical, but `==` fails between them - normalize to
    # NFC here so every downstream string comparison against this source
    # name just works.
    source = unicodedata.normalize("NFC", path.name)
    company = unicodedata.normalize("NFC", path.stem)

    documents = []
    # Corporate report PDFs are often owner-password-protected (restricts
    # copy/print) but readable with an empty user password; passing one
    # unconditionally is a no-op for PDFs that aren't encrypted at all.
    with pdfplumber.open(str(path), password="") as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            documents.append(
                Document(
                    page_content=_extract_page_content(page),
                    metadata={"source": source, "company": company, "page": page_number},
                )
            )
    return documents


def load_watchlist(raw_dir: Path) -> list[Document]:
    return [page_doc for path in sorted(raw_dir.glob("*.pdf")) for page_doc in load_pdf(path)]
