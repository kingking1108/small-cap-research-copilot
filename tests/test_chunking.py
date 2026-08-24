from langchain_core.documents import Document

from research_copilot.ingestion.chunking import chunk_documents


def test_chunk_documents_splits_long_text() -> None:
    long_text = "sentence. " * 500
    doc = Document(page_content=long_text, metadata={"source": "test.pdf"})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)


def test_chunk_documents_keeps_short_text_as_single_chunk() -> None:
    doc = Document(page_content="short filing excerpt", metadata={"source": "test.pdf"})

    chunks = chunk_documents([doc])

    assert len(chunks) == 1


def test_chunk_documents_propagates_page_metadata() -> None:
    long_text = "sentence. " * 500
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 7})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
