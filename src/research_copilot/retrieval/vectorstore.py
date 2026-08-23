import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from research_copilot.config import get_settings
from research_copilot.retrieval.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="research_copilot",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def reset_vectorstore() -> None:
    """Wipe the persisted collection so `ingest` is a clean full rebuild
    instead of appending duplicate chunks on every re-run."""
    settings = get_settings()
    shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)


def add_documents(documents: list[Document]) -> None:
    get_vectorstore().add_documents(documents)
