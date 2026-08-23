from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_copilot.config import get_settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
