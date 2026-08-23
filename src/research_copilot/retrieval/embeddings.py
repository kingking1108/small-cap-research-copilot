from langchain_openai import OpenAIEmbeddings

from research_copilot.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Embeddings client pointed at OVHcloud's BGE embeddings endpoint."""
    settings = get_settings()
    return OpenAIEmbeddings(
        base_url=settings.ovh_embeddings_base_url,
        api_key=settings.ovh_api_key,
        model=settings.ovh_embeddings_model,
        check_embedding_ctx_length=False,
    )
