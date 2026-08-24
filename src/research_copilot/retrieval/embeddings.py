from langchain_openai import OpenAIEmbeddings

from research_copilot.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Embeddings client pointed at any OpenAI-compatible embeddings endpoint.

    Configured via LLM_EMBEDDINGS_BASE_URL/LLM_API_KEY/LLM_EMBEDDINGS_MODEL.
    chunk_size is set conservatively low (LangChain's default is 1000) since
    some providers - e.g. OVHcloud - cap embedding requests at 25 texts per
    batch; lower it further via the provider's own limits if needed.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        base_url=settings.llm_embeddings_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_embeddings_model,
        check_embedding_ctx_length=False,
        chunk_size=20,
    )
