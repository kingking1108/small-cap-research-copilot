from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / .env.

    Chat and embeddings each point at an OpenAI-compatible API - this works
    with OVHcloud AI Endpoints (https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/),
    OpenAI itself, Azure OpenAI, Groq, Together, Mistral, OpenRouter, a local
    Ollama/vLLM/llama.cpp server, or anything else speaking the same API -
    just point the base URL/model at whichever provider you're using.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: str
    llm_chat_base_url: str
    llm_chat_model: str = "gpt-oss-120b"
    llm_embeddings_base_url: str
    llm_embeddings_model: str = "Qwen3-Embedding-8B"

    watchlist_dir: str = "data/raw"
    chroma_persist_dir: str = "data/processed/chroma"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Empirically tuned: for one golden-set case the chunk containing the
    # actual figure only ranked #11 against a natural-language query (generic
    # boilerplate like audit opinions out-scored it at k=5). Higher k costs
    # more tokens per search, but a miss here means a wrong/fabricated answer.
    retrieval_k: int = 12

    # Optional: agent tracing via https://cloud.langfuse.com (or self-hosted).
    # Leave unset to run without tracing - nothing else depends on these.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
