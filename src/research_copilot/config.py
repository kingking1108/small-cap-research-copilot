from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / .env.

    OVHcloud AI Endpoints assigns each model its own base URL. Copy the exact
    values from the "Documentation" tab of the model you picked in the catalog:
    https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ovh_api_key: str
    ovh_chat_base_url: str
    ovh_chat_model: str = "gpt-oss-120b"
    ovh_embeddings_base_url: str
    ovh_embeddings_model: str = "Qwen3-Embedding-8B"

    watchlist_dir: str = "data/raw"
    chroma_persist_dir: str = "data/processed/chroma"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
