import os

import pytest

# Settings requires these at import time; tests never make real network
# calls, so dummy values are enough to satisfy validation.
os.environ.setdefault("OVH_API_KEY", "test-key")
os.environ.setdefault("OVH_CHAT_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("OVH_EMBEDDINGS_BASE_URL", "https://example.invalid/v1")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """get_settings() is lru_cached; without clearing it, a test that
    monkeypatches an env var (e.g. LANGFUSE_PUBLIC_KEY, WATCHLIST_DIR) would
    silently see a stale cached Settings built by an earlier test, and its
    own env change would leak into whichever test runs next."""
    from research_copilot.config import get_settings
    from research_copilot.retrieval.vectorstore import get_known_companies

    get_settings.cache_clear()
    get_known_companies.cache_clear()
    yield
    get_settings.cache_clear()
    get_known_companies.cache_clear()
