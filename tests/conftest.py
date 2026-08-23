import os

# Settings requires these at import time; tests never make real network
# calls, so dummy values are enough to satisfy validation.
os.environ.setdefault("OVH_API_KEY", "test-key")
os.environ.setdefault("OVH_CHAT_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("OVH_EMBEDDINGS_BASE_URL", "https://example.invalid/v1")
