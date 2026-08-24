import os

from langfuse.langchain import CallbackHandler

from research_copilot.config import get_settings


def get_langfuse_handler() -> CallbackHandler | None:
    """Optional Langfuse tracing handler for the agent graph.

    Returns None (tracing disabled) unless LANGFUSE_PUBLIC_KEY/SECRET_KEY are
    configured, so the app behaves identically with or without Langfuse.

    The Langfuse SDK reads its config from the process environment rather
    than accepting it as constructor args, so we mirror our Settings into
    os.environ here instead of loading config two different ways.
    """
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_host)
    return CallbackHandler()
