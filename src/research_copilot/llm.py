from langchain_openai import ChatOpenAI

from research_copilot.config import get_settings


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Chat model client pointed at any OpenAI-compatible endpoint.

    We reuse langchain-openai and just override base_url/api_key, so this
    works against OVHcloud AI Endpoints, OpenAI, Azure OpenAI, Groq, a local
    Ollama/vLLM server, or any other OpenAI-spec-compatible provider -
    configure it via LLM_CHAT_BASE_URL/LLM_API_KEY/LLM_CHAT_MODEL.
    """
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.llm_chat_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_chat_model,
        temperature=temperature,
    )
