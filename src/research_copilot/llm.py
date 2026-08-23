from langchain_openai import ChatOpenAI

from research_copilot.config import get_settings


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Chat model client pointed at an OVHcloud AI Endpoints model.

    OVH's LLM APIs are OpenAI-spec compatible, so we reuse langchain-openai
    and just override base_url/api_key instead of calling the real OpenAI API.
    """
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.ovh_chat_base_url,
        api_key=settings.ovh_api_key,
        model=settings.ovh_chat_model,
        temperature=temperature,
    )
