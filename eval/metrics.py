from langchain_core.messages import HumanMessage, SystemMessage

from research_copilot.llm import get_chat_model

JUDGE_PROMPT = (
    "You are grading whether an AI-generated answer is fully supported by the "
    "provided source excerpts. Respond with a single float between 0 and 1: "
    "1.0 means every claim in the answer is directly supported by the sources, "
    "0.0 means the answer contains claims with no support in the sources at "
    "all. Respond with only the number, nothing else."
)


def faithfulness_score(answer: str, sources: str) -> float:
    """LLM-as-judge grounding check: does `answer` only claim what `sources` support."""
    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"),
        ]
    )
    try:
        return max(0.0, min(1.0, float(str(response.content).strip())))
    except ValueError:
        return 0.0
