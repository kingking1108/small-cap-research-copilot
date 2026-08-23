import re

from langchain_core.messages import HumanMessage, SystemMessage

from research_copilot.llm import get_chat_model

# Models often format large numbers with different separators than the
# source text (e.g. gpt-oss-120b uses U+202F narrow no-break space instead
# of the source's "."), so strip all whitespace-like separators before
# comparing rather than requiring an exact substring match.
_SEPARATORS = re.compile(r"[\s\xa0\u202f]")

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


def contains_expected(answer: str, expected_substrings: list[str]) -> bool:
    """Lightweight correctness signal: does the answer mention at least one of
    the expected substrings (e.g. alternate number formats). Complements
    faithfulness, which only checks grounding, not whether the answer
    actually contains the fact being asked for."""
    if not expected_substrings:
        return True
    normalized_answer = _SEPARATORS.sub("", answer.lower())
    return any(
        _SEPARATORS.sub("", substring.lower()) in normalized_answer
        for substring in expected_substrings
    )
