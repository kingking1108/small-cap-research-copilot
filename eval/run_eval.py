import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from eval.metrics import faithfulness_score
from research_copilot.agent.graph import build_agent_graph
from research_copilot.agent.tools import search_filings

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    agent = build_agent_graph()
    cases = load_golden_set()
    scores: list[float] = []

    for case in cases:
        question = case["question"]
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = str(result["messages"][-1].content)
        sources = search_filings.invoke(question)
        score = faithfulness_score(answer, sources)
        scores.append(score)
        print(f"[{score:.2f}] {question}")

    if scores:
        print(f"\nAverage faithfulness: {sum(scores) / len(scores):.2f} over {len(scores)} case(s)")
    else:
        print("Golden set is empty — add real questions to eval/golden_set.jsonl.")


if __name__ == "__main__":
    main()
