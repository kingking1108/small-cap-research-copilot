import json
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from eval.metrics import contains_expected, faithfulness_score, retrieval_rank
from research_copilot.agent.graph import build_agent_graph
from research_copilot.config import get_settings
from research_copilot.observability import get_langfuse_handler

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_tool_sources(messages: list) -> str:
    """Concatenate the actual tool outputs the agent used to produce its
    answer. Faithfulness must be judged against what the agent actually saw,
    not a fresh, independent retrieval that may surface different chunks."""
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "(no tools were called)"
    return "\n\n---\n\n".join(str(m.content) for m in tool_messages)


def main() -> None:
    agent = build_agent_graph()
    cases = load_golden_set()
    handler = get_langfuse_handler()
    operational_k = get_settings().retrieval_k

    faithfulness_scores: list[float] = []
    correctness_results: list[bool] = []
    retrieval_hits: list[bool] = []
    retrieval_misses: list[tuple[str, int | None]] = []

    for case in cases:
        question = case["question"]
        answerable = case.get("answerable", True)
        expected_source = case.get("expected_source")

        config = (
            {"callbacks": [handler], "run_name": question, "tags": ["eval"]} if handler else {}
        )
        result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        answer = str(result["messages"][-1].content)
        sources = extract_tool_sources(result["messages"])

        faithfulness = faithfulness_score(answer, sources)
        faithfulness_scores.append(faithfulness)

        tag = "ANSWERABLE" if answerable else "UNANSWERABLE"
        print(f"\n[{tag}] {question}")
        print(f"  answer:       {answer[:200]}")
        print(f"  faithfulness: {faithfulness:.2f}")

        expected = case.get("expected_answer_contains") or []
        if answerable and expected:
            correct = contains_expected(answer, expected)
            correctness_results.append(correct)
            print(f"  correct:      {correct} (expected one of {expected})")

        if answerable and expected_source:
            rank = retrieval_rank(question, expected_source, search_k=20)
            hit = rank is not None and rank <= operational_k
            retrieval_hits.append(hit)
            if not hit:
                retrieval_misses.append((question, rank))
            rank_display = str(rank) if rank is not None else "not in top 20"
            print(f"  retrieval:    rank {rank_display} (operational k={operational_k})")

    print("\n" + "=" * 60)
    if faithfulness_scores:
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        n = len(faithfulness_scores)
        print(f"Average faithfulness: {avg_faithfulness:.2f} over {n} case(s)")
    if correctness_results:
        accuracy = sum(correctness_results) / len(correctness_results)
        correct_n = sum(correctness_results)
        total_n = len(correctness_results)
        print(f"Correctness (answerable cases): {accuracy:.0%} ({correct_n}/{total_n})")
    if retrieval_hits:
        hit_rate = sum(retrieval_hits) / len(retrieval_hits)
        hit_n = sum(retrieval_hits)
        total_n = len(retrieval_hits)
        print(f"Retrieval hit@{operational_k}: {hit_rate:.0%} ({hit_n}/{total_n})")
        for question, rank in retrieval_misses:
            rank_display = str(rank) if rank is not None else "not in top 20"
            print(f"  MISS (rank {rank_display}): {question}")
    if not cases:
        print("Golden set is empty — add real questions to eval/golden_set.jsonl.")


if __name__ == "__main__":
    main()
