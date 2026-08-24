"""Smoke test for the configured LLM/embeddings endpoint connectivity.

Run this before `research-copilot ingest`/`ask` to catch config or auth
issues in isolation, without spinning up the full agent graph.

    python scripts/check_llm_connection.py
"""

from research_copilot.llm import get_chat_model
from research_copilot.retrieval.embeddings import get_embeddings


def main() -> None:
    print("Testing chat completion...")
    chat = get_chat_model()
    response = chat.invoke("Reply with exactly one word: OK")
    print(f"  -> {response.content!r}")

    print("Testing embeddings...")
    embeddings = get_embeddings()
    vector = embeddings.embed_query("test")
    print(f"  -> got vector of length {len(vector)}")

    print("\nBoth endpoints reachable.")


if __name__ == "__main__":
    main()
