# Small-Cap Research Copilot

An agentic RAG assistant for researching European small- and mid-cap equities
and convertible bonds. It answers analyst questions grounded in ingested
filings, can pull live price data, and cites its sources — with an automated
evaluation suite that scores answer faithfulness instead of relying on manual
spot-checks.

## Why this project

Financial research assistants are only useful if they don't hallucinate. This
project is a deliberately small but complete slice of that problem: retrieval
over real filings, an agent that decides when to use which tool, and a
grounding-based eval loop that turns "looks right" into a measured number.

## Architecture

```
                 ┌─────────────┐
   question ───▶ │  LangGraph  │
                 │    agent    │──▶ search_filings ──▶ Chroma (BGE embeddings)
                 │  (gpt-oss)  │──▶ get_stock_price ──▶ Yahoo Finance
                 └─────────────┘
                        │
                        ▼
                  cited answer
```

- **LLM & embeddings**: [OVHcloud AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/)
  (`gpt-oss-120b` for reasoning/tool-use, BGE for embeddings), accessed through
  an OpenAI-compatible API via `langchain-openai`.
- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) —
  an explicit state graph (`agent` ⇄ `tools`) rather than a black-box chain,
  so routing and tool dispatch are inspectable and testable.
- **Retrieval**: PDF filings chunked with `RecursiveCharacterTextSplitter`,
  embedded, and stored in a local Chroma vector store.
- **Tools**: `search_filings` (RAG lookup) and `get_stock_price` (Yahoo
  Finance via `yfinance`).
- **Evaluation**: a golden question set graded by an LLM-as-judge
  faithfulness check — does the answer only claim what the retrieved sources
  support (`eval/`).
- **Tracing (optional)**: [Langfuse](https://langfuse.com) via LangChain's
  callback interface — every agent step, tool call, and LLM call is traced
  with timing, prompts, and token usage when configured (see below).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in your OVHcloud API key + endpoint URLs
```

Add a few company filings (PDFs) to `data/raw/`, named after the company,
e.g. `data/raw/example_ag.pdf`.

**Optional — tracing with Langfuse:** sign up at
[cloud.langfuse.com](https://cloud.langfuse.com) (free tier) or self-host,
create a project, and copy its Public/Secret key into `.env`
(`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`). `research-copilot ask` and
`research-copilot eval` then send every agent run to Langfuse automatically —
open the project's Traces view to see each tool call, retrieved chunk, and
LLM call in order, with latency and token counts. Leave both keys blank to
run without tracing; nothing else changes.

## Usage

```bash
research-copilot ingest              # chunk + embed everything in data/raw/
research-copilot ask "What did Example AG report as FY revenue?"
research-copilot eval                # score answers against eval/golden_set.jsonl
```

## Testing

```bash
pytest --cov=src
ruff check .
```

## Status / roadmap

Retrieval, agent graph, tools, CLI, and eval harness are wired end-to-end and
verified against real filings for all 5 watchlist companies (Amadeus Fire,
Nagarro, Hypoport, SUSS MicroTec, Deutsche Beteiligungs AG): 100% correctness
and 96% average faithfulness across 13 answerable golden-set questions,
correct refusal on 3 deliberately unanswerable ones.

- [x] Pick a watchlist (5 issuers) and drop filings into `data/raw/`
- [x] Populate `eval/golden_set.jsonl` with real, verified questions,
      including deliberately unanswerable ones to test refusal behaviour
- [ ] Wire `ResearchReport` (`src/research_copilot/report/schema.py`) into a
      final graph node via `llm.with_structured_output(...)` for structured,
      citation-checked report output
- [ ] Track retrieval precision@k alongside faithfulness in `eval/metrics.py`

## Known limitations

**The hosted model is not fully deterministic even at `temperature=0`.**
Re-running the exact same golden-set question ("What was DBAG's 2025 group
result?") three times produced three different behaviours:

1. A correct, cited answer (24,698 thousand €) — but the retrieved chunks
   that specific run actually used didn't clearly contain that figure, so
   the faithfulness judge correctly scored it low (0.40) despite the number
   being right. Correct ≠ grounded, and this is exactly why the eval harness
   tracks both metrics separately rather than just checking the final answer.
2. A **fabricated number** (223,018,243.04 €) attributed to a citation that,
   on inspection, doesn't contain that figure anywhere — a genuine
   hallucination with a fake-looking source reference.
3. A correct, well-grounded answer after retrying the search 4 times (past
   the "stop after 2 attempts" guardrail in the system prompt, which the
   model doesn't reliably follow).

Root cause: the first couple of retrieval queries for "Konzernergebnis" kept
matching boilerplate audit-opinion text instead of the actual results table,
and the model's behaviour when a search comes up empty is inconsistent —
sometimes it persists, sometimes it fabricates a plausible-looking number
instead of retrying or admitting it doesn't know.

This wasn't caught by manual spot-checking earlier in the project — it only
surfaced because the eval suite runs the same questions repeatedly and
tracks faithfulness independently of correctness. A production system would
need either self-consistency sampling (run N times, flag disagreement),
stricter citation verification (reject numeric claims that don't literally
appear in the retrieved chunks), or a stronger base model — not attempted
here to keep the project scope basic, but a natural next step and a good
discussion point on evaluation methodology.

## License

MIT — add a `LICENSE` file before publishing if you want this explicit.
