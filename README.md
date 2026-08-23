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

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in your OVHcloud API key + endpoint URLs
```

Add a few company filings (PDFs) to `data/raw/`, named after the company,
e.g. `data/raw/example_ag.pdf`.

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

This is the scaffold — retrieval, agent graph, tools, CLI, and eval harness
are wired end-to-end, but need real content to be useful:

- [ ] Pick a watchlist (5-8 issuers) and drop their filings into `data/raw/`
- [ ] Replace the placeholder rows in `eval/golden_set.jsonl` with real
      questions, including a few that are deliberately unanswerable from the
      filings (to test refusal behaviour, not just recall)
- [ ] Wire `ResearchReport` (`src/research_copilot/report/schema.py`) into a
      final graph node via `llm.with_structured_output(...)` for structured,
      citation-checked report output
- [ ] Track retrieval precision@k alongside faithfulness in `eval/metrics.py`

## License

MIT — add a `LICENSE` file before publishing if you want this explicit.
