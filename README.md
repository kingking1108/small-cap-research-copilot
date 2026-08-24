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
              ┌─────────┴─────────┐
              ▼                   ▼
         cited answer     generate_report (structured
          (`ask`)          output + citation check, `report`)
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
  Finance via `yfinance`). `get_stock_price` resolves watchlist company
  names to their correct ticker itself (`agent/tools.py::WATCHLIST_TICKERS`)
  rather than trusting the model's guess — it reliably invented plausible
  but wrong European exchange suffixes (e.g. "NGR.DE" for Nagarro; the real
  ticker is "NA9.DE").
- **Structured reports**: `research-copilot report` routes the same
  agent/tools loop through a final `generate_report` node
  (`llm.with_structured_output(ResearchReport)`) instead of ending on plain
  prose, then checks every cited source against what was actually retrieved.
- **Evaluation**: a golden question set graded by an LLM-as-judge
  faithfulness check — does the answer only claim what the retrieved sources
  support — plus a retrieval hit@k check that queries the vector store
  directly, decoupled from the agent's own query reformulation (`eval/`).
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
research-copilot report "Example AG"  # structured, citation-checked ResearchReport
research-copilot eval                # score answers against eval/golden_set.jsonl
```

`report` runs the same agent/retrieval loop as `ask`, but ends in a
validated `ResearchReport` (`report/schema.py`: summary, cited key facts,
open questions) instead of free-text prose. Every key fact's source is
checked against what the agent actually retrieved
(`report/verify.py::find_unverified_claims`) — a claim citing a document
the agent never saw gets flagged as a `[WARNING]` in the CLI output rather
than silently passed through.

## Testing

```bash
pytest --cov=src
ruff check .
```

## Status / roadmap

Retrieval, agent graph, tools, CLI, and eval harness are wired end-to-end and
verified against real filings for all 5 watchlist companies (Amadeus Fire,
Nagarro, Hypoport, SUSS MicroTec, Deutsche Beteiligungs AG): 100% correctness,
100% retrieval hit@12, and ~99% average faithfulness across 13 answerable
golden-set questions, correct refusal on 3 deliberately unanswerable ones.

- [x] Pick a watchlist (5 issuers) and drop filings into `data/raw/`
- [x] Populate `eval/golden_set.jsonl` with real, verified questions,
      including deliberately unanswerable ones to test refusal behaviour
- [x] Diagnose and fix a reproducible hallucination (see below)
- [x] Wire `ResearchReport` into a final graph node
      (`build_report_graph` in `agent/graph.py`) with citation checking
      against the agent's actual tool outputs (`report/verify.py`)
- [x] Track retrieval precision alongside faithfulness in `eval/metrics.py`
      (`retrieval_rank` / hit@k, decoupled from agent query reformulation —
      see below for a bug this surfaced on its first real run)

## A hallucination, diagnosed and fixed

**The hosted model is not fully deterministic even at `temperature=0`.**
Re-running the exact same question ("What was DBAG's 2025 group result?")
repeatedly produced different behaviours across runs: a correct, cited
answer; a correct answer after 4 retried searches; and once a **fabricated
number** (223,018,243.04 €) attributed to a citation that, on inspection,
didn't contain that figure anywhere. A later real user session hit the same
question and the agent looped through 11 search rounds before answering —
same underlying issue, worse symptom.

**Investigation, in order:**

1. Added a system-prompt instruction capping retries at 2 attempts. Didn't
   hold reliably — the 11-round session happened *after* this was in place,
   confirming prompt-level limits are a suggestion, not a guarantee.
2. Replaced it with a hard limit enforced in code
   (`agent/graph.py::MAX_TOOL_CALLS`): after 3 tool calls, the model is
   invoked *without* tools bound, so it structurally cannot call another one
   and must produce a final answer. This reliably capped every run at ≤3
   tool calls — but repeated testing showed the exact same fabricated
   number reappearing. The round-limit fixed cost/latency, not correctness;
   forcing a conclusion after failed searches can make fabrication *more*
   likely, not less.
3. Traced the fabrication to its source: `similarity_search_with_score`
   showed the chunk containing the real figure (24,698 thousand €) ranked
   **#11** for the natural-language query "Deutsche Beteiligungs AG
   Konzernergebnis 2025" — generic audit-opinion boilerplate (which
   literally repeats the word "Konzernergebnis") out-scored the actual
   results table at the default `retrieval_k=5`. The agent's searches
   weren't failing at reasoning; retrieval just never handed it the right
   text.
4. Raised `retrieval_k` from 5 to 12 (see `config.py`) to cover that rank.
   Verified with 5 repeated runs of the same question (5/5 correct,
   1-3 tool calls each, no more 11-round loops) and a full golden-set
   regression (100% correctness, 100% faithfulness, no regressions
   elsewhere) — runtime per `eval` run increased from ~1:15 to ~3:07 due to
   more context per search call, an accepted trade-off for reliability.

The lesson that generalizes: a hallucination that *looks* like a reasoning
or prompt-adherence failure was actually a retrieval-ranking failure one
layer down, and no amount of prompt engineering on the agent would have
fixed it — only measuring where the correct chunk actually ranked did. Two
things this doesn't fully close out: `retrieval_k=12` was tuned against one
observed case, not a systematic sweep, and nothing here stops a *different*
query from having its answer ranked below #12. A more general fix — quantified
retrieval precision@k in the eval suite, or citation verification that
rejects numeric claims not literally present in the retrieved text — remains
a natural next step and is called out above.

**A subtler variant surfaced afterward on a different unanswerable
question** ("What's the conversion price of Nagarro's convertible bond?" —
Nagarro doesn't have one). Across 4 runs, one produced "the report mentions
the convertible bond but doesn't state a price" — a false claim about the
document's *content*, not a fabricated number. The retrieved chunks (loan
schedules, derivative financial instruments, stock option disclosures) are
genuinely debt/equity-adjacent but never mention a convertible bond at all;
the model occasionally conflates topical similarity with "this document
covers that". Unlike the DBAG case, this isn't a retrieval-ranking problem
— there's no correct chunk to rank higher, because the fact doesn't exist
in the corpus. It's also not caught by `report/verify.py`'s citation check,
which only verifies that a cited *filename* was actually retrieved, not
that a claim's *content* is actually supported by what that source says.
Catching this class of error needs sentence-level entailment checking
(does source text X actually support claim Y), which `eval/metrics.py`'s
faithfulness judge already approximates for `ask`/`eval` — extending the
same idea to `report`'s per-claim citations would close this gap, and is
the more precise version of the citation-verification idea above.

## A retrieval metric that caught a bug in itself

Adding `retrieval_rank`/hit@k (measures whether the expected source ranks
within the top-k results for a question, searched directly against the
vector store rather than through the agent — see `eval/metrics.py`) was
meant to generalize the manual rank-checking done during the DBAG
investigation above into something the eval suite runs automatically. Its
first real run reported **2 misses, both DBAG questions, rank "not in top
20"** — even though the exact same questions were answered correctly with
faithfulness 1.00 moments earlier. That contradiction was the signal
something was wrong with the metric, not the retrieval.

Root cause: macOS normalizes filenames containing umlauts to **NFD**
(decomposed — "a" + a separate combining diaeresis codepoint) at the
filesystem level. The `expected_source` strings typed into
`eval/golden_set.jsonl` are plain **NFC** (precomposed "ä") — the standard
form for typed text. `"...Geschäftsbericht...".pdf` from the two forms
looks character-for-character identical printed to a terminal, but
`==` between them is `False`, since the underlying codepoints differ. Every
DBAG chunk's `source` metadata (only DBAG's filename contains an umlaut)
silently failed every exact-match comparison, regardless of actual rank.

Fixed by normalizing to NFC once, at the point filenames enter the system
(`ingestion/loaders.py::load_pdf`), plus defensively at the two comparison
sites that existed before this fix was in place
(`eval/metrics.py::retrieval_rank`, `report/verify.py::find_unverified_claims`)
so a differently-normalized string from any future source (LLM output,
a new data source) doesn't silently reintroduce the same failure. Required
a full re-ingest to rebuild the persisted Chroma metadata; verified after:
retrieval hit@12 100% (13/13), no change to correctness or faithfulness.

The pattern worth remembering: a new metric reporting numbers that
contradict other signals you already trust is itself useful information —
here, the contradiction (correct answer + "source unreachable") was the
fastest route to finding a bug that had nothing to do with retrieval at
all.

## License

[MIT](LICENSE)
