import sys
from pathlib import Path

import typer
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from research_copilot.agent.graph import build_agent_graph, build_report_graph
from research_copilot.config import get_settings
from research_copilot.ingestion.chunking import chunk_documents
from research_copilot.ingestion.loaders import load_watchlist
from research_copilot.observability import get_langfuse_handler
from research_copilot.report.verify import find_unverified_claims
from research_copilot.retrieval.vectorstore import add_documents, reset_vectorstore

app = typer.Typer(help="Small-Cap Research Copilot CLI")


def _tool_call_status(message: AIMessage) -> str:
    call = message.tool_calls[0]
    if call["name"] == "get_stock_price":
        return f"Rufe Kursdaten ab: {call['args'].get('ticker', '')}..."
    query = call["args"].get("query", "")
    return f"Suche: {query}..." if query else "Suche in Filings..."


def _run_with_status(
    graph: CompiledStateGraph,
    initial_state: dict,
    config: dict,
    final_step_label: str | None = None,
) -> dict:
    """Run the graph via `.stream()` instead of `.invoke()`, updating a
    status line as each node finishes so a multi-round question (up to
    MAX_TOOL_CALLS search rounds, each a real LLM/API round-trip) shows
    what's actually happening instead of a blank prompt. `stream_mode`
    defaults to "updates": each yielded item is `{node_name: node_output}`,
    matching exactly the `{"messages": [...]}` / `{"report": ...}` dicts
    the graph's own nodes return (agent/graph.py), so no full-state
    resync is needed - just append/set as each node reports in."""
    console = Console()
    messages = list(initial_state.get("messages", []))
    report = None
    with console.status("Denke nach...") as status:
        for update in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if "messages" in node_output:
                    messages.extend(node_output["messages"])
                if "report" in node_output:
                    report = node_output["report"]

                if node_name == "agent":
                    latest = node_output["messages"][-1]
                    if latest.tool_calls:
                        status.update(_tool_call_status(latest))
                    elif final_step_label:
                        status.update(final_step_label)
                elif node_name == "tools":
                    status.update("Werte Ergebnisse aus...")
    return {"messages": messages, "report": report}


@app.command()
def ingest() -> None:
    """Parse and embed every PDF in the watchlist directory into the vector
    store. Always a clean full rebuild — re-running it after adding new PDFs
    does not duplicate the ones already ingested."""
    settings = get_settings()
    raw_dir = Path(settings.watchlist_dir)
    documents = load_watchlist(raw_dir)
    if not documents:
        typer.echo(f"No PDFs found in {raw_dir}. Add filings there first.")
        raise typer.Exit(code=1)
    chunks = chunk_documents(documents)
    reset_vectorstore()
    add_documents(chunks)
    pdf_count = len({doc.metadata["source"] for doc in documents})
    typer.echo(f"Ingested {pdf_count} PDF(s) ({len(documents)} pages) as {len(chunks)} chunks.")


@app.command()
def ask(question: str) -> None:
    """Ask the research agent a question and print its answer.

    Traced in Langfuse if LANGFUSE_PUBLIC_KEY/SECRET_KEY are set in .env.
    """
    agent = build_agent_graph()
    handler = get_langfuse_handler()
    config = {"callbacks": [handler], "run_name": question} if handler else {}
    result = _run_with_status(agent, {"messages": [HumanMessage(content=question)]}, config)
    Console().print(Markdown(result["messages"][-1].content))


@app.command()
def report(topic: str) -> None:
    """Generate a structured, citation-checked research report on a topic
    (e.g. a company name). Same agent/retrieval loop as `ask`, but ends in
    a validated ResearchReport instead of free-text prose.

    Traced in Langfuse if LANGFUSE_PUBLIC_KEY/SECRET_KEY are set in .env.
    """
    graph = build_report_graph()
    handler = get_langfuse_handler()
    run_name = f"report: {topic}"
    config = {"callbacks": [handler], "run_name": run_name} if handler else {}
    prompt = f"Erstelle einen Research-Report zu: {topic}"
    try:
        result = _run_with_status(
            graph,
            {"messages": [HumanMessage(content=prompt)]},
            config,
            final_step_label="Erstelle strukturierten Report...",
        )
    except Exception as exc:
        # The hosted model is not fully deterministic (see README) and can
        # return output that fails ResearchReport's structured-output
        # validation; fail gracefully like `ingest` does instead of an
        # unhandled traceback.
        typer.echo(f"Report generation failed: {exc}")
        raise typer.Exit(code=1) from exc

    research_report = result.get("report")
    if research_report is None:
        typer.echo("Could not generate a structured report.")
        raise typer.Exit(code=1)

    lines = [f"# {research_report.company}", "", research_report.summary]
    if research_report.key_facts:
        lines.append("\n## Key Facts")
        for fact in research_report.key_facts:
            page = f", S. {fact.page}" if fact.page is not None else ""
            lines.append(f"- {fact.claim} (Quelle: {fact.source}{page})")
    if research_report.open_questions:
        lines.append("\n## Open Questions")
        for question in research_report.open_questions:
            lines.append(f"- {question}")

    console = Console()
    console.print(Markdown("\n".join(lines)))

    unverified = find_unverified_claims(research_report, result["messages"])
    if unverified:
        warning_lines = []
        for claim in unverified:
            page = f", S. {claim.page}" if claim.page is not None else ""
            warning_lines.append(f"{claim.claim!r} -> cited source {claim.source!r}{page}")
        console.print(
            Panel(
                "\n".join(warning_lines),
                title="Unverified Claims",
                subtitle="Claims citing a source (or page) the agent never actually retrieved",
                border_style="yellow",
            )
        )


@app.command(name="eval")
def run_eval() -> None:
    """Run the evaluation suite against the golden question set.

    `eval/` lives at the project root (not inside the installed package)
    since it's a human-edited dataset, not library code — add the cwd to
    sys.path so it's importable when invoked as the installed console
    script. Run this command from the project root.
    """
    sys.path.insert(0, str(Path.cwd()))
    from eval.run_eval import main as run_eval_main

    run_eval_main()


if __name__ == "__main__":
    app()
