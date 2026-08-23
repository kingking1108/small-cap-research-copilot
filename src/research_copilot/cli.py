import sys
from pathlib import Path

import typer
from langchain_core.messages import HumanMessage

from research_copilot.agent.graph import build_agent_graph
from research_copilot.config import get_settings
from research_copilot.ingestion.chunking import chunk_documents
from research_copilot.ingestion.loaders import load_watchlist
from research_copilot.retrieval.vectorstore import add_documents

app = typer.Typer(help="Small-Cap Research Copilot CLI")


@app.command()
def ingest() -> None:
    """Parse and embed every PDF in the watchlist directory into the vector store."""
    settings = get_settings()
    raw_dir = Path(settings.watchlist_dir)
    documents = load_watchlist(raw_dir)
    if not documents:
        typer.echo(f"No PDFs found in {raw_dir}. Add filings there first.")
        raise typer.Exit(code=1)
    chunks = chunk_documents(documents)
    add_documents(chunks)
    typer.echo(f"Ingested {len(documents)} document(s) as {len(chunks)} chunks.")


@app.command()
def ask(question: str) -> None:
    """Ask the research agent a question and print its answer."""
    agent = build_agent_graph()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    typer.echo(result["messages"][-1].content)


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
