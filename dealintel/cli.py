"""Command line entry point: `python -m dealintel.cli ask "..."`."""

from __future__ import annotations

import httpx
import typer
from dotenv import load_dotenv

from .config import Settings
from .ingest import get_collection, is_stale
from .ingest import ingest as run_ingest
from .retriever import Retriever

app = typer.Typer(add_completion=False, help="Query internal GTM docs with citations.")


@app.command()
def ingest(
    reset: bool = typer.Option(True, help="Drop the collection before re-embedding."),
) -> None:
    """Chunk the markdown corpus and embed it into the local ChromaDB collection."""
    settings = Settings.from_env()
    chunks = run_ingest(settings, reset=reset)
    by_source: dict[str, int] = {}
    for chunk in chunks:
        by_source[chunk.source] = by_source.get(chunk.source, 0) + 1
    for source, count in sorted(by_source.items()):
        typer.echo(f"  {source}: {count} chunks")
    typer.echo(f"Ingested {len(chunks)} chunks into '{settings.collection}'.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    top_k: int = typer.Option(3, help="Chunks to retrieve."),
    max_distance: float = typer.Option(
        0.75, help="Refuse when the nearest chunk is further than this."
    ),
    show_sources: bool = typer.Option(True, help="Print retrieval scores."),
) -> None:
    """Answer a question strictly from the retrieved chunks, or refuse."""
    settings = Settings.from_env(top_k=top_k, max_distance=max_distance)
    # An unbuilt index must not look like "the docs don't cover this" — that
    # reads as an authoritative "no" to whoever asked.
    if get_collection(settings).count() == 0:
        typer.secho(
            "Error: the index is empty. Run: python -m dealintel.cli ingest", fg="red"
        )
        raise typer.Exit(code=2)
    if is_stale(settings):
        typer.secho(
            "Warning: data/ has changed since the last ingest; answers may be stale. "
            "Run: python -m dealintel.cli ingest",
            fg="yellow",
            err=True,
        )
    try:
        result = Retriever(settings).answer(question)
    except (RuntimeError, httpx.HTTPError) as exc:
        typer.secho(f"Error: {exc}", fg="red")
        raise typer.Exit(code=2) from exc
    typer.echo(f"\n{result.text}\n")
    if result.refused:
        typer.echo("(refused: no chunk within the relevance threshold)")
    elif show_sources:
        typer.echo("Sources:")
        for match in result.matches:
            flag = "" if match.distance <= settings.max_distance else "  [below threshold]"
            typer.echo(f"  {match.chunk.citation}  (distance {match.distance:.3f}){flag}")
    raise typer.Exit(code=1 if result.refused else 0)


def main() -> None:
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
