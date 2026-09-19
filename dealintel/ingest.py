"""Chunk the markdown corpus by H2/H3 heading and embed it into ChromaDB."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .config import Settings
from .models import Chunk

_HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """Chunk one document at its H2/H3 boundaries.

    An H1 is treated as the document title and names the preamble chunk. Content
    before the first heading becomes its own chunk unless it is only whitespace.
    """
    heading, buffer, chunks = "Preamble", [], []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{len(chunks)}::{heading}",
                    source=source,
                    heading=heading,
                    text=body,
                )
            )
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            buffer.append(line)
            continue
        hashes, title = match.groups()
        if len(hashes) == 1 and not chunks and not "\n".join(buffer).strip():
            heading = title  # H1 with no content yet: name the preamble after it
            continue
        flush()
        heading = title
    flush()
    return chunks


def chunk_corpus(data_dir: Path) -> list[Chunk]:
    """Chunk every markdown file in `data_dir`, sorted for deterministic ids."""
    return [
        chunk
        for path in sorted(data_dir.glob("*.md"))
        for chunk in chunk_markdown(path.name, path.read_text(encoding="utf-8"))
    ]


def get_collection(settings: Settings, reset: bool = False) -> Collection:
    """Open (or create) the persistent collection, configured for cosine distance."""
    client = chromadb.PersistentClient(path=str(settings.persist_dir))
    if reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:  # noqa: BLE001 - an absent collection is not an error
            pass
    return client.get_or_create_collection(
        name=settings.collection, metadata={"hnsw:space": "cosine"}
    )


def fingerprint(chunks: list[Chunk]) -> str:
    """Content hash of the corpus, used to detect an index that has gone stale."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:16]


def is_stale(settings: Settings) -> bool:
    """True when `data/` no longer matches what was indexed."""
    stored = get_collection(settings).metadata or {}
    return stored.get("fingerprint") != fingerprint(chunk_corpus(settings.data_dir))


def ingest(settings: Settings, reset: bool = True) -> list[Chunk]:
    """Chunk the corpus and embed it, mirroring `data/` exactly.

    Sections that were renamed or deleted since the last run are removed, so an
    incremental ingest cannot leave a superseded chunk behind to be cited later.
    """
    if not settings.data_dir.is_dir():
        raise FileNotFoundError(f"No corpus directory at {settings.data_dir}")
    if not (chunks := chunk_corpus(settings.data_dir)):
        raise ValueError(f"No markdown chunks found in {settings.data_dir}")
    # The raw body is stored in metadata so retrieval returns the section text
    # verbatim, independent of how the embedded document is composed.
    metadatas: list[dict[str, Any]] = [
        {"source": c.source, "heading": c.heading, "text": c.text} for c in chunks
    ]
    collection = get_collection(settings, reset=reset)
    ids = [c.chunk_id for c in chunks]
    collection.upsert(
        ids=ids, documents=[c.embedding_text() for c in chunks], metadatas=metadatas
    )
    if orphans := sorted(set(collection.get(include=[])["ids"]) - set(ids)):
        collection.delete(ids=orphans)
    # hnsw:space is fixed at creation and cannot be passed to modify().
    collection.modify(metadata={"fingerprint": fingerprint(chunks)})
    return chunks
