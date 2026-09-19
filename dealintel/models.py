"""Typed records passed between the chunker, the store, and the answerer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One section of a source document, addressable by file and heading."""

    chunk_id: str
    source: str
    heading: str
    text: str
    model_config = {"frozen": True}

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.heading}"

    def embedding_text(self) -> str:
        """Heading is prepended so the section title carries retrieval signal."""
        return f"{self.heading}\n\n{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned by a vector search, with its cosine distance."""

    chunk: Chunk
    distance: float
    model_config = {"frozen": True}


class Answer(BaseModel):
    """The result of a query: either a grounded, cited answer or a refusal."""

    question: str
    text: str
    refused: bool
    citations: list[str] = Field(default_factory=list)
    matches: list[RetrievedChunk] = Field(default_factory=list)
    model_config = {"frozen": True}
