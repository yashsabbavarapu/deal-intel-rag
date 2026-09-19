"""Runtime configuration, resolved from the environment with safe defaults."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

REFUSAL = "Insufficient context in internal docs to answer confidently."
ROOT = Path(__file__).resolve().parent.parent

_ENV = {
    "max_distance": ("DEALINTEL_MAX_DISTANCE", float),
    "top_k": ("DEALINTEL_TOP_K", int),
    "lexical_weight": ("DEALINTEL_LEXICAL_WEIGHT", float),
    "data_dir": ("DEALINTEL_DATA_DIR", Path),
    "persist_dir": ("DEALINTEL_PERSIST_DIR", Path),
    "gemini_model": ("GEMINI_MODEL", str),
    "gemini_api_key": ("GEMINI_API_KEY", str),
}


class Settings(BaseModel):
    """Immutable settings for ingestion and retrieval."""

    data_dir: Path = Field(default_factory=lambda: ROOT / "data")
    persist_dir: Path = Field(default_factory=lambda: ROOT / ".chroma")
    collection: str = "deal_intel"
    top_k: int = 3
    # Cosine distance (0 = identical, 2 = opposite). When no chunk is nearer than
    # this, the question is treated as out of domain and refused.
    max_distance: float = 0.78
    fetch_multiplier: int = 4  # candidates pulled per top_k before reranking
    lexical_weight: float = 0.35  # rerank strength; 0 disables the hybrid step
    gemini_model: str = "gemini-3.5-flash-lite"
    # Tried in order if the primary model keeps failing (quota, 403, outage).
    fallback_models: tuple[str, ...] = ("gemini-3.5-flash", "gemini-flash-latest")
    gemini_api_key: str | None = None
    request_timeout: float = 60.0
    max_attempts: int = 4  # tries per model for retryable statuses (429, 5xx)
    backoff_base: float = 1.0  # seconds; doubled per attempt, with full jitter
    model_config = {"frozen": True}

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        env = {
            field: cast(raw)
            for field, (var, cast) in _ENV.items()
            if (raw := os.getenv(var))
        }
        return cls(**{**env, **overrides})
