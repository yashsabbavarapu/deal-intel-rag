"""IDF-weighted lexical scoring, used to rerank vector hits.

The embedding model is a small bi-encoder that treats antonyms as near-synonyms:
"Competitor X Weaknesses" and "Competitor X Strengths" sit almost on top of each
other. Scoring rare-term overlap pulls the literally-matching section back up.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = frozenset(
    """a an the is are was were do does did we our us you your i what when where which
    who whom how why can could should would will shall may might must of for to in on
    at by with from about as and or if it its this that these those have has had be
    been being not no any all""".split()
)

# Crude suffix stripping so "displace" matches "Displacement". A real stemmer is
# overkill here; the length floors keep short words intact.
_SUFFIXES = ("ements", "ement", "ations", "ation", "ingly", "ness",
             "ions", "ing", "ies", "ers", "ed", "es", "s", "e")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= (6 if suffix == "e" else 4):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords and 1-2 character tokens, then stem."""
    return [
        stem(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 2
    ]


@dataclass(frozen=True)
class LexicalIndex:
    """Corpus term statistics for BM25-style inverse document frequency."""

    n_docs: int
    doc_freq: Counter[str] = field(default_factory=Counter)
    heading_weight: float = 2.0

    @classmethod
    def build(cls, documents: list[str]) -> "LexicalIndex":
        freq: Counter[str] = Counter()
        for doc in documents:
            freq.update(set(tokenize(doc)))
        return cls(n_docs=max(len(documents), 1), doc_freq=freq)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, heading: str, body: str) -> float:
        """Share of the query's rare-term mass present in this chunk, in [0, 1].

        Common terms contribute little, so a section sharing only the word every
        chunk uses ("competitor") cannot outrank a literal match.
        """
        terms = set(tokenize(query))
        denominator = sum(self.idf(t) for t in terms) * self.heading_weight
        if not terms or denominator <= 0:
            return 0.0
        in_heading, in_body = set(tokenize(heading)), set(tokenize(body))
        matched = sum(
            self.idf(t) * (self.heading_weight if t in in_heading else 1.0)
            for t in terms
            if t in in_heading or t in in_body
        )
        return min(matched / denominator, 1.0)
