"""Tune the refusal threshold against the corpus. No API key, no network.

Run this whenever `data/` changes materially. It reports top-k recall and the
distance distributions that `max_distance` has to separate, then recommends a
threshold at the midpoint of the gap.

    python -m eval.calibrate
"""

from __future__ import annotations

import argparse

from dealintel.config import Settings
from dealintel.ingest import ingest, is_stale
from dealintel.retriever import Retriever

from .questions import ALL, IN_DOMAIN, REFUSAL_SET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reingest", action="store_true", help="rebuild the index first")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.reingest or is_stale(settings):
        print("Index missing or stale; re-ingesting...")
        ingest(settings)

    retriever = Retriever(settings)

    print(f"\nRetrieval (top_k={settings.top_k}, lexical_weight={settings.lexical_weight})")
    hits = top1 = 0
    scored = [c for c in ALL if c.section]
    for case in scored:
        headings = [m.chunk.heading for m in retriever.retrieve(case.question)]
        found = case.section in headings
        hits += found
        top1 += headings[0] == case.section
        if not found:
            print(f"  MISS  {case.question}\n        wanted {case.section!r}, got {headings}")
    print(f"  top-1 {top1}/{len(scored)}   top-{settings.top_k} recall {hits}/{len(scored)}")

    # The guardrail compares the nearest chunk's cosine distance to the threshold,
    # so these two distributions are exactly what it has to tell apart.
    answerable = [
        min(m.distance for m in retriever.retrieve(c.question))
        for c in IN_DOMAIN
    ]
    unanswerable = [
        min(m.distance for m in retriever.retrieve(c.question))
        for c in REFUSAL_SET
    ]
    worst, best = max(answerable), min(unanswerable)

    print("\nNearest-chunk cosine distance")
    print(f"  answerable    min={min(answerable):.3f}  max={worst:.3f}  (n={len(answerable)})")
    print(f"  unanswerable  min={best:.3f}  max={max(unanswerable):.3f}  (n={len(unanswerable)})")

    if best <= worst:
        print(
            f"\n  OVERLAP of {worst - best:.3f}: no threshold separates these sets.\n"
            "  The distance gate cannot be the only defence - the grounded prompt and\n"
            "  citation check carry it. Prefer a threshold near the answerable max so\n"
            "  real questions are not refused."
        )
        recommended = round(worst + 0.02, 2)
    else:
        recommended = round((worst + best) / 2, 2)
        print(f"\n  gap = {best - worst:+.3f}")

    print(f"  current  max_distance = {settings.max_distance}")
    print(f"  suggested max_distance = {recommended}   (DEALINTEL_MAX_DISTANCE)")

    misses = [c for c in REFUSAL_SET if min(m.distance for m in retriever.retrieve(c.question)) <= settings.max_distance]
    if misses:
        print("\n  Passes the distance gate despite being unanswerable (the model must refuse these):")
        for case in misses:
            print(f"    {case.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
