"""End-to-end behaviour against the real Gemini API. Needs GEMINI_API_KEY.

    python -m eval.live_eval            # all cases
    python -m eval.live_eval --category near-domain

Checks that each question is answered or refused as labelled, and that every
answer carries a verified citation.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from dealintel.config import Settings
from dealintel.ingest import ingest, is_stale
from dealintel.retriever import Retriever

from .questions import ALL, Case


def run(cases: list[Case]) -> int:
    settings = Settings.from_env()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    if is_stale(settings):
        print("Index stale; re-ingesting...")
        ingest(settings)

    retriever = Retriever(settings)
    failures: list[str] = []
    for case in cases:
        try:
            answer = retriever.answer(case.question)
        except Exception as exc:
            failures.append(f"{case.question}: {exc}")
            print(f"[ERROR] {case.category:13} | {case.question}\n         {exc}\n")
            continue

        ok = answer.refused == case.expect_refusal
        # An answer with no verified citation is a bug even when it reads well.
        if not answer.refused and not answer.citations:
            ok = False
        nearest = min((m.distance for m in answer.matches), default=9.0)
        print(f"[{'PASS' if ok else 'FAIL'}] {case.category:13} d={nearest:.3f} "
              f"refused={answer.refused!s:5} | {case.question}")
        if case.note:
            print(f"         expect: {case.note}")
        print(f"         -> {answer.text[:220].replace(chr(10), ' ')}")
        if answer.citations:
            print(f"         cites: {answer.citations}")
        print()
        if not ok:
            failures.append(case.question)
        sys.stdout.flush()

    print(f"=== {len(cases) - len(failures)}/{len(cases)} as expected ===")
    for failure in failures:
        print(f"  FAILED: {failure}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="in-domain | false-premise | near-domain | out-of-domain")
    args = parser.parse_args()
    load_dotenv()
    cases = [c for c in ALL if not args.category or c.category == args.category]
    return run(cases)


if __name__ == "__main__":
    raise SystemExit(main())
