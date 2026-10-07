"""Step 8: measure retrieval quality. Does search find the right file for each test question?

Usage:
    python evaluate.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag import RAG

QUESTIONS_FILE = Path("eval_questions.json")
K_VALUES = [1, 3, 5]  # measure "is the right file in the top k?" for each of these


@dataclass
class EvalCase:
    question: str
    expected_sources: list[str]  # file(s) that contain the answer


@dataclass
class EvalResult:
    case: EvalCase
    retrieved_sources: list[str]  # source file of each retrieved chunk, best match first

    def rank_of(self, source: str) -> int | None:
        """Position (1 = best) of the first chunk from this file, or None if it wasn't retrieved."""
        if source in self.retrieved_sources:
            return self.retrieved_sources.index(source) + 1
        return None

    def hit_at(self, k: int) -> bool:
        """True if every expected file appears among the top k chunks."""
        return all(source in self.retrieved_sources[:k] for source in self.case.expected_sources)

    def reciprocal_rank(self) -> float:
        """1 / rank of the first correct chunk: 1.0 if it's first, 0.5 if second, ..., 0 if missing."""
        ranks = [r for r in (self.rank_of(s) for s in self.case.expected_sources) if r is not None]
        return 1 / min(ranks) if ranks else 0.0


class Evaluator:
    """Runs every test question through retrieval and scores where the expected files landed."""

    def __init__(self, rag: RAG, k_values: list[int] = K_VALUES):
        self.rag = rag
        self.k_values = k_values

    @staticmethod
    def load_cases(path: Path = QUESTIONS_FILE) -> list[EvalCase]:
        with open(path, encoding="utf-8") as f:
            return [EvalCase(**item) for item in json.load(f)]

    def run(self, cases: list[EvalCase]) -> list[EvalResult]:
        # Search once with the largest k. Top-3 is just the first 3 of the top-5 list.
        max_k = max(self.k_values)
        results = []
        for case in cases:
            found = self.rag.search(case.question, max_k)
            results.append(EvalResult(case, [r.chunk.source for r in found]))
        return results

    def report(self, results: list[EvalResult]) -> None:
        # The "file → rank" text for each row, e.g. "employee_handbook.md:4, product_faq.txt:1".
        rank_texts = [
            ", ".join(f"{source}:{r.rank_of(source) or '–'}" for source in r.case.expected_sources)
            for r in results
        ]

        # Column widths in characters. The rank column grows to fit its longest value,
        # so rows with several expected files don't push the ✅/❌ columns out of line.
        question_width = 58
        rank_header = "Expected file → rank"
        rank_width = max(len(rank_header), *(len(text) for text in rank_texts))

        k_header = " ".join(f"@{k:<2}" for k in self.k_values)
        print(f"{'Question':<{question_width}} {rank_header:<{rank_width}} {k_header}")
        print("─" * (question_width + 1 + rank_width + 1 + len(k_header)))

        for result, rank_text in zip(results, rank_texts):
            question = result.case.question
            if len(question) > question_width - 2:
                question = question[: question_width - 3] + "…"
            marks = " ".join(f"{'✅' if result.hit_at(k) else '❌':<3}" for k in self.k_values)
            print(f"{question:<{question_width}} {rank_text:<{rank_width}} {marks}")

        # Show what was retrieved instead, for questions that missed at the middle k.
        k_check = self.k_values[len(self.k_values) // 2]
        misses = [r for r in results if not r.hit_at(k_check)]
        if misses:
            print(f"\nMisses at k={k_check} (what was retrieved instead):")
            for result in misses:
                print(f"  • {result.case.question}")
                print(f"    expected: {', '.join(result.case.expected_sources)}")
                print(f"    got:      {', '.join(result.retrieved_sources[:k_check])}")

        print("\nSummary")
        total = len(results)
        for k in self.k_values:
            hits = sum(r.hit_at(k) for r in results)
            print(f"  hit rate @{k}: {hits}/{total} ({hits / total:.0%})")
        mrr = sum(r.reciprocal_rank() for r in results) / total
        print(f"  MRR:         {mrr:.2f}   (1.00 = the right file is always the top result)")


if __name__ == "__main__":
    evaluator = Evaluator(RAG())
    cases = evaluator.load_cases()
    print(f"Evaluating retrieval on {len(cases)} questions\n")
    run_result = evaluator.run(cases)
    evaluator.report(run_result)
