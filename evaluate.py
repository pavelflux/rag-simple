"""Step 8: measure retrieval quality. Does search find the right file for each test question?

Usage:
    python evaluate.py            # plain search
    python evaluate.py --split    # split each question into topics first (query decomposition)
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import colors
from rag import RAG

QUESTIONS_FILE = Path("eval_questions.json")
K_VALUES = [1, 3, 5]  # measure "is the right file in the top k?" for each of these


@dataclass
class EvalCase:
    question: str
    expected_sources: list[str]  # file(s) that contain the answer
    # Only for cases whose answers are also graded by an LLM judge (step 9, evaluate_answers.py):
    requires_llm_eval: bool = False
    reference_answer: str = ""  # what a correct answer says
    rubric: list[str] = field(default_factory=list)  # concrete points the judge checks one by one


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

    def __init__(self, rag: RAG, k_values: list[int] = K_VALUES, split: bool = False):
        self.rag = rag
        self.k_values = k_values
        self.split = split  # split each question into topics before searching (query decomposition)

    @staticmethod
    def load_cases(path: Path = QUESTIONS_FILE) -> list[EvalCase]:
        with open(path, encoding="utf-8") as f:
            return [EvalCase(**item) for item in json.load(f)]

    def run(self, cases: list[EvalCase]) -> list[EvalResult]:
        # Search once with the largest k. Top-3 is just the first 3 of the top-5 list.
        max_k = max(self.k_values)
        results = []
        for case in cases:
            found = self.rag.search(case.question, max_k, self.split)
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
        print(colors.bold(f"{'Question':<{question_width}} {rank_header:<{rank_width}} {k_header}"))
        print(colors.status("─" * (question_width + 1 + rank_width + 1 + len(k_header))))

        k_check = self.k_values[len(self.k_values) // 2]  # the "middle" k, used to flag misses
        for result, rank_text in zip(results, rank_texts):
            question = result.case.question
            if len(question) > question_width - 2:
                question = question[: question_width - 3] + "…"
            marks = " ".join(f"{'✅' if result.hit_at(k) else '❌':<3}" for k in self.k_values)

            # Rank cell: red = a file missed the top k_check, yellow = found but not all at rank 1.
            ranks = [result.rank_of(source) for source in result.case.expected_sources]
            rank_cell = f"{rank_text:<{rank_width}}"  # pad first, then color
            if not result.hit_at(k_check):
                rank_cell = colors.bad(rank_cell)
            elif any(rank != 1 for rank in ranks):
                rank_cell = colors.score(rank_cell)
            print(f"{question:<{question_width}} {rank_cell} {marks}")

        # Show what was retrieved instead, for questions that missed at the middle k.
        misses = [r for r in results if not r.hit_at(k_check)]
        if misses:
            print(colors.header(f"\nMisses at k={k_check} (what was retrieved instead):"))
            for result in misses:
                print(f"  • {colors.bold(result.case.question)}")
                print(colors.good(f"    expected: {', '.join(result.case.expected_sources)}"))
                print(colors.bad(f"    got:      {', '.join(result.retrieved_sources[:k_check])}"))

        print(colors.header("\nSummary"))
        total = len(results)
        for k in self.k_values:
            hits = sum(r.hit_at(k) for r in results)
            print(f"  hit rate @{k}: " + colors.score(f"{hits}/{total} ({hits / total:.0%})"))
        mrr = sum(r.reciprocal_rank() for r in results) / total
        print("  MRR:         " + colors.score(f"{mrr:.2f}") + colors.status("   (1.00 = the right file is always the top result)"))


if __name__ == "__main__":
    split = "--split" in sys.argv
    if split:
        load_dotenv()  # splitting calls Claude, so it needs the API key from .env

    evaluator = Evaluator(RAG(), split=split)
    cases = evaluator.load_cases()
    mode = "with query splitting" if split else "plain search"
    print(colors.header(f"Evaluating retrieval on {len(cases)} questions ({mode})\n"))
    run_result = evaluator.run(cases)
    evaluator.report(run_result)
