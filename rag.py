"""Command-line entry point for the RAG app.

Usage:
    python rag.py search "how many vacation days do I get?" [-k 4] [--full]
    python rag.py ask "how many vacation days do I get?" [-k 4] [--show-prompt]
    python rag.py agent "how many vacation days do I get?"
    python rag.py compare "how many vacation days do I get?"    # ask vs agent, side by side
    python rag.py --rebuild search "..."    # throw away the saved index and rebuild it
"""

import argparse
import hashlib
from functools import cached_property

from dotenv import load_dotenv

from agent import Agent, AgentAnswer
from chroma_store import ChromaStore
from embeddings import Embedder
from generator import Answer, Generator
from ingest import Chunker, Document, DocumentLoader
from vector_store import SearchResult


class RAG:
    """Ties the pieces together: load → chunk → embed → store, then search and answer."""

    def __init__(self, rebuild: bool = False):
        self.embedder = Embedder()
        self.chunker = Chunker()
        self.store = ChromaStore()  # step 4's VectorStore lived in memory; this one is on disk
        self.sync_index(rebuild)

    def sync_index(self, rebuild: bool = False) -> None:
        """Make the saved index match ./docs, re-embedding only files that changed."""
        if rebuild:
            self.store.clear()

        documents = DocumentLoader().load()
        indexed = self.store.indexed_hashes()  # {file: hash when it was indexed}
        changed = False

        for doc in documents:
            doc_hash = self._hash(doc)
            if indexed.get(doc.source) == doc_hash:
                continue  # unchanged since last run: keep its stored vectors

            status = "changed" if doc.source in indexed else "new"
            self.store.delete_source(doc.source)  # drop the old chunks, if any
            chunks = self.chunker.chunk([doc])
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]), doc_hash)
            print(f"Indexed {doc.source} ({status}, {len(chunks)} chunks)")
            changed = True

        # Files that were indexed before but no longer exist in ./docs.
        for source in indexed.keys() - {doc.source for doc in documents}:
            self.store.delete_source(source)
            print(f"Removed {source} (file deleted)")
            changed = True

        if not changed:
            print(f"Index up to date ({len(self.store)} chunks)")
        print()

    @staticmethod
    def _hash(doc: Document) -> str:
        """A fingerprint of the document's text: any edit gives a completely different hash."""
        return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()

    @cached_property
    def generator(self) -> Generator:
        # Created on first use, so `search` works without an API key.
        return Generator()

    @cached_property
    def agent(self) -> Agent:
        # The agent's search tool is this class's own search method.
        return Agent(search=self.search)

    def search(self, question: str, k: int = 4) -> list[SearchResult]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k)

    def ask(self, question: str, k: int = 4) -> tuple[Answer, list[SearchResult]]:
        """Retrieve the top-k chunks, then let Claude answer from them."""
        results = self.search(question, k)
        return self.generator.answer(question, results), results


def print_results(results: list[SearchResult], full: bool) -> None:
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = chunk.heading or "(no heading)"
        print(f"{rank}. [{result.score:.3f}] {chunk.source} #{chunk.index} · {heading}")
        if full:
            print(chunk.text)
        else:
            preview = " ".join(chunk.text.split())  # collapse newlines for a one-line preview
            print(f"   {preview[:160]}{'…' if len(preview) > 160 else ''}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the documents in ./docs")
    parser.add_argument("--rebuild", action="store_true", help="re-index all documents from scratch")
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="show the chunks most similar to a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=4, help="number of results (default 4)")
    search.add_argument("--full", action="store_true", help="print whole chunks, not previews")

    ask = commands.add_parser("ask", help="answer a question with Claude, using retrieved chunks")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=4, help="number of chunks to send (default 4)")
    ask.add_argument("--show-prompt", action="store_true", help="print the exact prompt sent")

    agent = commands.add_parser("agent", help="let Claude search the docs itself with a tool")
    agent.add_argument("question")
    agent.add_argument("--show-responses", action="store_true", help="print each raw Claude response")

    compare = commands.add_parser("compare", help="run ask and agent on the same question")
    compare.add_argument("question")
    compare.add_argument("--show-responses", action="store_true", help="print each raw Claude response")

    args = parser.parse_args()
    rag = RAG(rebuild=args.rebuild)

    if args.command == "search":
        print(f"Searching {len(rag.store)} chunks for: {args.query!r}\n")
        print_results(rag.search(args.query, args.k), args.full)

    elif args.command == "ask":
        answer, results = rag.ask(args.question, args.k)

        if args.show_prompt:
            print("=== Prompt sent to Claude ===\n")
            print(rag.generator.build_prompt(args.question, results))
            print()

        print("=== Answer ===\n")
        print(answer.text)
        print(f"\n({answer.input_tokens} input tokens, {answer.output_tokens} output tokens)\n")

        print("=== Chunks given to Claude ===\n")
        print_results(results, full=False)

    elif args.command == "agent":
        print("=== Agent searches ===\n")
        print_agent_answer(rag.agent.run(args.question, show_responses=args.show_responses))

    elif args.command == "compare":
        print("=== 1) Fixed pipeline (step 5): one search, then answer ===\n")
        answer, results = rag.ask(args.question)
        sources = ", ".join(f"{r.chunk.source}#{r.chunk.index}" for r in results)
        print(f"  🔎 search({args.question!r}) → {sources}\n")
        print(answer.text)
        print(f"\n(1 search, 1 API call, {answer.input_tokens} input / {answer.output_tokens} output tokens)\n")

        print("=== 2) Agent (step 7): Claude decides what to search ===\n")
        print_agent_answer(rag.agent.run(args.question, show_responses=args.show_responses))


def print_agent_answer(answer: AgentAnswer) -> None:
    print("\n=== Answer ===\n")
    print(answer.text)
    print(
        f"\n({len(answer.searches)} searches, {answer.turns} API calls, "
        f"{answer.input_tokens} input / {answer.output_tokens} output tokens)\n"
    )


if __name__ == "__main__":
    load_dotenv()
    main()
