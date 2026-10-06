"""Command-line entry point for the RAG app.

Usage:
    python rag.py search "how many vacation days do I get?" [-k 4] [--full]
"""

import argparse

from embeddings import Embedder
from ingest import Chunker, DocumentLoader
from vector_store import SearchResult, VectorStore


class RAG:
    """Ties the pieces together: load → chunk → embed → store, then search."""

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

        # Indexing. For now this runs on every start; step 6 saves it to disk.
        chunks = Chunker().chunk(DocumentLoader().load())
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))

    def search(self, question: str, k: int = 4) -> list[SearchResult]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k)


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
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="show the chunks most similar to a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=4, help="number of results (default 4)")
    search.add_argument("--full", action="store_true", help="print whole chunks, not previews")

    args = parser.parse_args()
    rag = RAG()

    if args.command == "search":
        print(f"Searching {len(rag.store)} chunks for: {args.query!r}\n")
        print_results(rag.search(args.query, args.k), args.full)


if __name__ == "__main__":
    main()
