"""Step 4: a tiny in-memory vector store, built from scratch with numpy."""

from dataclasses import dataclass

import numpy as np

from ingest import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity between the query and this chunk


class VectorStore:
    """Keeps chunks and their vectors side by side: row i of self.vectors belongs to self.chunks[i]."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape (number of chunks, dimensions)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        # Normalize every vector to length 1 once, at insert time.
        # For unit vectors, cosine similarity is just the dot product (|a| = |b| = 1).
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        """Return the k chunks most similar to the query, best first."""
        query = query_vector / np.linalg.norm(query_vector)

        # One matrix-vector multiplication scores the query against every chunk at once:
        # (n, d) @ (d,) → (n,) — one cosine similarity per chunk.
        scores = self.vectors @ query

        # Indices of the k highest scores, highest first.
        top_indices = np.argsort(scores)[::-1][:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self.chunks)
