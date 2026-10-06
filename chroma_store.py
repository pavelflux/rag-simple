"""Step 6: a persistent vector store backed by ChromaDB (replaces the numpy VectorStore)."""

import numpy as np
import chromadb

from ingest import Chunk
from vector_store import SearchResult


class ChromaStore:
    """Same job as VectorStore (add + search), but saved to disk in ./chroma_db.

    Every chunk also stores its document's hash, so we can tell which files changed
    since they were indexed and re-embed only those.
    """

    def __init__(self, path: str = "chroma_db", collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Rank by cosine distance, like our numpy store. (Chroma's default is L2 distance.)
            configuration={"hnsw": {"space": "cosine"}},
            # We embed chunks ourselves, so Chroma doesn't need its own embedding model.
            embedding_function=None,
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray, doc_hash: str) -> None:
        self.collection.add(
            ids=[f"{c.source}#{c.index}" for c in chunks],  # unique and stable per chunk
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "heading": c.heading, "index": c.index, "doc_hash": doc_hash}
                for c in chunks
            ],
        )

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from one file."""
        self.collection.delete(where={"source": source})

    def clear(self) -> None:
        """Remove every chunk from the collection."""
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def indexed_hashes(self) -> dict[str, str]:
        """{file name: hash it had when indexed}, read from the stored chunk metadata."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return {m["source"]: m["doc_hash"] for m in metadatas}

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        # query() accepts many queries at once, so every field is a list per query; we sent one.
        documents, metadatas, distances = (
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )

        results = []
        for text, meta, distance in zip(documents, metadatas, distances):
            chunk = Chunk(text=text, source=meta["source"], heading=meta["heading"], index=meta["index"])
            # Chroma returns cosine *distance* (0 = identical). Similarity = 1 - distance.
            results.append(SearchResult(chunk, score=1 - distance))
        return results

    def __len__(self) -> int:
        return self.collection.count()
