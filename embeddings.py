"""Step 3: turn text into vectors (embeddings) with a local model."""

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import Chunker, DocumentLoader, count_tokens


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How closely two vectors point in the same direction: 1 = same, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Embedder:
    """Wraps a sentence-transformers model that runs on your own machine."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Downloaded from Hugging Face on first use (~90 MB), then cached locally.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one vector per text, as a (len(texts), dimensions) array."""
        return self.model.encode(texts)

    def token_count(self, text: str) -> int:
        """Exact token count using the model's own tokenizer."""
        return len(self.model.tokenizer(text)["input_ids"])

    @property
    def max_tokens(self) -> int:
        """Input longer than this is silently cut off before embedding."""
        return self.model.max_seq_length


# ---------- Demo ----------

EXAMPLE_PAIRS = [
    # same meaning, almost no shared words
    ("How many vacation days do I get?", "Employees receive 28 days of paid leave per year."),
    # related topic
    ("How many vacation days do I get?", "I'd like to take some time off this summer."),
    # unrelated
    ("How many vacation days do I get?", "The robot battery lasts about 10 hours."),
    # paraphrase with numbers written differently
    ("The robot can carry 600 kg.", "Maximum payload is six hundred kilograms."),
    # same word, different meaning
    ("The bank of the river was muddy.", "The bank approved my loan."),
]


if __name__ == "__main__":
    embedder = Embedder()

    print("1) What an embedding looks like\n")
    vector = embedder.embed(["How many vacation days do I get?"])[0]
    print(f"   'How many vacation days do I get?' → {len(vector)} numbers")
    print(f"   first 8: {np.round(vector[:8], 3)}\n")

    print("2) Cosine similarity between example sentences\n")
    for a, b in EXAMPLE_PAIRS:
        vec_a, vec_b = embedder.embed([a, b])
        print(f"   {cosine_similarity(vec_a, vec_b):.2f}  {a!r}")
        print(f"         {b!r}\n")

    print("3) Embedding all chunks from ./docs\n")
    chunks = Chunker().chunk(DocumentLoader().load())
    vectors = embedder.embed([c.text for c in chunks])
    print(f"   {len(chunks)} chunks → array of shape {vectors.shape}")

    # Check our words×1.3 estimate against the model's real tokenizer.
    longest = max(chunks, key=lambda c: embedder.token_count(c.text))
    print(
        f"   longest chunk: {embedder.token_count(longest.text)} real tokens "
        f"(we estimated {count_tokens(longest.text)}), model limit is {embedder.max_tokens}"
    )
