"""Print what's stored in the Chroma index as formatted JSON.

Usage:
    python peek.py        # first 3 chunks
    python peek.py 10     # first 10 chunks
"""

import json
import sys

import chromadb

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3

col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
r = col.get(limit=limit, include=["documents", "metadatas", "embeddings"])

records = []
for id_, document, metadata, embedding in zip(r["ids"], r["documents"], r["metadatas"], r["embeddings"]):
    records.append({
        "id": id_,
        "metadata": metadata,
        "document": document,
        "embedding": {
            "dimensions": len(embedding),
            # numpy floats aren't JSON-serializable, so convert to plain Python floats
            "first_8": [round(float(x), 4) for x in embedding[:8]],
        },
    })

output = {"total_chunks": col.count(), "shown": len(records), "chunks": records}
print(json.dumps(output, indent=2, ensure_ascii=False))
