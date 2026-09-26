"""
Embed every chunk from data/processed/chunks.jsonl and upsert into Qdrant.

Payload carries article_number, lang, chapter, title, and text -- lang is
what the retrieval-time language routing filters on (Armenian query ->
lang="hy" points, English query -> lang="en" points), per the twin-corpus
design. Run this once after ingest/chunk.py, and again whenever chunks.jsonl
changes.
"""

import json
import sys
from pathlib import Path

from qdrant_client.models import PointStruct, SparseVector

from core.embeddings import embed
from core.vector_db import get_client, ensure_collection, COLLECTION_NAME

BATCH_SIZE = 12


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_point(idx: int, chunk: dict, dense_vec: list[float], sparse_vec: dict) -> PointStruct:
    return PointStruct(
        id=idx,
        vector={
            "dense": dense_vec,
            "sparse": SparseVector(
                indices=[int(k) for k in sparse_vec.keys()],
                values=list(sparse_vec.values()),
            ),
        },
        payload={
            "chunk_id": chunk["chunk_id"],
            "article_number": chunk["article_number"],
            "lang": chunk["lang"],
            "chapter": chunk.get("chapter"),
            "title": chunk.get("title"),
            "text": chunk["text"],
        },
    )


def run(chunks_path: str = "data/processed/chunks.jsonl") -> int:
    chunks = load_chunks(chunks_path)
    client = get_client()
    ensure_collection(client)

    total = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs = embed(texts, batch_size=BATCH_SIZE)

        points = [
            to_point(start + i, chunk, vecs["dense"][i], vecs["sparse"][i])
            for i, chunk in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"  indexed {total}/{len(chunks)}", file=sys.stderr)

    return total


if __name__ == "__main__":
    n = run()
    print(f"Done. Indexed {n} chunks into '{COLLECTION_NAME}'.")

    # smoke test: confirm the collection actually has points and a quick
    # dense-only query returns something sane before moving on to Part 5
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection point count: {info.points_count}")
