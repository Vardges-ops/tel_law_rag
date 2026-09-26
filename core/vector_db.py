"""
Qdrant client + collection schema for hybrid (dense + sparse) retrieval.

Defaults to embedded/local mode (no Docker required) via QdrantClient(path=...).
Set QDRANT_URL env var to point at a real server instead (e.g. the
docker-compose one) without changing any calling code.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
)

COLLECTION_NAME = "law_articles"
DENSE_DIM = 1024  # bge-m3 dense output size

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url)
        else:
            _client = QdrantClient(path="./storage/qdrant")
    return _client


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
