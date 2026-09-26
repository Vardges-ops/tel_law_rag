"""
Retrieval pipeline for one query:

1. Detect language -> pick the lang filter ("hy" or "en") for the primary
   search, per the twin-corpus design (articles are identical in meaning
   across languages where they align; see ingest/parse.py reconciliation
   report for the handful that don't).
2. Run dense search and sparse search separately against Qdrant, both
   filtered by the detected language, then fuse with Reciprocal Rank Fusion.
3. Dedupe by article_number+lang and rerank with bge-reranker-v2-m3.
"""
import re

from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

from core.vector_db import get_client, COLLECTION_NAME
from core.embeddings import embed_query
from core.reranker import rerank
from core.config import TOP_K_RETRIEVE, TOP_K_RERANK, RRF_K



def _search_dense(client, query_vec, lang_filter, limit):
    flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang_filter))]) if lang_filter else None
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=flt,
        limit=limit,
    ).points


def _search_sparse(client, sparse_vec, lang_filter, limit):
    flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang_filter))]) if lang_filter else None
    sv = SparseVector(indices=[int(k) for k in sparse_vec.keys()], values=list(sparse_vec.values()))
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=sv,
        using="sparse",
        query_filter=flt,
        limit=limit,
    ).points


def _rrf_combined_rank(*ranked_lists: list, k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across lists.
    Returns payload dicts (deduped by point id) sorted by fused score."""
    scores: dict[int, float] = {}
    payloads: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank_idx, point in enumerate(ranked):
            scores[point.id] = scores.get(point.id, 0.0) + 1.0 / (k + rank_idx + 1)
            payloads[point.id] = point.payload
    ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [payloads[i] for i in ordered_ids]


def retrieve(query: str, top_k_final: int = TOP_K_RERANK) -> list[dict]:

    client = get_client()

    lang = "hy" if re.compile(r'[\u0530-\u058F]').search(query) else "en"

    vecs = embed_query(query)

    dense_hits = _search_dense(client, vecs["dense"], lang, TOP_K_RETRIEVE)
    sparse_hits = _search_sparse(client, vecs["sparse"], lang, TOP_K_RETRIEVE)

    fused = _rrf_combined_rank(dense_hits, sparse_hits)

    # dedupe by (article_number, lang) keeping first (highest fused rank) occurrence
    seen = set()
    deduped = []
    for p in fused:
        key = (p["article_number"], p["lang"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return rerank(query, deduped[:TOP_K_RETRIEVE], top_k=top_k_final)
