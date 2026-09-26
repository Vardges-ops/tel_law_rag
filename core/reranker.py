from FlagEmbedding import FlagReranker
from core.config import RERANKER_MODEL

_reranker: FlagReranker | None = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """candidates: list of dicts each with a 'text' field (payload from Qdrant).
    Returns the same dicts, top_k of them, sorted by rerank score desc,
    each with a 'rerank_score' field added."""
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
