"""
bge-m3 embedding wrapper. bge-m3 natively produces BOTH a dense vector and
a lexical/sparse vector from one forward pass, which is exactly what the
hybrid retrieval design (dense + sparse fused via Qdrant's Query API) needs
-- no separate BM25 index required.
"""

from FlagEmbedding import BGEM3FlagModel

_model: BGEM3FlagModel | None = None


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        # use_fp16=True halves memory with negligible quality loss on CPU/GPU
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


def embed(texts: list[str], batch_size: int = 12) -> dict:
    """Returns {'dense': list[list[float]], 'sparse': list[dict[int, float]]}
    dense: 1024-dim vector per text, cosine-ready (already normalized).
    sparse: token_id -> weight dict per text, for Qdrant's sparse vector field.
    """
    model = get_model()
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # colbert multi-vector not used here; dense+sparse hybrid is enough
    )
    dense = [v.tolist() for v in out["dense_vecs"]]
    # bge-m3 sparse output is {token_id: weight}; keys arrive as numpy int64/str
    # depending on version -- normalize to str keys, which is what Qdrant expects.
    sparse = [
        {str(k): float(v) for k, v in weights.items()}
        for weights in out["lexical_weights"]
    ]
    return {"dense": dense, "sparse": sparse}


def embed_query(text: str, is_query: bool = True) -> dict:
    """Single-text convenience wrapper. bge-m3 doesn't require a different
    prefix for queries vs documents (unlike e.g. e5 models), so is_query is
    currently just documentation of intent, not a behavioral switch --
    kept as a parameter in case we add instruction-tuned prefixes later."""
    result = embed([text])
    return {"dense": result["dense"][0], "sparse": result["sparse"][0]}
