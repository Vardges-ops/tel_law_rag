"""
Turn reranked chunks into the final context sent to the LLM.

The reranker has already selected the most relevant chunks, so we pass those
chunks directly to the model instead of expanding them into full articles.

Each chunk is labeled [1], [2], ... so the generation layer can validate
which article numbers were actually present in the context.
"""

from typing import List


def assemble(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns:
        context_string:
            Labeled retrieval chunks to send to the LLM.

        index_map:
            Mapping between context labels and article numbers, used later
            to validate LLM citations.
    """

    blocks: List[str] = []
    index_map: List[dict] = []


    seen_chunks = set()

    for r in retrieved:
        chunk_id = r.get("chunk_id")

        dedupe_key = chunk_id or (
            r.get("article_number"),
            r.get("lang"),
            r.get("text"),
        )

        if dedupe_key in seen_chunks:
            continue

        seen_chunks.add(dedupe_key)

        index = len(blocks) + 1
        text = r.get("text", "").strip()

        if not text:
            continue

        blocks.append(f"[{index}] {text}")

        index_map.append(
            {
                "index": index,
                "article_number": r["article_number"],
                "lang": r["lang"],
            }
        )

    return "\n\n".join(blocks), index_map