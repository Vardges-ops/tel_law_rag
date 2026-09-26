"""
Prompt template + structured-output parsing shared by every provider adapter.
Kept provider-agnostic: providers/*.py each call build_prompt() then send it
their own way, and all funnel their raw text response through parse_response()
so citation validation is identical regardless of which model answered.
"""

import json
import re

SYSTEM_PROMPT = """You are a legal assistant answering questions about the \
Republic of Armenia Law on Electronic Communications, using ONLY the context \
provided below. Each context block is labeled with an index like [1], [2].

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, set "sufficient" \
to false and explain briefly what is missing -- do not guess or use outside knowledge.
- Every factual claim must be traceable to a specific labeled context block.
- Respond in the same language as the question.
- Respond with ONLY a JSON object, no markdown fences, no preamble, in this \
exact shape:
{"answer": "...", "cited_articles": [<article numbers as they appear in the \
context labels, e.g. "12", "17.1">], "sufficient": true}
"""


def build_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

JSON response:"""


def parse_response(raw_text: str, index_map: list[dict]) -> dict:
    """Parse the model's JSON output and validate cited_articles against
    what was actually in the context (index_map from rag/context.py).
    Never trust the model's citations blindly -- drop any that don't
    correspond to a real context block and flag it."""
    cleaned = re.sub(r'^```json|```$', '', raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "answer": raw_text.strip(),
            "cited_articles": [],
            "sufficient": False,
            "error": "malformed_json",
        }

    valid_numbers = {m["article_number"] for m in index_map}
    cited = parsed.get("cited_articles", [])
    validated = [c for c in cited if str(c) in valid_numbers]
    dropped = [c for c in cited if str(c) not in valid_numbers]

    result = {
        "answer": parsed.get("answer", ""),
        "cited_articles": validated,
        "sufficient": parsed.get("sufficient", True),
    }
    if dropped:
        result["dropped_citations"] = dropped
    return result
