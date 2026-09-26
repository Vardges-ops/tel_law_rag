from rag.retrieve import retrieve
from rag.context import assemble
from rag.generate import build_prompt, parse_response
from providers import ALL_PROVIDERS


def run(query: str, provider_name: str = "gemini") -> dict:
    retrieved = retrieve(query)
    context, index_map = assemble(retrieved)

    if not context:
        return {
            "query": query,
            "answer": "No relevant articles were found for this question.",
            "cited_articles": [],
            "sufficient": False,
        }

    prompt = build_prompt(query, context)
    provider = ALL_PROVIDERS[provider_name]
    result = provider.stream(prompt)

    if result.error:
        return {
            "query": query,
            "answer": "",
            "cited_articles": [],
            "sufficient": False,
            "error": result.error,
        }

    parsed = parse_response(result.text, index_map)
    parsed["query"] = query
    parsed["provider"] = provider_name
    parsed["ttft_ms"] = result.ttft_ms
    parsed["total_ms"] = result.total_ms
    parsed["prompt_tokens"] = result.prompt_tokens
    parsed["completion_tokens"] = result.completion_tokens
    return parsed
