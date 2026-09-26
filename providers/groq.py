import time
import json
import httpx
from providers.base import Provider, GenerationResult, timed_result
from core.config import GROQ_API_KEY, GENERATION_TEMPERATURE

MODEL = "openai/gpt-oss-20b"
URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(Provider):
    name = "groq"

    def stream(self, prompt: str, temperature: float = GENERATION_TEMPERATURE, timeout: float = 30.0) -> GenerationResult:
        def _call(start):
            first_token_at = None
            chunks = []
            usage = {}
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": True,
            }
            with httpx.stream("POST", URL, headers=headers, json=body, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        continue
                    payload = json.loads(raw)
                    delta = payload["choices"][0].get("delta", {}).get("content", "")
                    if delta and first_token_at is None:
                        first_token_at = time.monotonic()
                    chunks.append(delta)
                    if payload.get("x_groq", {}).get("usage"):
                        usage = payload["x_groq"]["usage"]

            total = time.monotonic() - start
            ttft = (first_token_at - start) * 1000 if first_token_at else None
            return GenerationResult(
                provider=self.name,
                text="".join(chunks),
                ttft_ms=ttft,
                total_ms=total * 1000,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )

        return timed_result(self.name, _call)
