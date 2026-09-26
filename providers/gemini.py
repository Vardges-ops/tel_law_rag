import time
import json
import httpx

from providers.base import Provider, GenerationResult, timed_result
from core.config import GEMINI_API_KEY, GENERATION_TEMPERATURE

MODEL = "gemini-3.6-flash"
URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:streamGenerateContent?alt=sse&key={{key}}"
)


class GeminiProvider(Provider):
    name = "gemini"

    def stream(
        self,
        prompt: str,
        temperature: float = GENERATION_TEMPERATURE,
        timeout: float = 30.0,
    ) -> GenerationResult:

        def _call(start):
            first_token_at = None
            chunks = []
            usage = {}

            body = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                },
            }

            with httpx.stream(
                "POST",
                URL.format(key=GEMINI_API_KEY),
                json=body,
                timeout=timeout,
            ) as resp:

                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    payload = json.loads(
                        line[len("data:"):].strip()
                    )

                    # Extract generated text
                    candidates = payload.get("candidates", [])

                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])

                        for part in parts:
                            text_piece = part.get("text", "")

                            if text_piece:
                                if first_token_at is None:
                                    first_token_at = time.monotonic()

                                chunks.append(text_piece)

                    # Gemini usage metadata
                    if payload.get("usageMetadata"):
                        usage = payload["usageMetadata"]

            total = time.monotonic() - start

            ttft = (
                (first_token_at - start) * 1000
                if first_token_at
                else None
            )

            return GenerationResult(
                provider=self.name,
                text="".join(chunks),
                ttft_ms=ttft,
                total_ms=total * 1000,
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
            )

        return timed_result(self.name, _call)