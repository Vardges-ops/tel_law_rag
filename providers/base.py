"""
Every provider adapter implements this same interface so the benchmark
runner and the app's SSE endpoint can treat Gemini/Groq/Cerebras identically.
This uniformity is the entire point of the benchmark being fair.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class GenerationResult:
    provider: str
    text: str
    ttft_ms: float | None       # time to first streamed token; None if non-streaming path used
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None    # "rate_limit" | "timeout" | "malformed" | "other" | None


class Provider(ABC):
    name: str

    @abstractmethod
    def stream(self, prompt: str, temperature: float = 0.0, timeout: float = 30.0) -> GenerationResult:
        """Blocking call that internally streams and measures TTFT, but
        returns the fully assembled result. The app's SSE endpoint wraps
        this differently (chunk-by-chunk to the client); the benchmark
        runner just needs the final GenerationResult + timings."""
        raise NotImplementedError


def timed_result(provider_name: str, fn) -> GenerationResult:
    """Shared error-classification wrapper so every adapter reports
    failures the same way instead of leaking provider-specific exceptions."""
    start = time.monotonic()
    try:
        return fn(start)
    except TimeoutError:
        return GenerationResult(provider=provider_name, text="", ttft_ms=None,
                                 total_ms=(time.monotonic() - start) * 1000, error="timeout")
    except Exception as e:
        msg = str(e).lower()
        error_class = "rate_limit" if ("429" in msg or "rate" in msg) else "other"
        return GenerationResult(provider=provider_name, text="", ttft_ms=None,
                                 total_ms=(time.monotonic() - start) * 1000, error=error_class)
