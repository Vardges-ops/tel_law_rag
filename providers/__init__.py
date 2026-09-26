from providers.gemini import GeminiProvider
from providers.groq import GroqProvider
from providers.cerebras import CerebrasProvider

ALL_PROVIDERS = {
    "gemini": GeminiProvider(),
    "groq": GroqProvider(),
    "cerebras": CerebrasProvider(),
}
