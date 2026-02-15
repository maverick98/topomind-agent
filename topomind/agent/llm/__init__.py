"""
LLM transport layer.

Exposes:
- LLMClient (abstract interface)
- OllamaClient (local backend)
- GroqClient (remote backend)
"""

from .llm_client import LLMClient
from .ollama_client import OllamaClient
from .groq_client import GroqClient

__all__ = [
    "LLMClient",
    "OllamaClient",
    "GroqClient",
]
