"""LLM client abstraction for Layer 5 feedback generation.

Active provider is controlled by LLM_PROVIDER in .env.
Never call LLM SDKs directly from pipeline code — always use this module.

Supported providers: gemini (default) | claude | deepseek

Usage:
    from itf_analysis.feedback.llm_client import get_llm_client

    client = get_llm_client()   # reads LLM_PROVIDER from .env
    text = client.generate(system_prompt=..., user_prompt=...)
"""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class LLMClientError(Exception):
    """Raised when an LLM call fails (timeout, empty response, API error)."""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        """Return generated text. Raises LLMClientError on failure."""


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class GeminiClient(LLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
    ) -> None:
        import google.generativeai as genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise LLMClientError("GEMINI_API_KEY not set in environment / .env")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        try:
            model = self._genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_prompt,
                generation_config=self._genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                ),
            )
            text = response.text.strip()
            if not text:
                raise LLMClientError("Gemini returned empty response")
            return text
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"Gemini API error: {e}") from e


class ClaudeClient(LLMClient):
    """Anthropic Claude via anthropic SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        import anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMClientError("ANTHROPIC_API_KEY not set in environment / .env")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = msg.content[0].text.strip()
            if not text:
                raise LLMClientError("Claude returned empty response")
            return text
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"Claude API error: {e}") from e


class DeepSeekClient(LLMClient):
    """DeepSeek via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        import openai

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise LLMClientError("DEEPSEEK_API_KEY not set in environment / .env")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise LLMClientError("DeepSeek returned empty response")
            return text
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"DeepSeek API error: {e}") from e


class GroqClient(LLMClient):
    """Groq via OpenAI-compatible API (free tier)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        import openai

        api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise LLMClientError("GROQ_API_KEY not set in environment / .env")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise LLMClientError("Groq returned empty response")
            return text
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"Groq API error: {e}") from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "gemini":   GeminiClient,
    "claude":   ClaudeClient,
    "deepseek": DeepSeekClient,
    "groq":     GroqClient,
}


def get_llm_client() -> LLMClient:
    """Create and return the LLM client configured by LLM_PROVIDER in .env.

    Raises:
        LLMClientError: If LLM_PROVIDER is unknown or the required API key is missing.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise LLMClientError(
            f"Unknown LLM_PROVIDER '{provider}'. Valid options: {list(_PROVIDERS)}"
        )
    return cls()
