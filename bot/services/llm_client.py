"""LLM client abstraction for Google Gemini REST API and OpenAI-compatible endpoints."""

from abc import ABC, abstractmethod
import json
import logging
from typing import Any
import httpx
from bot.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised when an LLM call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body_excerpt: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body_excerpt = body_excerpt

        if is_retryable is not None:
            self._is_retryable = is_retryable
        else:
            # Default retry logic: network errors (None), 429, and 5xx are retryable
            if status_code is None:
                self._is_retryable = True
            elif status_code == 429 or status_code >= 500:
                self._is_retryable = True
            else:
                self._is_retryable = False

    @property
    def is_retryable(self) -> bool:
        """Whether this error should be retried."""
        return self._is_retryable


class LLMTruncationError(LLMError):
    """Exception raised when LLM generation reaches maximum token limit."""

    def __init__(self, message: str, truncated_text: str = "") -> None:
        super().__init__(message, status_code=200, is_retryable=True)
        self.truncated_text = truncated_text


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        allow_truncated: bool = False,
    ) -> str:
        """Generate text completion given system instruction and user content."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini REST API implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        max_output_tokens: int = 8192,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
            self._owns_client = True
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        allow_truncated: bool = False,
    ) -> str:
        client = await self._get_client()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {
                    "thinkingBudget": 0,
                },
            },
        }

        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            logger.error("Gemini request network error: %s", exc)
            raise LLMError(f"Network error connecting to Gemini API: {exc}", status_code=None) from exc

        if response.status_code != 200:
            excerpt = response.text[:200]
            logger.error("Gemini API non-200 response [%s]: %s", response.status_code, excerpt)
            raise LLMError(
                f"Gemini API returned status {response.status_code}: {excerpt}",
                status_code=response.status_code,
                body_excerpt=excerpt,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise LLMError(f"Failed to decode Gemini JSON response: {exc}", status_code=200) from exc

        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
            prompt_feedback = data.get("promptFeedback", {})
            raise LLMError(
                f"Gemini returned no candidates (feedback: {prompt_feedback})",
                status_code=200,
                is_retryable=False,
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text_pieces = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        full_text = "".join(text_pieces).strip()

        if finish_reason == "MAX_TOKENS":
            if not allow_truncated:
                logger.warning(
                    "Gemini generation truncated due to finishReason MAX_TOKENS (model=%s, output_chars=%d). "
                    "Raising retryable LLMTruncationError.",
                    self.model,
                    len(full_text),
                )
                raise LLMTruncationError(
                    f"Gemini output truncated due to MAX_TOKENS (model: {self.model})",
                    truncated_text=full_text,
                )
            else:
                logger.warning(
                    "Gemini generation truncated due to MAX_TOKENS persisted on retry (model=%s, output_chars=%d). "
                    "Returning truncated output for sanitization.",
                    self.model,
                    len(full_text),
                )
                if not full_text:
                    raise LLMError(
                        "Gemini returned empty parts with finishReason MAX_TOKENS",
                        status_code=200,
                        is_retryable=False,
                    )
                return full_text

        if not parts:
            raise LLMError(
                f"Gemini returned empty parts in candidate (finishReason: {finish_reason})",
                status_code=200,
                is_retryable=False,
            )

        if not full_text:
            raise LLMError("Gemini returned empty text", status_code=200, is_retryable=False)

        return full_text

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible REST API client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 8192,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
            self._owns_client = True
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        allow_truncated: bool = False,
    ) -> str:
        client = await self._get_client()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }

        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            logger.error("OpenAI-compatible API network error: %s", exc)
            raise LLMError(f"Network error connecting to OpenAI API: {exc}", status_code=None) from exc

        if response.status_code != 200:
            excerpt = response.text[:200]
            logger.error("OpenAI API non-200 response [%s]: %s", response.status_code, excerpt)
            raise LLMError(
                f"OpenAI API returned status {response.status_code}: {excerpt}",
                status_code=response.status_code,
                body_excerpt=excerpt,
            )

        try:
            data = response.json()
        except Exception:
            # Fallback: router might have returned SSE streaming lines (data: {...})
            accumulated: list[str] = []
            for line in response.text.splitlines():
                line = line.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        choices_chunk = chunk.get("choices", [])
                        if choices_chunk:
                            delta = choices_chunk[0].get("delta", {})
                            chunk_content = delta.get("content") or choices_chunk[0].get("message", {}).get("content")
                            if chunk_content:
                                accumulated.append(chunk_content)
                    except Exception:
                        pass
            if accumulated:
                return "".join(accumulated).strip()
            raise LLMError(f"Failed to decode OpenAI JSON response: {response.text[:200]}", status_code=200)

        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("OpenAI returned no choices", status_code=200, is_retryable=False)

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "unknown")
        message = choice.get("message", {})
        content = (message.get("content") or "").strip()

        if finish_reason == "length":
            if not allow_truncated:
                logger.warning(
                    "OpenAI generation truncated due to finish_reason 'length' (model=%s, output_chars=%d). "
                    "Raising retryable LLMTruncationError.",
                    self.model,
                    len(content),
                )
                raise LLMTruncationError(
                    f"OpenAI output truncated due to length (model: {self.model})",
                    truncated_text=content,
                )
            else:
                logger.warning(
                    "OpenAI generation truncated due to finish_reason 'length' persisted on retry (model=%s, output_chars=%d). "
                    "Returning truncated output for sanitization.",
                    self.model,
                    len(content),
                )
                if not content:
                    raise LLMError(
                        "OpenAI returned empty message content with finish_reason 'length'",
                        status_code=200,
                        is_retryable=False,
                    )
                return content

        if not content:
            raise LLMError("OpenAI returned empty message content", status_code=200, is_retryable=False)

        return content

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()


_shared_llm_client: BaseLLMClient | None = None
_shared_http_client: httpx.AsyncClient | None = None


def get_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Return configured singleton LLM client."""
    global _shared_llm_client, _shared_http_client
    if _shared_llm_client is not None:
        return _shared_llm_client

    if settings is None:
        settings = get_settings()

    # Configure httpx client with proxy if specified
    client_kwargs: dict[str, Any] = {"timeout": 120.0}
    if settings.PROXY_URL:
        client_kwargs["proxy"] = settings.PROXY_URL

    _shared_http_client = httpx.AsyncClient(**client_kwargs)

    if settings.AI_PROVIDER == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")
        _shared_llm_client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            max_output_tokens=settings.AI_MAX_OUTPUT_TOKENS,
            http_client=_shared_http_client,
        )
    elif settings.AI_PROVIDER == "openai":
        if not settings.OPENAI_BASE_URL or not settings.OPENAI_API_KEY or not settings.OPENAI_MODEL:
            raise ValueError("OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL are required")
        _shared_llm_client = OpenAIClient(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            max_output_tokens=settings.AI_MAX_OUTPUT_TOKENS,
            http_client=_shared_http_client,
        )
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {settings.AI_PROVIDER}")

    return _shared_llm_client


async def close_llm_client() -> None:
    """Close singleton LLM and HTTP clients."""
    global _shared_llm_client, _shared_http_client
    if _shared_llm_client is not None:
        try:
            await _shared_llm_client.close()
        finally:
            _shared_llm_client = None

    if _shared_http_client is not None and not _shared_http_client.is_closed:
        try:
            await _shared_http_client.aclose()
        finally:
            _shared_http_client = None
