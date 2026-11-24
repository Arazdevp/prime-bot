"""LLM client abstraction for Google Gemini REST API and OpenAI-compatible endpoints."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import re
import time
from typing import Any
import httpx
from bot.config import Settings, get_settings

logger = logging.getLogger(__name__)

_DEFAULT_HTTP_TIMEOUT = httpx.Timeout(180.0, connect=30.0)
_MAX_RETRY_AFTER_SECONDS = 300.0
_MIN_AUTO_CLAMP_TOKENS = 512
_RETRY_DELAY_JSON_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', re.IGNORECASE)
_RETRY_AFTER_JSON_RE = re.compile(r'"retry[_]?after(?:_seconds)?"\s*:\s*"?(\d+(?:\.\d+)?)"?', re.IGNORECASE)
_RETRY_IN_TEXT_RE = re.compile(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_AFFORDABLE_TOKENS_RE = re.compile(r"can\s+only\s+afford\s+(\d+)", re.IGNORECASE)


def _clamp_retry_after(seconds: float) -> float | None:
    """Clamp a positive retry delay into a safe [1.0, 300.0] second window."""
    if seconds <= 0:
        return None
    return min(_MAX_RETRY_AFTER_SECONDS, max(1.0, float(seconds)))


def _safe_parse_json(response: Any) -> Any:
    """Best-effort parse JSON from response.text without triggering mock .json() traps."""
    raw_text = getattr(response, "text", None)
    if isinstance(raw_text, str) and raw_text.strip():
        try:
            return json.loads(raw_text)
        except Exception:
            return None
    return None


def _is_transient_in_flight_402(
    status_code: int | None,
    error_obj: Any = None,
    raw_text: str | None = None,
) -> bool:
    """Return True when HTTP 402 is OpenRouter's transient in-flight credit reservation limit.

    OpenRouter returns HTTP 402 with metadata.reason='in_flight_budget_exhausted' and
    metadata.limit_source='openrouter_in_flight_budget' when active/recently finished
    requests are still settling. Unlike genuine credit exhaustion (e.g. 'openrouter_credits'
    or 'weight_exceeds_budget'), this condition is temporary and safe to retry after waiting.
    """
    if status_code != 402:
        return False

    inner: dict[str, Any] | None = None
    if isinstance(error_obj, dict):
        nested = error_obj.get("error")
        inner = nested if isinstance(nested, dict) else error_obj

    msg_text = ""
    if isinstance(inner, dict):
        metadata = inner.get("metadata")
        if isinstance(metadata, dict):
            reason = str(metadata.get("reason") or "").lower()
            limit_source = str(metadata.get("limit_source") or "").lower()
            if "in_flight" in reason or "in-flight" in reason or "in_flight" in limit_source or "in-flight" in limit_source:
                return True
            if reason in ("weight_exceeds_budget", "insufficient_credits") or limit_source == "openrouter_credits":
                return False
        raw_msg = inner.get("message")
        if isinstance(raw_msg, str):
            msg_text = raw_msg

    combined = f"{msg_text} {raw_text or ''}".lower()
    if ("in-flight" in combined or "in_flight" in combined) and any(
        kw in combined for kw in ("settle", "budget", "available credits", "retry")
    ):
        return True
    return False


def _parse_affordable_tokens(error_obj: Any = None, raw_text: str | None = None) -> int | None:
    """Extract affordable token budget from OpenRouter 'can only afford N' HTTP 402 messages."""
    candidates: list[str] = []
    if isinstance(error_obj, dict):
        inner = error_obj.get("error") if isinstance(error_obj.get("error"), dict) else error_obj
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            candidates.append(inner["message"])
    if isinstance(raw_text, str) and raw_text:
        candidates.append(raw_text)

    for text in candidates:
        match = _AFFORDABLE_TOKENS_RE.search(text)
        if match:
            try:
                val = int(match.group(1))
                if val > 0:
                    return val
            except ValueError:
                pass
    return None


def _parse_retry_after(response: Any, parsed_json: Any = None) -> float | None:
    """Extract rate-limit or in-flight settle delay in seconds from HTTP headers or API error body."""
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        raw_retry = headers.get("retry-after")
        if isinstance(raw_retry, str) and raw_retry.strip():
            val = raw_retry.strip()
            try:
                return _clamp_retry_after(float(val))
            except ValueError:
                try:
                    dt = parsedate_to_datetime(val)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    delta = (dt - datetime.now(timezone.utc)).total_seconds()
                    clamped = _clamp_retry_after(delta)
                    if clamped is not None:
                        return clamped
                except Exception:
                    pass

        # OpenRouter / OpenAI-compatible reset timestamp header (epoch ms or seconds)
        raw_reset = headers.get("x-ratelimit-reset")
        if isinstance(raw_reset, str) and raw_reset.strip():
            try:
                reset_val = float(raw_reset.strip())
                if reset_val > 1e12:
                    reset_val /= 1000.0
                delta = reset_val - time.time()
                if 0 < delta <= 600.0:
                    return _clamp_retry_after(delta)
            except ValueError:
                pass

    # Check structured JSON error metadata (e.g. OpenRouter error.metadata.retry_after)
    if isinstance(parsed_json, dict):
        inner = parsed_json.get("error") if isinstance(parsed_json.get("error"), dict) else parsed_json
        containers = []
        if isinstance(inner, dict):
            if isinstance(inner.get("metadata"), dict):
                containers.append(inner["metadata"])
            containers.append(inner)
        if isinstance(parsed_json.get("metadata"), dict):
            containers.append(parsed_json["metadata"])
        for container in containers:
            for key in ("retry_after", "retry_after_seconds", "retryAfter", "retryAfterSeconds"):
                raw_val = container.get(key)
                if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
                    clamped = _clamp_retry_after(float(raw_val))
                    if clamped is not None:
                        return clamped
                elif isinstance(raw_val, str) and raw_val.strip():
                    try:
                        clamped = _clamp_retry_after(float(raw_val.strip()))
                        if clamped is not None:
                            return clamped
                    except ValueError:
                        pass

    raw_text = getattr(response, "text", None)
    if isinstance(raw_text, str) and raw_text:
        for pattern in (_RETRY_DELAY_JSON_RE, _RETRY_AFTER_JSON_RE, _RETRY_IN_TEXT_RE):
            match = pattern.search(raw_text)
            if match:
                try:
                    return _clamp_retry_after(float(match.group(1)))
                except ValueError:
                    pass

    return None


def _extract_openai_message_text(raw_content: Any) -> str:
    """Extract text content from OpenAI-compatible message.content (str or list of blocks)."""
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        pieces: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in (None, "text", "output_text") and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
        return "".join(pieces).strip()
    return ""


class LLMError(Exception):
    """Exception raised when an LLM call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body_excerpt: str | None = None,
        is_retryable: bool | None = None,
        retry_after: float | None = None,
        is_throttle: bool | None = None,
        is_in_flight_budget: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body_excerpt = body_excerpt
        self.retry_after = retry_after
        self.is_in_flight_budget = bool(is_in_flight_budget)
        if is_throttle is not None:
            self.is_throttle = bool(is_throttle)
        else:
            self.is_throttle = (status_code == 429) or self.is_in_flight_budget

        if is_retryable is not None:
            self._is_retryable = is_retryable
        elif self.is_in_flight_budget:
            self._is_retryable = True
        else:
            # Default retry logic: network errors (None), 408, 429, and 5xx are retryable
            if status_code is None:
                self._is_retryable = True
            elif status_code in (408, 429) or status_code >= 500:
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
            self._client = httpx.AsyncClient(timeout=_DEFAULT_HTTP_TIMEOUT)
            self._owns_client = True
        return self._client

    async def _post_generate(
        self,
        payload: dict[str, Any],
        allow_truncated: bool,
        model: str | None = None,
    ) -> str:
        """Send a Gemini generateContent request and extract clean non-thought text."""
        client = await self._get_client()
        target_model = (model or self.model).strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            logger.error("Gemini request network error: %s", exc)
            raise LLMError(f"Network error connecting to Gemini API: {exc}", status_code=None) from exc

        if response.status_code != 200:
            excerpt = response.text[:400]
            parsed_err = _safe_parse_json(response)
            retry_after = _parse_retry_after(response, parsed_err) if response.status_code in (429, 503) else None
            logger.error("Gemini API non-200 response [%s] (model=%s): %s", response.status_code, target_model, excerpt)
            raise LLMError(
                f"Gemini API returned status {response.status_code}: {excerpt}",
                status_code=response.status_code,
                body_excerpt=excerpt,
                retry_after=retry_after,
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
        non_thought_pieces = [
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and "text" in p and not p.get("thought")
        ]
        if non_thought_pieces:
            text_pieces = non_thought_pieces
        else:
            text_pieces = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        full_text = "".join(text_pieces).strip()

        if finish_reason == "MAX_TOKENS":
            if not allow_truncated:
                logger.warning(
                    "Gemini generation truncated due to finishReason MAX_TOKENS (model=%s, output_chars=%d). "
                    "Raising retryable LLMTruncationError.",
                    target_model,
                    len(full_text),
                )
                raise LLMTruncationError(
                    f"Gemini output truncated due to MAX_TOKENS (model: {target_model})",
                    truncated_text=full_text,
                )
            logger.warning(
                "Gemini generation truncated due to MAX_TOKENS persisted on retry (model=%s, output_chars=%d). "
                "Returning truncated output for sanitization.",
                target_model,
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

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        allow_truncated: bool = False,
    ) -> str:
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
        return await self._post_generate(payload, allow_truncated=allow_truncated)

    async def generate_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.85,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a multi-turn conversational reply using Gemini REST contents history."""
        contents = [
            {
                "role": msg["role"],
                "parts": [{"text": msg["text"]}],
            }
            for msg in messages
            if msg.get("text")
        ]
        if not contents:
            raise LLMError("Cannot generate chat reply without messages", status_code=400, is_retryable=False)

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens or self.max_output_tokens),
            },
        }
        return await self._post_generate(payload, allow_truncated=True, model=model)

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
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort.strip() if isinstance(reasoning_effort, str) and reasoning_effort.strip() else None
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_DEFAULT_HTTP_TIMEOUT)
            self._owns_client = True
        return self._client

    async def _post_chat_completions(
        self,
        messages_payload: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        allow_truncated: bool,
        model: str | None = None,
    ) -> str:
        """Send an OpenAI-compatible chat/completions request with 402 clamp and SSE handling."""
        client = await self._get_client()
        target_model = (model or self.model).strip()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        effective_max_tokens = int(max_tokens)

        for clamp_attempt in range(2):
            payload: dict[str, Any] = {
                "model": target_model,
                "messages": messages_payload,
                "temperature": float(temperature),
                "max_tokens": effective_max_tokens,
                "stream": False,
            }
            if self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}

            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                logger.error("OpenAI-compatible API network error: %s", exc)
                raise LLMError(f"Network error connecting to OpenAI API: {exc}", status_code=None) from exc

            if response.status_code != 200:
                excerpt = response.text[:400]
                parsed_err = _safe_parse_json(response)
                is_in_flight = _is_transient_in_flight_402(response.status_code, parsed_err, response.text)

                if response.status_code == 402 and not is_in_flight and clamp_attempt == 0:
                    affordable = _parse_affordable_tokens(parsed_err, response.text)
                    if affordable is not None and affordable >= _MIN_AUTO_CLAMP_TOKENS and affordable < effective_max_tokens:
                        clamped = max(_MIN_AUTO_CLAMP_TOKENS, min(effective_max_tokens - 1, int(affordable * 0.9)))
                        logger.warning(
                            "OpenAI-compatible API returned 402 affordable token cap (%d < requested %d). "
                            "Clamping max_tokens to %d and retrying once.",
                            affordable,
                            effective_max_tokens,
                            clamped,
                        )
                        effective_max_tokens = clamped
                        continue

                retry_after = (
                    _parse_retry_after(response, parsed_err)
                    if response.status_code in (402, 429, 503)
                    else None
                )
                if is_in_flight:
                    logger.warning(
                        "OpenRouter transient in-flight budget 402 (retry_after=%s): %s",
                        retry_after,
                        excerpt,
                    )
                else:
                    logger.error("OpenAI API non-200 response [%s]: %s", response.status_code, excerpt)
                raise LLMError(
                    f"OpenAI API returned status {response.status_code}: {excerpt}",
                    status_code=response.status_code,
                    body_excerpt=excerpt,
                    is_retryable=True if is_in_flight else None,
                    retry_after=retry_after,
                    is_throttle=True if is_in_flight else None,
                    is_in_flight_budget=is_in_flight,
                )

            try:
                data = response.json()
            except Exception:
                # Fallback: provider might have returned SSE streaming lines (data: {...})
                accumulated: list[str] = []
                sse_finish_reason: str | None = None
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if not payload_str or payload_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload_str)
                            choices_chunk = chunk.get("choices", [])
                            if choices_chunk:
                                first_choice = choices_chunk[0]
                                if first_choice.get("finish_reason"):
                                    sse_finish_reason = str(first_choice["finish_reason"])
                                delta = first_choice.get("delta", {})
                                chunk_content = delta.get("content") or first_choice.get("message", {}).get("content")
                                if isinstance(chunk_content, str) and chunk_content:
                                    accumulated.append(chunk_content)
                        except Exception:
                            pass
                if accumulated:
                    sse_text = "".join(accumulated).strip()
                    if sse_finish_reason == "length" and not allow_truncated:
                        raise LLMTruncationError(
                            f"OpenAI SSE output truncated due to length (model: {target_model})",
                            truncated_text=sse_text,
                        )
                    return sse_text
                raise LLMError(f"Failed to decode OpenAI JSON response: {response.text[:400]}", status_code=200)

            # OpenRouter / compatible gateways may return HTTP 200 with a JSON error object
            if isinstance(data, dict) and data.get("error"):
                err_obj = data["error"]
                if isinstance(err_obj, dict):
                    err_msg = str(err_obj.get("message") or err_obj)
                    raw_code = err_obj.get("code")
                    try:
                        err_code = int(raw_code) if raw_code is not None else 502
                    except (TypeError, ValueError):
                        err_code = 502
                else:
                    err_msg = str(err_obj)
                    err_code = 502

                is_in_flight = _is_transient_in_flight_402(err_code, data, err_msg)
                if err_code == 402 and not is_in_flight and clamp_attempt == 0:
                    affordable = _parse_affordable_tokens(data, err_msg)
                    if affordable is not None and affordable >= _MIN_AUTO_CLAMP_TOKENS and affordable < effective_max_tokens:
                        clamped = max(_MIN_AUTO_CLAMP_TOKENS, min(effective_max_tokens - 1, int(affordable * 0.9)))
                        logger.warning(
                            "OpenRouter 200-wrapped 402 affordable token cap (%d < requested %d). "
                            "Clamping max_tokens to %d and retrying once.",
                            affordable,
                            effective_max_tokens,
                            clamped,
                        )
                        effective_max_tokens = clamped
                        continue

                retry_after = _parse_retry_after(response, data) if err_code in (402, 429, 503) else None
                if is_in_flight:
                    logger.warning(
                        "OpenRouter 200-wrapped transient in-flight budget 402 (retry_after=%s): %s",
                        retry_after,
                        err_msg[:400],
                    )
                else:
                    logger.error("OpenAI API returned error payload [code=%s]: %s", err_code, err_msg[:400])
                raise LLMError(
                    f"OpenAI API returned error {err_code}: {err_msg[:400]}",
                    status_code=err_code,
                    body_excerpt=err_msg[:400],
                    is_retryable=True if is_in_flight else None,
                    retry_after=retry_after,
                    is_throttle=True if is_in_flight else None,
                    is_in_flight_budget=is_in_flight,
                )

            break

        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("OpenAI returned no choices", status_code=200, is_retryable=False)

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "unknown")
        message = choice.get("message", {})
        content = _extract_openai_message_text(message.get("content"))

        if finish_reason == "length":
            if not allow_truncated:
                logger.warning(
                    "OpenAI generation truncated due to finish_reason 'length' (model=%s, output_chars=%d). "
                    "Raising retryable LLMTruncationError.",
                    target_model,
                    len(content),
                )
                raise LLMTruncationError(
                    f"OpenAI output truncated due to length (model: {target_model})",
                    truncated_text=content,
                )
            else:
                logger.warning(
                    "OpenAI generation truncated due to finish_reason 'length' persisted on retry (model=%s, output_chars=%d). "
                    "Returning truncated output for sanitization.",
                    target_model,
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

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        allow_truncated: bool = False,
    ) -> str:
        return await self._post_chat_completions(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=self.max_output_tokens,
            allow_truncated=allow_truncated,
        )

    async def generate_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.85,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a multi-turn conversational reply via OpenAI-compatible chat/completions."""
        chat_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        has_turn = False
        for msg in messages:
            txt = (msg.get("text") or "").strip()
            if not txt:
                continue
            role = "assistant" if msg.get("role") == "model" else "user"
            chat_messages.append({"role": role, "content": txt})
            has_turn = True

        if not has_turn:
            raise LLMError("Cannot generate chat reply without messages", status_code=400, is_retryable=False)

        return await self._post_chat_completions(
            chat_messages,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens or self.max_output_tokens),
            allow_truncated=True,
            model=model,
        )

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
    client_kwargs: dict[str, Any] = {"timeout": _DEFAULT_HTTP_TIMEOUT}
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
        raw_effort = getattr(settings, "AI_REASONING_EFFORT", None)
        reasoning_effort = raw_effort if isinstance(raw_effort, str) and raw_effort.strip() else None
        _shared_llm_client = OpenAIClient(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            max_output_tokens=settings.AI_MAX_OUTPUT_TOKENS,
            http_client=_shared_http_client,
            reasoning_effort=reasoning_effort,
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


_shared_ai_chat_gemini_client: GeminiClient | None = None
_shared_ai_chat_openai_client: OpenAIClient | None = None
_shared_ai_chat_http_client: httpx.AsyncClient | None = None


def _get_ai_chat_http_client(settings: Settings) -> httpx.AsyncClient:
    """Return shared httpx.AsyncClient for interactive AI chat clients."""
    global _shared_ai_chat_http_client
    if _shared_ai_chat_http_client is None or _shared_ai_chat_http_client.is_closed:
        client_kwargs: dict[str, Any] = {"timeout": httpx.Timeout(60.0, connect=20.0)}
        if settings.PROXY_URL:
            client_kwargs["proxy"] = settings.PROXY_URL
        _shared_ai_chat_http_client = httpx.AsyncClient(**client_kwargs)
    return _shared_ai_chat_http_client


def get_ai_chat_client(
    settings: Settings | None = None,
    provider: str | None = None,
) -> GeminiClient | OpenAIClient:
    """Return dedicated interactive AI chat client singleton for 'gemini' or 'openai'."""
    global _shared_ai_chat_gemini_client, _shared_ai_chat_openai_client
    if settings is None:
        settings = get_settings()

    target = (provider or getattr(settings, "AI_CHAT_PROVIDER", "gemini") or "gemini").strip().lower()
    if target == "gemini":
        if _shared_ai_chat_gemini_client is not None:
            return _shared_ai_chat_gemini_client
        api_key = (settings.AI_CHAT_API_KEY or settings.GEMINI_API_KEY or "").strip()
        if not api_key:
            raise ValueError("AI_CHAT_API_KEY (or GEMINI_API_KEY) is required for Gemini AI chat")
        http_client = _get_ai_chat_http_client(settings)
        _shared_ai_chat_gemini_client = GeminiClient(
            api_key=api_key,
            model=settings.AI_CHAT_MODEL,
            max_output_tokens=settings.AI_CHAT_MAX_OUTPUT_TOKENS,
            http_client=http_client,
        )
        return _shared_ai_chat_gemini_client

    if target == "openai":
        if _shared_ai_chat_openai_client is not None:
            return _shared_ai_chat_openai_client
        base_url = (getattr(settings, "AI_CHAT_OPENAI_BASE_URL", None) or settings.OPENAI_BASE_URL or "").strip()
        api_key = (getattr(settings, "AI_CHAT_OPENAI_API_KEY", None) or settings.OPENAI_API_KEY or "").strip()
        model = (getattr(settings, "AI_CHAT_OPENAI_MODEL", None) or settings.OPENAI_MODEL or "").strip()
        if not base_url or not api_key or not model:
            raise ValueError(
                "AI_CHAT_OPENAI_BASE_URL/OPENAI_BASE_URL, API_KEY, and MODEL are required for OpenAI AI chat"
            )
        raw_effort = getattr(settings, "AI_REASONING_EFFORT", None)
        reasoning_effort = raw_effort if isinstance(raw_effort, str) and raw_effort.strip() else None
        http_client = _get_ai_chat_http_client(settings)
        _shared_ai_chat_openai_client = OpenAIClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_output_tokens=settings.AI_CHAT_MAX_OUTPUT_TOKENS,
            http_client=http_client,
            reasoning_effort=reasoning_effort,
        )
        return _shared_ai_chat_openai_client

    raise ValueError(f"Unsupported AI_CHAT_PROVIDER: {target}")


async def close_ai_chat_client() -> None:
    """Close dedicated AI chat clients and shared HTTP client."""
    global _shared_ai_chat_gemini_client, _shared_ai_chat_openai_client, _shared_ai_chat_http_client
    for attr_name, client_obj in (
        ("gemini", _shared_ai_chat_gemini_client),
        ("openai", _shared_ai_chat_openai_client),
    ):
        if client_obj is not None:
            try:
                await client_obj.close()
            except Exception:
                pass
            if attr_name == "gemini":
                _shared_ai_chat_gemini_client = None
            else:
                _shared_ai_chat_openai_client = None

    if _shared_ai_chat_http_client is not None and not _shared_ai_chat_http_client.is_closed:
        try:
            await _shared_ai_chat_http_client.aclose()
        finally:
            _shared_ai_chat_http_client = None
