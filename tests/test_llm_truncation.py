"""Unit tests for LLM token limits, thinking budget, and truncation recovery."""

import json
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from bot.services.llm_client import (
    GeminiClient,
    LLMTruncationError,
    OpenAIClient,
)
from bot.services.summarizer import summarize_topic_messages


@pytest.mark.asyncio
async def test_gemini_client_payload_tokens_and_thinking():
    """Verify Gemini payload includes maxOutputTokens and thinkingConfig."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "خلاصه معتبر"}]},
                "finishReason": "STOP",
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_response)

    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.8-flash",
        max_output_tokens=4096,
        http_client=mock_client,
    )

    result = await client.generate("system prompt", "user content")
    assert result == "خلاصه معتبر"

    # Verify JSON payload
    call_args = mock_client.post.call_args
    assert call_args is not None
    payload = call_args.kwargs.get("json", {})
    gen_config = payload.get("generationConfig", {})
    assert gen_config.get("maxOutputTokens") == 4096
    assert gen_config.get("thinkingConfig") == {"thinkingBudget": 0}


@pytest.mark.asyncio
async def test_gemini_client_truncation_raises_and_persists():
    """Verify Gemini finishReason=MAX_TOKENS raises error on first attempt and returns text when allowed."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "متن ناقص خلاصه‌سازی به دلیل سقف..."}]},
                "finishReason": "MAX_TOKENS",
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_response)

    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.8-flash",
        http_client=mock_client,
    )

    # 1. First attempt (allow_truncated=False) must raise retryable LLMTruncationError
    with pytest.raises(LLMTruncationError) as exc_info:
        await client.generate("sys", "user", allow_truncated=False)

    assert exc_info.value.is_retryable is True
    assert "متن ناقص خلاصه‌سازی" in exc_info.value.truncated_text

    # 2. Retry attempt (allow_truncated=True) must return the truncated text for sanitization
    output = await client.generate("sys", "user", allow_truncated=True)
    assert output == "متن ناقص خلاصه‌سازی به دلیل سقف..."


@pytest.mark.asyncio
async def test_openai_client_payload_tokens():
    """Verify OpenAI-compatible payload includes max_tokens."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "خلاصه کامل"},
                "finish_reason": "stop",
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_response)

    client = OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o",
        max_output_tokens=6000,
        http_client=mock_client,
    )

    result = await client.generate("system prompt", "user content")
    assert result == "خلاصه کامل"

    call_args = mock_client.post.call_args
    payload = call_args.kwargs.get("json", {})
    assert payload.get("max_tokens") == 6000


@pytest.mark.asyncio
async def test_openai_client_truncation_raises_and_persists():
    """Verify OpenAI finish_reason=length raises error on first attempt and returns text when allowed."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "متن ناتمام اوپن‌ای‌آی"},
                "finish_reason": "length",
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_response)

    client = OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o",
        http_client=mock_client,
    )

    # 1. First attempt must raise LLMTruncationError
    with pytest.raises(LLMTruncationError) as exc_info:
        await client.generate("sys", "user", allow_truncated=False)

    assert exc_info.value.is_retryable is True
    assert "متن ناتمام اوپن‌ای‌آی" in exc_info.value.truncated_text

    # 2. Retry attempt must return truncated text
    output = await client.generate("sys", "user", allow_truncated=True)
    assert output == "متن ناتمام اوپن‌ای‌آی"


@pytest.mark.asyncio
async def test_summarize_topic_messages_truncation_recovery_and_sanitize():
    """Verify summarize_topic_messages retries once on truncation and sanitizes persisted output."""
    mock_llm = AsyncMock()

    call_count = 0

    async def mock_generate(system_prompt, user_content, allow_truncated=False):
        nonlocal call_count
        call_count += 1
        if not allow_truncated:
            # First call raises truncation error
            raise LLMTruncationError(
                "Truncated on first attempt",
                truncated_text="📌 <b>خلاصه تاپیک: تست</b>\n<a href='#55'>پیام ناتمام",
            )
        # Second call returns truncated text that has unclosed tags
        return "📌 <b>خلاصه تاپیک: تست</b>\n<a href='#55'>پیام ناتمام"

    mock_llm.generate = AsyncMock(side_effect=mock_generate)

    messages = [
        {"id": i, "chat_id": -1002458921841, "thread_id": 10, "message_id": 50 + i, "sender_name": "علی", "text": f"پیام {i}"}
        for i in range(1, 10)
    ]

    result = await summarize_topic_messages(
        topic_name="تست",
        messages=messages,
        chat_id=-1002458921841,
        thread_id=10,
        min_messages=5,
        llm_client=mock_llm,
    )

    assert result is not None
    # Verify retry occurred
    assert call_count == 2
    # Verify 3-part topic link generated for thread_id=10
    assert 'href="https://t.me/c/2458921841/10/55"' in result
    # Verify unclosed <a> tag was safely closed by sanitizer
    assert result.endswith("</a>")
