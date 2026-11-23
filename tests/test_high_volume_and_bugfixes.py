"""Unit tests verifying high-volume token/rate-limit resilience and core bug fixes."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.types import Chat, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.handlers.admin import cmd_status, cmd_topics
from bot.main import create_bot_session
from bot.scheduler.jobs import (
    nightly_summary_job,
    run_summary_for_topic,
)
from bot.services.llm_client import (
    BaseLLMClient,
    GeminiClient,
    LLMError,
    LLMTruncationError,
    OpenAIClient,
)
from bot.services.publisher import send_summary_parts
from bot.services.summarizer import (
    reset_llm_pacer,
    sanitize_telegram_html,
    summarize_topic_messages,
)
from bot.utils.retry import async_retry
from bot.utils.text_split import smart_split


@pytest.mark.asyncio
async def test_llm_client_rate_limit_headers_openrouter_200_error_and_thought_filter():
    """Verify Retry-After / X-RateLimit-Reset / retryDelay parsing, OpenRouter 200 error body, 408 retryable, list content, and Gemini thought filter."""
    # 1. HTTP 408 is retryable
    assert LLMError("timeout", status_code=408).is_retryable is True

    # 2. OpenAIClient parses Retry-After numeric header on 429
    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.headers = httpx.Headers({"Retry-After": "42"})
    resp_429.text = "rate limit"

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False
    mock_http.post = AsyncMock(return_value=resp_429)

    oa = OpenAIClient("https://openrouter.ai/api/v1", "key", "google/gemini-2.5-flash", http_client=mock_http)
    with pytest.raises(LLMError) as exc_429:
        await oa.generate("sys", "user")
    assert exc_429.value.status_code == 429
    assert exc_429.value.is_retryable is True
    assert exc_429.value.retry_after == 42.0

    # 3. OpenAIClient parses OpenRouter X-RateLimit-Reset (epoch ms) header
    future_reset_ms = str(int((time.time() + 25) * 1000))
    resp_reset = MagicMock(spec=httpx.Response)
    resp_reset.status_code = 429
    resp_reset.headers = httpx.Headers({"X-RateLimit-Reset": future_reset_ms})
    resp_reset.text = "rate limit"
    mock_http.post = AsyncMock(return_value=resp_reset)
    with pytest.raises(LLMError) as exc_reset:
        await oa.generate("sys", "user")
    assert exc_reset.value.retry_after is not None
    assert 20.0 <= exc_reset.value.retry_after <= 26.0

    # 4. OpenRouter HTTP 200 with JSON error payload raises retryable LLMError with inner code
    resp_or_200_err = MagicMock(spec=httpx.Response)
    resp_or_200_err.status_code = 200
    resp_or_200_err.headers = httpx.Headers({})
    resp_or_200_err.text = '{"error":{"message":"Provider overloaded","code":502}}'
    resp_or_200_err.json.return_value = {"error": {"message": "Provider overloaded", "code": 502}}
    mock_http.post = AsyncMock(return_value=resp_or_200_err)
    with pytest.raises(LLMError) as exc_or:
        await oa.generate("sys", "user")
    assert exc_or.value.status_code == 502
    assert exc_or.value.is_retryable is True

    # 5. OpenAIClient handles list-formatted message.content without crashing
    resp_list = MagicMock(spec=httpx.Response)
    resp_list.status_code = 200
    resp_list.headers = httpx.Headers({})
    resp_list.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "text", "text": "بخش اول "},
                        {"type": "text", "text": "بخش دوم"},
                    ]
                },
            }
        ]
    }
    mock_http.post = AsyncMock(return_value=resp_list)
    assert await oa.generate("sys", "user") == "بخش اول بخش دوم"

    # 6. GeminiClient parses JSON retryDelay and filters thought=True parts
    resp_gem_429 = MagicMock(spec=httpx.Response)
    resp_gem_429.status_code = 429
    resp_gem_429.headers = httpx.Headers({})
    resp_gem_429.text = '{"error":{"details":[{"retryDelay":"35s"}]}}'
    mock_http.post = AsyncMock(return_value=resp_gem_429)

    gem = GeminiClient("key", "gemini-2.5-flash", http_client=mock_http)
    with pytest.raises(LLMError) as exc_gem:
        await gem.generate("sys", "user")
    assert exc_gem.value.retry_after == 35.0

    resp_gem_thought = MagicMock(spec=httpx.Response)
    resp_gem_thought.status_code = 200
    resp_gem_thought.headers = httpx.Headers({})
    resp_gem_thought.json.return_value = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "Internal reasoning in English...", "thought": True},
                        {"text": "خلاصه نهایی فارسی"},
                    ]
                },
            }
        ]
    }
    mock_http.post = AsyncMock(return_value=resp_gem_thought)
    assert await gem.generate("sys", "user") == "خلاصه نهایی فارسی"


@pytest.mark.asyncio
async def test_async_retry_429_honors_retry_after_and_spans_minute_window():
    """Verify 429 honors Retry-After and without it sleeps >60s across 3 retries."""
    sleeps: list[float] = []
    cooldowns: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    # 1. With Retry-After=45, recovers on attempt 2
    attempts = 0

    async def flaky_with_header() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMError("rate limited", status_code=429, retry_after=45.0)
        return "ok"

    with patch("bot.utils.retry.asyncio.sleep", side_effect=fake_sleep):
        res = await async_retry(flaky_with_header, retries=3, on_rate_limit=cooldowns.append)
    assert res == "ok"
    assert len(sleeps) == 1
    assert 45.0 <= sleeps[0] <= 47.0
    assert len(cooldowns) == 1

    # 2. Without Retry-After, 3 retries span > 60 seconds total (20+35+50=105s)
    sleeps.clear()
    cooldowns.clear()

    async def always_429() -> str:
        raise LLMError("quota", status_code=429)

    with patch("bot.utils.retry.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMError):
            await async_retry(always_429, retries=3, on_rate_limit=cooldowns.append)
    assert len(sleeps) == 3
    assert sum(sleeps) >= 105.0
    assert len(cooldowns) == 4  # 3 retries + final permanent failure cooldown


@pytest.mark.asyncio
async def test_map_reduce_hierarchical_reduce_and_pacing():
    """Verify huge topics trigger hierarchical reduce so no LLM call overflows CHUNK_MAX_CHARS, with inter-chunk pacing."""
    reset_llm_pacer()
    received_lengths: list[int] = []
    sleeps: list[float] = []

    class VerboseMockLLM(BaseLLMClient):
        async def generate(self, system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            received_lengths.append(len(user_content))
            # Return ~500 chars per map chunk so 10 chunks = ~5000 chars (> 1500 limit -> hierarchical reduce!)
            if "خلاصه نهایی" in user_content:
                return "📌 <b>خلاصه تاپیک: تست پرترافیک</b>\n\n🔹 <b>محورهای اصلی بحث</b>\n• جمع‌بندی نهایی."
            return "• نکته کلیدی مهم از این بخش: " + ("متن خلاصه میانی " * 30)

        async def close(self) -> None:
            pass

    messages = [
        {
            "id": i,
            "chat_id": -100123456,
            "thread_id": 10,
            "message_id": i,
            "sender_name": f"کاربر {i % 5}",
            "text": f"پیام تستی شماره {i} درباره موضوع فنی بسیار مفصل و طولانی " + ("توضیحات " * 12),
            "media_type": None,
            "reply_to_name": None,
            "reply_to_snippet": None,
            "created_at": "2026-11-22T10:00:00+00:00",
        }
        for i in range(1, 101)
    ]

    mock_settings = MagicMock()
    mock_settings.CHUNK_MAX_CHARS = 1500
    mock_settings.AI_MAX_RETRIES = 2
    mock_settings.AI_CHUNK_DELAY_SECONDS = 3.0
    mock_settings.TIMEZONE = "Asia/Tehran"

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    with patch("bot.services.summarizer.get_settings", return_value=mock_settings), patch(
        "bot.services.summarizer.asyncio.sleep", side_effect=fake_sleep
    ):
        result = await summarize_topic_messages(
            topic_name="تست پرترافیک",
            messages=messages,
            chat_id=-100123456,
            thread_id=10,
            min_messages=5,
            llm_client=VerboseMockLLM(),
        )

    assert result is not None
    assert "خلاصه تاپیک: تست پرترافیک" in result
    # Every single LLM user_content (map, intermediate reduce, and final merge) must stay bounded!
    # Allow ~300 chars for the Persian wrapper header/footer around CHUNK_MAX_CHARS=1500
    assert max(received_lengths) <= 1850, f"Oversized prompt detected: {max(received_lengths)}"
    # Pacing slept between consecutive calls (first call had 0 sleep)
    assert len(sleeps) >= len(received_lengths) - 1
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_map_reduce_skips_safety_blocked_chunk_but_raises_on_auth():
    """Verify a single safety-blocked chunk in Map-Reduce is skipped while auth errors still raise."""
    reset_llm_pacer()

    messages = [
        {
            "id": i,
            "chat_id": -100123456,
            "thread_id": 10,
            "message_id": i,
            "sender_name": "علی",
            "text": f"پیام شماره {i} " + ("محتوای بحث فنی " * 15),
            "media_type": None,
            "reply_to_name": None,
            "reply_to_snippet": None,
            "created_at": "2026-11-22T10:00:00+00:00",
        }
        for i in range(1, 35)
    ]

    mock_settings = MagicMock()
    mock_settings.CHUNK_MAX_CHARS = 1200
    mock_settings.AI_MAX_RETRIES = 1
    mock_settings.AI_CHUNK_DELAY_SECONDS = 0.0
    mock_settings.TIMEZONE = "Asia/Tehran"

    # 1. Chunk 2 raises safety block (status_code=200, is_retryable=False) -> skipped, topic still succeeds
    call_idx = 0

    async def safety_on_second(system_prompt, user_content, allow_truncated=False):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 2:
            raise LLMError("Gemini returned no candidates (SAFETY)", status_code=200, is_retryable=False)
        return "📌 <b>خلاصه معتبر</b>"

    mock_llm = AsyncMock(spec=BaseLLMClient)
    mock_llm.generate = AsyncMock(side_effect=safety_on_second)

    with patch("bot.services.summarizer.get_settings", return_value=mock_settings):
        res = await summarize_topic_messages("تست", messages, chat_id=-100123456, thread_id=10, llm_client=mock_llm)
    assert res == "📌 <b>خلاصه معتبر</b>"

    # 2. Auth error (401) must NEVER be skipped
    async def auth_fail(system_prompt, user_content, allow_truncated=False):
        raise LLMError("Invalid API key", status_code=401, is_retryable=False)

    mock_llm.generate = AsyncMock(side_effect=auth_fail)
    with patch("bot.services.summarizer.get_settings", return_value=mock_settings):
        with pytest.raises(LLMError) as exc_auth:
            await summarize_topic_messages("تست", messages, chat_id=-100123456, thread_id=10, llm_client=mock_llm)
    assert exc_auth.value.status_code == 401
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_truncation_retry_failure_falls_back_to_initial_truncated_text():
    """Verify that if retry after LLMTruncationError fails with network/429 error, initial truncated text is sanitized and returned."""
    reset_llm_pacer()
    calls = 0

    async def truncate_then_fail(system_prompt, user_content, allow_truncated=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMTruncationError("max tokens", truncated_text="<b>خلاصه ناتمام اما ارزشمند")
        raise LLMError("network down", status_code=None)

    mock_llm = AsyncMock(spec=BaseLLMClient)
    mock_llm.generate = AsyncMock(side_effect=truncate_then_fail)

    mock_settings = MagicMock()
    mock_settings.CHUNK_MAX_CHARS = 24000
    mock_settings.AI_MAX_RETRIES = 0
    mock_settings.AI_CHUNK_DELAY_SECONDS = 0.0
    mock_settings.TIMEZONE = "Asia/Tehran"

    messages = [
        {"id": i, "chat_id": -1001, "thread_id": 1, "message_id": i, "sender_name": "کاربر", "text": f"متن پیام {i}"}
        for i in range(1, 7)
    ]
    with patch("bot.services.summarizer.get_settings", return_value=mock_settings):
        out = await summarize_topic_messages("عمومی", messages, chat_id=-1001, thread_id=1, llm_client=mock_llm)
    assert out == "<b>خلاصه ناتمام اما ارزشمند</b>"
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_same_day_manual_then_nightly_never_silently_deletes_new_messages(tmp_path: Path):
    """Verify manual summary earlier in the day does NOT cause later same-day messages to be silently deleted."""
    reset_llm_pacer()
    await init_db(str(tmp_path / "same_day.db"))
    try:
        chat_id = -100333
        thread_id = 5
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, thread_id, "بحث روزانه", is_active=1)

        # Morning messages on 2026-11-22
        for mid in (1, 2):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=mid,
                user_id=10,
                sender_name="سارا",
                text=f"پیام صبح شماره {mid}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-22T06:00:00+00:00",
            )

        bot = AsyncMock(spec=Bot)
        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate.side_effect = ["خلاصه نوبت صبح", "خلاصه نوبت عصر", "خلاصه شب بعد از keep"]

        # 1. Manual run at noon (mark_summarized=True)
        with patch("bot.services.summarizer.asyncio.sleep", new_callable=AsyncMock):
            ok1, _ = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=True,
                is_nightly=False,
                llm_client=mock_llm,
            )
            assert ok1 is True
            assert bot.send_message.call_count == 1
            assert await repository.get_summary_record(chat_id, thread_id, "2026-11-22") is None

            # 2. Afternoon messages arrive later on the SAME day (2026-11-22)
            for mid in (3, 4):
                await repository.insert_message(
                    chat_id=chat_id,
                    thread_id=thread_id,
                    message_id=mid,
                    user_id=11,
                    sender_name="رضا",
                    text=f"پیام عصر شماره {mid}",
                    media_type=None,
                    reply_to_message_id=None,
                    reply_to_name=None,
                    reply_to_snippet=None,
                    created_at="2026-11-22T15:00:00+00:00",
                )

            # 3. Nightly run with keep=True first, then newer messages arrive after summary created_at
            bot.send_message.reset_mock()
            ok2, _ = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=False,
                is_nightly=False,
                llm_client=mock_llm,
            )
            assert ok2 is True
            assert bot.send_message.call_count == 1
            assert "خلاصه نوبت عصر" in bot.send_message.call_args.kwargs["text"]

            # Insert a newer message timestamped AFTER the keep summary's created_at
            future_ts = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=5,
                user_id=12,
                sender_name="مریم",
                text="پیام جدیدتر بعد از اجرای keep",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at=future_ts,
            )

            bot.send_message.reset_mock()
            ok3, _ = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=True,
                is_nightly=False,
                llm_client=mock_llm,
            )
            assert ok3 is True
            assert bot.send_message.call_count == 1
            assert "خلاصه شب بعد از keep" in bot.send_message.call_args.kwargs["text"]
            assert await repository.fetch_unsummarized_messages(chat_id, thread_id) == []
    finally:
        await close_db()
        reset_llm_pacer()


@pytest.mark.asyncio
async def test_nightly_job_skips_left_groups_and_preserves_last_run_on_failure(tmp_path: Path):
    """Verify left/disallowed groups are excluded and last_nightly_run is only updated when total_failed == 0."""
    reset_llm_pacer()
    await init_db(str(tmp_path / "nightly_flags.db"))
    try:
        active_chat = -100101
        left_chat = -100102
        disallowed_chat = -100103

        await repository.upsert_known_group(active_chat, "گروه فعال")
        await repository.ensure_group_settings(active_chat, default_min_messages=2)
        await repository.upsert_known_group(left_chat, "گروه ترک‌شده")
        await repository.ensure_group_settings(left_chat, default_min_messages=2)
        await repository.mark_known_group_left(left_chat)
        await repository.upsert_known_group(disallowed_chat, "گروه غیرمجاز")
        await repository.ensure_group_settings(disallowed_chat, default_min_messages=2)

        enabled = await repository.list_enabled_groups()
        assert {g["chat_id"] for g in enabled} == {active_chat, disallowed_chat}

        # Insert 2 messages before cutoff in active_chat and disallowed_chat
        for cid in (active_chat, disallowed_chat):
            for mid in (1, 2):
                await repository.insert_message(
                    chat_id=cid,
                    thread_id=1,
                    message_id=mid,
                    user_id=1,
                    sender_name="کاربر",
                    text=f"پیام مهم {mid}",
                    media_type=None,
                    reply_to_message_id=None,
                    reply_to_name=None,
                    reply_to_snippet=None,
                    created_at="2026-11-21T12:00:00+00:00",
                )

        bot = AsyncMock(spec=Bot)
        failing_llm = AsyncMock(spec=BaseLLMClient)
        failing_llm.generate.side_effect = LLMError("auth failed", status_code=401, is_retryable=False)

        mock_cfg = MagicMock()
        mock_cfg.ALLOWED_CHAT_IDS = [active_chat, left_chat]
        mock_cfg.ADMIN_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.MIN_MESSAGES_PER_TOPIC = 2
        mock_cfg.SEND_DELAY_SECONDS = 0.0
        mock_cfg.CHUNK_MAX_CHARS = 24000
        mock_cfg.AI_MAX_RETRIES = 0
        mock_cfg.AI_CHUNK_DELAY_SECONDS = 0.0

        run_time = datetime(2026, 11, 22, 0, 1, 0, tzinfo=timezone.utc)
        with patch("bot.scheduler.jobs.get_settings", return_value=mock_cfg), patch(
            "bot.services.summarizer.get_settings", return_value=mock_cfg
        ), patch("bot.services.publisher.get_settings", return_value=mock_cfg):
            await nightly_summary_job(bot, llm_client=failing_llm, run_time=run_time)

            # Failed run MUST NOT advance last_nightly_run
            assert await repository.get_last_nightly_run() is None

            # Retry with working LLM -> succeeds only for active_chat (disallowed_chat untouched) and advances last_nightly_run
            working_llm = AsyncMock(spec=BaseLLMClient)
            working_llm.generate.return_value = "خلاصه موفق"
            await nightly_summary_job(bot, llm_client=working_llm, run_time=run_time)
            assert await repository.get_last_nightly_run() is not None
            assert await repository.fetch_unsummarized_messages(active_chat, 1) == []
            assert len(await repository.fetch_unsummarized_messages(disallowed_chat, 1)) == 2
    finally:
        await close_db()
        reset_llm_pacer()


@pytest.mark.asyncio
async def test_set_group_helpers_upsert_missing_row_and_delete_old_summaries(tmp_path: Path):
    """Verify set_group_* create missing group_settings rows and delete_old_summaries purges stale records."""
    await init_db(str(tmp_path / "upsert_settings.db"))
    try:
        await repository.set_group_mode(-100901, "whitelist")
        assert (await repository.get_group_settings(-100901))["mode"] == "whitelist"

        await repository.set_group_enabled(-100902, 0)
        assert (await repository.get_group_settings(-100902))["enabled"] == 0

        await repository.set_group_destination(-100903, "aggregate", 42)
        gs3 = await repository.get_group_settings(-100903)
        assert gs3["destination_mode"] == "aggregate"
        assert gs3["destination_thread_id"] == 42

        await repository.set_group_min_messages(-100904, 15)
        assert (await repository.get_group_settings(-100904))["min_messages"] == 15

        # Test delete_old_summaries
        await repository.save_summary_record(-100901, 1, "2026-11-22", ["fresh"], sent_count=1)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        db = get_db()
        await db.execute(
            "INSERT INTO summaries (chat_id, thread_id, day, parts_json, sent_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (-100901, 2, "2026-11-10", '["old"]', 1, old_ts),
        )
        await db.commit()

        removed = await repository.delete_old_summaries(7)
        assert removed == 1
        assert await repository.get_summary_record(-100901, 2, "2026-11-10") is None
        assert await repository.get_summary_record(-100901, 1, "2026-11-22") is not None
    finally:
        await close_db()


def test_create_bot_session_socks5_and_http_proxies():
    """Verify create_bot_session creates AiohttpSession cleanly for SOCKS5 and HTTP without TypeError."""
    direct = create_bot_session(None)
    assert isinstance(direct, AiohttpSession)

    socks = create_bot_session("socks5://127.0.0.1:10808")
    assert isinstance(socks, AiohttpSession)
    assert socks.proxy == "socks5://127.0.0.1:10808"

    http_s = create_bot_session("http://127.0.0.1:7890")
    assert isinstance(http_s, AiohttpSession)
    assert http_s.proxy == "http://127.0.0.1:7890"


def test_smart_split_progress_guards_entity_integrity_and_no_empty_tag_chunks():
    """Verify smart_split never infinite-loops on oversized tags, avoids splitting &amp;, and skips tag-only chunks."""
    # 1. Oversized <a> tag larger than limit must terminate cleanly
    huge_url = "https://example.com/" + ("a" * 300)
    text_with_huge_tag = f'<a href="{huge_url}">متن لینک مهم</a> و ادامه متن.'
    parts = smart_split(text_with_huge_tag, limit=80)
    assert len(parts) >= 1
    assert all(len(p) <= 80 for p in parts)
    assert "متن لینک مهم" in "".join(parts)

    # 2. Hard cut right inside &amp; does not tear the entity
    no_space_entities = ("x" * 28) + "&amp;" + ("y" * 30)
    parts_ent = smart_split(no_space_entities, limit=30)
    assert all(len(p) <= 30 for p in parts_ent)
    assert "&amp;" in "".join(parts_ent)
    assert not any(p.endswith("&a") or p.endswith("&am") or p.endswith("&") for p in parts_ent)

    # 3. Tag-only empty whitespace chunks are not emitted
    tag_only = "<b>   </b>" + ("متن واقعی " * 30)
    parts_clean = smart_split(tag_only, limit=60)
    assert all(p.strip() != "<b></b>" and p.strip() != "<b>   </b>" for p in parts_clean)


def test_sanitize_telegram_html_urls_code_invalid_links_and_block_tags():
    """Verify __ in URLs/code isn't corrupted, invalid href is unwrapped, nested <a> is prevented, and <p>/<br>/<li> convert cleanly."""
    raw = (
        '<p>بررسی <a href="https://example.com/foo__bar__baz">مستندات</a> و <code>__init__</code></p>'
        "<br/>"
        "<ul><li>مورد اول با <a href=\"#\">لینک نامعتبر</a></li>"
        '<li>مورد دوم با <a href="https://a.com">بیرونی <a href="https://b.com">تودرتو</a></a></li></ul>'
    )
    out = sanitize_telegram_html(raw)
    assert 'href="https://example.com/foo__bar__baz"' in out
    assert "<u>" not in out
    assert "<code>__init__</code>" in out
    assert 'href="#"' not in out
    assert "لینک نامعتبر" in out
    assert "&lt;p&gt;" not in out
    assert "&lt;br" not in out
    assert "• مورد اول" in out
    # No nested <a>
    assert out.count("<a ") == 2  # example.com and a.com only


@pytest.mark.asyncio
async def test_publisher_retries_network_error_and_plain_text_fallback():
    """Verify send_summary_parts retries transient TelegramNetworkError and handles RetryAfter during plain-text fallback."""
    bot = AsyncMock(spec=Bot)
    attempts = 0

    async def flaky_network(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TelegramNetworkError(method="sendMessage", message="ClientOSError")
        return AsyncMock()

    bot.send_message = AsyncMock(side_effect=flaky_network)
    with patch("bot.services.publisher.asyncio.sleep", new_callable=AsyncMock):
        delivered, sent_count = await send_summary_parts(bot, -1001, 1, ["<b>سلام</b>"])
    assert delivered is True
    assert sent_count == 1
    assert attempts == 2

    # HTML parse error -> plain-text fallback -> RetryAfter -> succeeds
    fb_attempts = 0

    async def html_fail_then_flood_then_ok(**kwargs):
        nonlocal fb_attempts
        fb_attempts += 1
        if fb_attempts == 1:
            assert kwargs["parse_mode"] == "HTML"
            raise TelegramBadRequest(method="sendMessage", message="Bad Request: can't parse entities")
        if fb_attempts == 2:
            assert kwargs["parse_mode"] is None
            raise TelegramRetryAfter(method="sendMessage", message="Flood", retry_after=1)
        assert kwargs["parse_mode"] is None
        return AsyncMock()

    bot.send_message = AsyncMock(side_effect=html_fail_then_flood_then_ok)
    with patch("bot.services.publisher.asyncio.sleep", new_callable=AsyncMock):
        delivered2, sent_count2 = await send_summary_parts(bot, -1001, 1, ["<b>متن</b>"])
    assert delivered2 is True
    assert sent_count2 == 1
    assert fb_attempts == 3


@pytest.mark.asyncio
async def test_cmd_topics_splits_long_message_and_cmd_status_aggregate_fallback(tmp_path: Path):
    """Verify /topics splits outputs >4000 chars and /status shows fallback name when aggregate topic has no name."""
    await init_db(str(tmp_path / "admin_cmds.db"))
    try:
        chat_id = -1007777
        await repository.ensure_group_settings(chat_id)
        for tid in range(10, 100):
            await repository.upsert_topic(chat_id, tid, name=f"تاپیک بحث تخصصی شماره {tid} با عنوان نسبتا طولانی")

        msg = MagicMock(spec=Message)
        msg.chat = MagicMock(spec=Chat)
        msg.chat.id = chat_id
        msg.chat.type = "supergroup"
        msg.is_topic_message = True
        msg.message_thread_id = 1
        msg.delete = AsyncMock()
        msg.bot = AsyncMock(spec=Bot)
        msg.bot.send_message = AsyncMock()

        await cmd_topics(msg)
        assert msg.bot.send_message.call_count >= 2
        for call in msg.bot.send_message.call_args_list:
            assert len(call.kwargs["text"]) <= 4000

        # Set aggregate destination to topic 999 (no name in topics table)
        await repository.set_group_destination(chat_id, "aggregate", 999)
        msg.bot.send_message.reset_mock()
        await cmd_status(msg)
        status_text = msg.bot.send_message.call_args.kwargs["text"]
        assert "تاپیک 999" in status_text
        assert "«»" not in status_text
    finally:
        await close_db()
