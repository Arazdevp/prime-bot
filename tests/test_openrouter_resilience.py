"""Unit tests for OpenRouter HTTP 402 in-flight budget recovery, strict LLM serialization, and chunk checkpointing."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import httpx
import pytest
from aiogram import Bot

from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.scheduler.jobs import run_summary_for_group, run_summary_for_topic
from bot.services.llm_client import BaseLLMClient, LLMError, OpenAIClient
from bot.services.summarizer import reset_llm_pacer, summarize_topic_messages
from bot.utils.retry import async_retry


@pytest.mark.asyncio
async def test_openrouter_402_in_flight_budget_is_retryable_and_honors_retry_after():
    """Verify OpenRouter HTTP 402 in_flight_budget_exhausted (both HTTP 402 & 200-wrapped) is retryable with Retry-After."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False
    oa = OpenAIClient(
        "https://openrouter.ai/api/v1",
        "sk-or-test",
        "deepseek/deepseek-v4.1-flash-1017",
        http_client=mock_http,
    )

    # 1. Exact OpenRouter HTTP 402 in_flight_budget_exhausted response with Retry-After header
    body_in_flight = (
        '{"error":{"message":"This request would exceed your available credits given your current '
        'in-flight requests. Retry after in-flight requests settle, or add credits.","code":402,'
        '"metadata":{"reason":"in_flight_budget_exhausted","limit_source":"openrouter_in_flight_budget"}}}'
    )
    resp_402 = MagicMock(spec=httpx.Response)
    resp_402.status_code = 402
    resp_402.headers = httpx.Headers({"Retry-After": "45"})
    resp_402.text = body_in_flight
    mock_http.post = AsyncMock(return_value=resp_402)

    with pytest.raises(LLMError) as exc_info:
        await oa.generate("sys", "user")

    assert exc_info.value.status_code == 402
    assert exc_info.value.is_retryable is True
    assert exc_info.value.is_throttle is True
    assert exc_info.value.is_in_flight_budget is True
    assert exc_info.value.retry_after == 45.0
    # Body excerpt should retain full metadata reason (not cut off at 200 chars)
    assert "in_flight_budget_exhausted" in (exc_info.value.body_excerpt or "")

    # 2. HTTP 200-wrapped OpenRouter 402 with retry_after inside metadata (>120s allowed up to 300s)
    payload_200_402 = {
        "error": {
            "code": 402,
            "message": "This request would exceed your available credits given your current in-flight requests.",
            "metadata": {
                "reason": "in_flight_budget_exhausted",
                "limit_source": "openrouter_in_flight_budget",
                "retry_after": 180,
            },
        }
    }
    resp_200_402 = MagicMock(spec=httpx.Response)
    resp_200_402.status_code = 200
    resp_200_402.headers = httpx.Headers({})
    resp_200_402.text = '{"error":{"code":402}}'
    resp_200_402.json.return_value = payload_200_402
    mock_http.post = AsyncMock(return_value=resp_200_402)

    with pytest.raises(LLMError) as exc_200:
        await oa.generate("sys", "user")

    assert exc_200.value.status_code == 402
    assert exc_200.value.is_retryable is True
    assert exc_200.value.is_throttle is True
    assert exc_200.value.is_in_flight_budget is True
    assert exc_200.value.retry_after == 180.0

    # 3. Genuine permanent 402 (empty balance / openrouter_credits) MUST fail fast (is_retryable=False)
    resp_402_empty = MagicMock(spec=httpx.Response)
    resp_402_empty.status_code = 402
    resp_402_empty.headers = httpx.Headers({})
    resp_402_empty.text = (
        '{"error":{"message":"Insufficient credits. Add more at openrouter.ai","code":402,'
        '"metadata":{"reason":"weight_exceeds_budget","limit_source":"openrouter_credits"}}}'
    )
    mock_http.post = AsyncMock(return_value=resp_402_empty)

    with pytest.raises(LLMError) as exc_empty:
        await oa.generate("sys", "user")

    assert exc_empty.value.status_code == 402
    assert exc_empty.value.is_retryable is False
    assert exc_empty.value.is_throttle is False
    assert exc_empty.value.is_in_flight_budget is False


@pytest.mark.asyncio
async def test_openrouter_402_can_only_afford_auto_clamps_max_tokens_and_reasoning_effort():
    """Verify 'can only afford N' auto-clamps max_tokens once (>=512), fails fast if <512, and passes reasoning_effort."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False

    resp_afford = MagicMock(spec=httpx.Response)
    resp_afford.status_code = 402
    resp_afford.headers = httpx.Headers({})
    resp_afford.text = (
        '{"error":{"message":"This request requires more credits, or fewer max_tokens. '
        'You requested up to 8192 tokens, but can only afford 3000.","code":402}}'
    )

    resp_ok = MagicMock(spec=httpx.Response)
    resp_ok.status_code = 200
    resp_ok.headers = httpx.Headers({})
    resp_ok.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "خلاصه موفق با سقف توکن تعدیل‌شده"}}]
    }

    mock_http.post = AsyncMock(side_effect=[resp_afford, resp_ok])
    oa = OpenAIClient(
        "https://openrouter.ai/api/v1",
        "sk-or-test",
        "deepseek/deepseek-v4.1-flash-1017",
        max_output_tokens=8192,
        http_client=mock_http,
        reasoning_effort="none",
    )

    result = await oa.generate("sys", "user")
    assert result == "خلاصه موفق با سقف توکن تعدیل‌شده"
    assert mock_http.post.call_count == 2

    first_payload = mock_http.post.call_args_list[0].kwargs["json"]
    second_payload = mock_http.post.call_args_list[1].kwargs["json"]
    assert first_payload["max_tokens"] == 8192
    assert first_payload["reasoning"] == {"effort": "none"}
    assert second_payload["max_tokens"] == 2700  # int(3000 * 0.9)

    # Affordable tokens < 512 must NOT clamp; fails fast as non-retryable 402
    resp_tiny = MagicMock(spec=httpx.Response)
    resp_tiny.status_code = 402
    resp_tiny.headers = httpx.Headers({})
    resp_tiny.text = '{"error":{"message":"You requested up to 8192 tokens, but can only afford 120.","code":402}}'
    mock_http.post = AsyncMock(return_value=resp_tiny)

    with pytest.raises(LLMError) as exc_tiny:
        await oa.generate("sys", "user")
    assert exc_tiny.value.status_code == 402
    assert exc_tiny.value.is_retryable is False
    assert mock_http.post.call_count == 1


@pytest.mark.asyncio
async def test_async_retry_handles_in_flight_402_with_throttle_retries_and_callbacks():
    """Verify async_retry uses throttle_retries, respects Retry-After up to 300s, and fires in-flight callback."""
    sleeps: list[float] = []
    rate_cooldowns: list[float] = []
    in_flight_cooldowns: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    attempts = 0

    async def flaky_in_flight() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            raise LLMError(
                "in-flight budget exhausted",
                status_code=402,
                is_retryable=True,
                retry_after=150.0 if attempts == 1 else None,
                is_throttle=True,
                is_in_flight_budget=True,
            )
        return "recovered"

    with patch("bot.utils.retry.asyncio.sleep", side_effect=fake_sleep):
        res = await async_retry(
            flaky_in_flight,
            retries=2,
            throttle_retries=5,
            on_rate_limit=rate_cooldowns.append,
            on_in_flight_budget=in_flight_cooldowns.append,
        )

    assert res == "recovered"
    assert attempts == 5
    assert len(sleeps) == 4
    # First retry honors Retry-After=150s (> old 120s cap!)
    assert 150.0 <= sleeps[0] <= 152.0
    assert len(rate_cooldowns) == 4
    assert len(in_flight_cooldowns) == 4


@pytest.mark.asyncio
async def test_summarizer_serializes_concurrent_calls_and_adapts_pacing_after_in_flight_402():
    """Verify strictly 1 in-flight LLM request at a time and adaptive inter-chunk slowdown after 402."""
    reset_llm_pacer()
    active_in_flight = 0
    max_concurrent = 0
    recorded_sleeps: list[float] = []
    real_sleep = asyncio.sleep

    class TrackingLLM(BaseLLMClient):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            nonlocal active_in_flight, max_concurrent
            active_in_flight += 1
            max_concurrent = max(max_concurrent, active_in_flight)
            try:
                # Yield to event loop so any concurrent caller would enter generate() if serialization failed
                await real_sleep(0.02)
                self.calls += 1
                if self.calls == 1:
                    raise LLMError(
                        "OpenRouter in-flight budget",
                        status_code=402,
                        is_retryable=True,
                        retry_after=15.0,
                        is_throttle=True,
                        is_in_flight_budget=True,
                    )
                return "📌 <b>خلاصه موفق</b>"
            finally:
                active_in_flight -= 1

        async def close(self) -> None:
            pass

    mock_settings = MagicMock()
    mock_settings.CHUNK_MAX_CHARS = 24000
    mock_settings.AI_MAX_RETRIES = 2
    mock_settings.AI_THROTTLE_MAX_RETRIES = 5
    mock_settings.AI_CHUNK_DELAY_SECONDS = 2.0
    mock_settings.TIMEZONE = "Asia/Tehran"

    msgs = [
        {
            "id": i,
            "chat_id": -100999,
            "thread_id": 10,
            "message_id": i,
            "sender_name": "کاربر",
            "text": f"پیام تستی شماره {i}",
            "created_at": "2026-11-24T08:00:00+00:00",
        }
        for i in range(1, 7)
    ]

    llm = TrackingLLM()

    async def capture_sleep(sec: float) -> None:
        recorded_sleeps.append(sec)
        await real_sleep(0)

    with patch("bot.services.summarizer.get_settings", return_value=mock_settings), patch(
        "asyncio.sleep", side_effect=capture_sleep
    ):
        # Launch two concurrent summarizations (e.g. nightly + admin preview)
        res1, res2 = await asyncio.gather(
            summarize_topic_messages("تاپیک ۱", msgs, chat_id=-100999, thread_id=10, llm_client=llm),
            summarize_topic_messages("تاپیک ۲", msgs, chat_id=-100999, thread_id=11, llm_client=llm),
        )

    assert res1 == "📌 <b>خلاصه موفق</b>"
    assert res2 == "📌 <b>خلاصه موفق</b>"
    # Strictly single in-flight request across concurrent callers
    assert max_concurrent == 1
    # 1st sleep: retry backoff honoring Retry-After (~15s + jitter)
    # 2nd sleep: second caller's adaptive pacing (2.0 base + 10.0 adaptive = ~12.0s)
    assert len(recorded_sleeps) == 2
    assert 15.0 <= recorded_sleeps[0] <= 17.0
    assert 10.0 <= recorded_sleeps[1] <= 13.0
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_failed_topic_in_flight_402_enforces_cooldown_before_next_topic(tmp_path: Path):
    """Verify if topic 1 exhausts in-flight 402 retries, topic 2 sleeps the shared cooldown before calling LLM."""
    reset_llm_pacer()
    await init_db(str(tmp_path / "cross_topic_cooldown.db"))
    try:
        chat_id = -100555
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, 10, "The Hub", is_active=1)
        await repository.upsert_topic(chat_id, 20, "DEEP WORK", is_active=1)

        for tid in (10, 20):
            for idx in range(1, 3):
                await repository.insert_message(
                    chat_id=chat_id,
                    thread_id=tid,
                    message_id=tid * 100 + idx,
                    user_id=1,
                    sender_name="آرش",
                    text=f"پیام تستی تاپیک {tid} شماره {idx}",
                    media_type=None,
                    reply_to_message_id=None,
                    reply_to_name=None,
                    reply_to_snippet=None,
                    created_at="2026-11-24T06:00:00+00:00",
                )

        topic_calls = 0

        async def fail_first_topic_succeed_second(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            nonlocal topic_calls
            topic_calls += 1
            if "The Hub" in user_content:
                raise LLMError(
                    "in-flight budget exhausted",
                    status_code=402,
                    is_retryable=True,
                    retry_after=40.0,
                    is_throttle=True,
                    is_in_flight_budget=True,
                )
            return "📌 <b>خلاصه DEEP WORK</b>"

        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate = AsyncMock(side_effect=fail_first_topic_succeed_second)
        bot = AsyncMock(spec=Bot)

        recorded_sleeps: list[float] = []
        mock_cfg = MagicMock()
        mock_cfg.ADMIN_IDS = []
        mock_cfg.ALLOWED_CHAT_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.MIN_MESSAGES_PER_TOPIC = 2
        mock_cfg.SEND_DELAY_SECONDS = 0.0
        mock_cfg.CHUNK_MAX_CHARS = 24000
        mock_cfg.AI_MAX_RETRIES = 0
        mock_cfg.AI_THROTTLE_MAX_RETRIES = 0
        mock_cfg.AI_CHUNK_DELAY_SECONDS = 0.0

        async def capture_sleep(sec: float) -> None:
            recorded_sleeps.append(sec)

        with patch("bot.scheduler.jobs.get_settings", return_value=mock_cfg), patch(
            "bot.services.summarizer.get_settings", return_value=mock_cfg
        ), patch("bot.services.publisher.get_settings", return_value=mock_cfg), patch(
            "asyncio.sleep", side_effect=capture_sleep
        ):
            report = await run_summary_for_group(bot, chat_id, mark_summarized=True, llm_client=mock_llm)

        assert report == {"success": 1, "skipped": 0, "failed": 1}
        assert topic_calls == 2
        # Topic 2 MUST have waited for the 40s in-flight cooldown recorded by Topic 1's failure!
        assert any(s >= 35.0 for s in recorded_sleeps), f"Expected cooldown sleep >=35s, got {recorded_sleeps}"
    finally:
        await close_db()
        reset_llm_pacer()


@pytest.mark.asyncio
async def test_map_reduce_chunk_checkpointing_resumes_without_repaying_and_cleans_up(tmp_path: Path):
    """Verify completed Map chunks persist across failed runs, dry_run never writes cache, and cleanup purges stale rows."""
    reset_llm_pacer()
    await init_db(str(tmp_path / "chunk_cache.db"))
    try:
        chat_id = -100888
        thread_id = 42
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, thread_id, "The Hub", is_active=1)

        # Insert enough messages to produce 3+ map chunks at CHUNK_MAX_CHARS=1200
        for mid in range(1, 35):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=mid,
                user_id=10,
                sender_name="سارا",
                text=f"پیام طولانی شماره {mid} درباره پروژه هیلز " + ("جزئیات فنی مهم " * 8),
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at=f"2026-11-24T06:{mid:02d}:00+00:00",
            )

        bot = AsyncMock(spec=Bot)
        mock_cfg = MagicMock()
        mock_cfg.ADMIN_IDS = []
        mock_cfg.ALLOWED_CHAT_IDS = []
        mock_cfg.TIMEZONE = "Asia/Tehran"
        mock_cfg.MIN_MESSAGES_PER_TOPIC = 2
        mock_cfg.SEND_DELAY_SECONDS = 0.0
        mock_cfg.CHUNK_MAX_CHARS = 1200
        mock_cfg.AI_MAX_RETRIES = 0
        mock_cfg.AI_THROTTLE_MAX_RETRIES = 0
        mock_cfg.AI_CHUNK_DELAY_SECONDS = 0.0

        # 1. Dry-run MUST NOT write anything to map_chunk_cache
        dry_llm = AsyncMock(spec=BaseLLMClient)
        dry_llm.generate = AsyncMock(return_value="📌 <b>پیش‌نمایش</b>")
        with patch("bot.scheduler.jobs.get_settings", return_value=mock_cfg), patch(
            "bot.services.summarizer.get_settings", return_value=mock_cfg
        ), patch("bot.services.publisher.get_settings", return_value=mock_cfg):
            ok_dry, _ = await run_summary_for_topic(
                bot, chat_id, thread_id, mark_summarized=False, is_nightly=False, llm_client=dry_llm, dry_run=True
            )
        assert ok_dry is True
        db = get_db()
        cur = await db.execute("SELECT COUNT(*) AS cnt FROM map_chunk_cache;")
        assert (await cur.fetchone())["cnt"] == 0

        # 2. First real run: chunks 1 & 2 succeed, chunk 3 fails with permanent 402
        run1_calls = 0

        async def fail_on_chunk_3(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            nonlocal run1_calls
            run1_calls += 1
            if run1_calls == 3:
                raise LLMError("Insufficient credits", status_code=402, is_retryable=False)
            return f"خلاصه بخش {run1_calls}"

        llm1 = AsyncMock(spec=BaseLLMClient)
        llm1.generate = AsyncMock(side_effect=fail_on_chunk_3)

        with patch("bot.scheduler.jobs.get_settings", return_value=mock_cfg), patch(
            "bot.services.summarizer.get_settings", return_value=mock_cfg
        ), patch("bot.services.publisher.get_settings", return_value=mock_cfg):
            ok1, _ = await run_summary_for_topic(
                bot, chat_id, thread_id, mark_summarized=True, is_nightly=False, llm_client=llm1
            )
        assert ok1 is False
        assert run1_calls == 3

        # Exactly 2 completed chunks must be cached in SQLite!
        cur = await db.execute("SELECT COUNT(*) AS cnt FROM map_chunk_cache WHERE chat_id = ? AND thread_id = ?;", (chat_id, thread_id))
        cached_after_fail = (await cur.fetchone())["cnt"]
        assert cached_after_fail == 2

        # 3. Second real run: reuses chunks 1 & 2 from SQLite without calling LLM for them!
        run2_prompts: list[str] = []

        async def succeed_all(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            run2_prompts.append(user_content)
            if "ترکیب کرده" in user_content:
                return "📌 <b>خلاصه نهایی کامل همه پیام‌ها</b>"
            return "خلاصه بخش باقی‌مانده"

        llm2 = AsyncMock(spec=BaseLLMClient)
        llm2.generate = AsyncMock(side_effect=succeed_all)

        with patch("bot.scheduler.jobs.get_settings", return_value=mock_cfg), patch(
            "bot.services.summarizer.get_settings", return_value=mock_cfg
        ), patch("bot.services.publisher.get_settings", return_value=mock_cfg):
            ok2, _ = await run_summary_for_topic(
                bot, chat_id, thread_id, mark_summarized=True, is_nightly=False, llm_client=llm2
            )

        assert ok2 is True
        # Chunks 1 & 2 were skipped via cache, so neither "بخش 1 از" nor "بخش 2 از" was sent to LLM!
        assert not any(p.startswith("بخش 1 از ") for p in run2_prompts)
        assert not any(p.startswith("بخش 2 از ") for p in run2_prompts)
        assert any(p.startswith("بخش 3 از ") for p in run2_prompts)
        # Final merge received the cached partial summaries from run 1
        merge_prompt = run2_prompts[-1]
        assert "خلاصه بخش 1" in merge_prompt
        assert "خلاصه بخش 2" in merge_prompt

        # Cache must be cleared after final summary is persisted
        cur = await db.execute("SELECT COUNT(*) AS cnt FROM map_chunk_cache WHERE chat_id = ? AND thread_id = ?;", (chat_id, thread_id))
        assert (await cur.fetchone())["cnt"] == 0

        # 4. Verify delete_old_cached_map_chunks cleans up expired rows only
        await repository.save_cached_map_chunk(chat_id, thread_id, "fresh_hash", "fresh")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        await db.execute(
            "INSERT INTO map_chunk_cache (chat_id, thread_id, chunk_hash, summary_text, created_at) "
            "VALUES (?, ?, ?, ?, ?);",
            (chat_id, thread_id, "stale_hash", "stale", old_ts),
        )
        await db.commit()

        deleted = await repository.delete_old_cached_map_chunks(7)
        assert deleted == 1
        assert await repository.get_cached_map_chunk(chat_id, thread_id, "stale_hash") is None
        assert await repository.get_cached_map_chunk(chat_id, thread_id, "fresh_hash") == "fresh"
    finally:
        await close_db()
        reset_llm_pacer()


@pytest.mark.asyncio
async def test_migration_v5_to_v6_creates_map_chunk_cache_table(tmp_path: Path):
    """Upgrading from user_version=5 adds map_chunk_cache table while preserving existing rows."""
    db_file = tmp_path / "v5_upgrade.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript(
        """
        CREATE TABLE group_settings (
            chat_id INTEGER PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'blacklist',
            destination_mode TEXT NOT NULL DEFAULT 'per_topic',
            destination_thread_id INTEGER,
            enabled INTEGER NOT NULL DEFAULT 1,
            min_messages INTEGER NOT NULL DEFAULT 5
        );
        INSERT INTO group_settings (chat_id, mode, destination_mode, destination_thread_id, enabled, min_messages)
        VALUES (-100777, 'blacklist', 'per_topic', NULL, 1, 5);
        PRAGMA user_version = 5;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        assert (await cur.fetchone())[0] >= 6

        gs = await repository.get_group_settings(-100777)
        assert gs is not None and gs["min_messages"] == 5

        await repository.save_cached_map_chunk(-100777, 10, "hash_abc", "خلاصه کش‌شده")
        assert await repository.get_cached_map_chunk(-100777, 10, "hash_abc") == "خلاصه کش‌شده"
    finally:
        await close_db()
