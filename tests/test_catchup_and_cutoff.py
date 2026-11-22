"""Unit tests for summary windowing cutoff, catch-up scheduling, and date heading derivation."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo
import pytest
from aiogram import Bot
from bot.database.db import close_db, init_db
from bot.database import repository
from bot.scheduler.jobs import (
    is_nightly_run_missed,
    run_catchup_if_needed,
    run_summary_for_topic,
)
from bot.services.llm_client import BaseLLMClient
from bot.utils.dates import compute_nightly_cutoff, format_date_heading


def test_compute_nightly_cutoff():
    """Verify compute_nightly_cutoff returns the latest 00:00:00 in TIMEZONE converted to UTC."""
    tz = ZoneInfo("Asia/Tehran")

    # 1. Run at 00:01:00 Asia/Tehran
    run_time_0001 = datetime(2026, 11, 22, 0, 1, 0, tzinfo=tz)
    cutoff_0001 = compute_nightly_cutoff(run_time=run_time_0001, tz_name="Asia/Tehran")
    # Expected: 2026-11-22 00:00:00+03:30 -> 2026-11-21 20:30:00+00:00
    assert cutoff_0001 == datetime(2026, 11, 21, 20, 30, 0, tzinfo=timezone.utc)

    # 2. Run at 08:30:00 Asia/Tehran (e.g. catch-up or non-zero SUMMARY_TIME_HOUR)
    run_time_0830 = datetime(2026, 11, 22, 8, 30, 0, tzinfo=tz)
    cutoff_0830 = compute_nightly_cutoff(run_time=run_time_0830, tz_name="Asia/Tehran")
    assert cutoff_0830 == datetime(2026, 11, 21, 20, 30, 0, tzinfo=timezone.utc)


def test_is_nightly_run_missed():
    """Verify is_nightly_run_missed correctly evaluates scheduled time vs last run."""
    tz = ZoneInfo("Asia/Tehran")

    # Before scheduled time (e.g. 00:00:30 when scheduled is 00:01:00)
    now_before = datetime(2026, 11, 22, 0, 0, 30, tzinfo=tz)
    assert is_nightly_run_missed(now=now_before, last_run_iso=None) is False

    # After scheduled time with no previous run recorded
    now_after = datetime(2026, 11, 22, 9, 0, 0, tzinfo=tz)
    assert is_nightly_run_missed(now=now_after, last_run_iso=None) is True

    # After scheduled time with run completed earlier today (e.g. at 00:02:00)
    today_run = datetime(2026, 11, 22, 0, 2, 0, tzinfo=tz).astimezone(timezone.utc).isoformat()
    assert is_nightly_run_missed(now=now_after, last_run_iso=today_run) is False

    # After scheduled time with run from yesterday
    yesterday_run = datetime(2026, 11, 21, 0, 2, 0, tzinfo=tz).astimezone(timezone.utc).isoformat()
    assert is_nightly_run_missed(now=now_after, last_run_iso=yesterday_run) is True


@pytest.mark.asyncio
async def test_nightly_cutoff_leaves_newer_messages_in_db(tmp_path: Path):
    """Verify messages after cutoff remain in the database for the next day."""
    db_file = tmp_path / "test_cutoff.db"
    await init_db(str(db_file))

    try:
        chat_id = -100111
        thread_id = 5
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, thread_id, "تاپیک تست", is_active=1)

        # Message 1 & 2: before midnight cutoff (e.g. Nov 21 14:00 UTC)
        await repository.insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=1,
            user_id=10,
            sender_name="Ali",
            text="بحث روز قبل بخش ۱",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T14:00:00+00:00",
        )
        await repository.insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=2,
            user_id=11,
            sender_name="Reza",
            text="بحث روز قبل بخش ۲",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-21T18:00:00+00:00",
        )

        # Message 3: after midnight cutoff (e.g. Nov 22 02:00 UTC)
        await repository.insert_message(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=3,
            user_id=12,
            sender_name="Sara",
            text="پیام روز جدید که باید بماند",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-22T02:00:00+00:00",
        )

        # Cutoff = Nov 22 00:00:00 Tehran -> Nov 21 20:30:00 UTC
        cutoff = "2026-11-21T20:30:00+00:00"

        # Check unsummarized before cutoff
        unsummarized_before = await repository.fetch_unsummarized_messages(chat_id, thread_id, before=cutoff)
        assert len(unsummarized_before) == 2
        assert [m["message_id"] for m in unsummarized_before] == [1, 2]

        # Mock LLM Client & Bot
        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate.return_value = "خلاصه مباحث روز قبل"
        bot = AsyncMock(spec=Bot)

        # Run summary for topic with cutoff
        success, details = await run_summary_for_topic(
            bot=bot,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            cutoff=cutoff,
            llm_client=mock_llm,
        )
        assert success is True

        # Verify only messages 1 & 2 were deleted; message 3 stays in DB for next day
        remaining = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(remaining) == 1
        assert remaining[0]["message_id"] == 3
        assert remaining[0]["text"] == "پیام روز جدید که باید بماند"

    finally:
        await close_db()


@pytest.mark.asyncio
async def test_date_heading_derived_from_earliest_message():
    """Verify summary heading date derives from earliest message's Tehran timestamp."""
    # Earliest message on 2026-11-20 18:00 UTC (2026-11-20 21:30 Tehran -> Friday 29 Aban 1405)
    dt_nov20 = datetime(2026, 11, 20, 21, 30, tzinfo=ZoneInfo("Asia/Tehran"))
    heading = format_date_heading(dt_nov20)

    assert "جمعه 29 آبان 1405" in heading
    assert "1405/08/29" in heading
    assert "2026-11-20" in heading


@pytest.mark.asyncio
async def test_run_catchup_if_needed(tmp_path: Path):
    """Verify catch-up executes if scheduled run was missed and sets last_nightly_run in meta table."""
    db_file = tmp_path / "test_catchup.db"
    await init_db(str(db_file))

    try:
        # Initially last_nightly_run is None
        assert await repository.get_last_nightly_run() is None

        bot = AsyncMock(spec=Bot)
        mock_llm = AsyncMock(spec=BaseLLMClient)

        # Run catch-up
        executed = await run_catchup_if_needed(bot, llm_client=mock_llm)
        # Assuming current time is daytime (13:00 Tehran) > 00:01 scheduled time -> executed is True
        assert executed is True

        # Verify last_nightly_run was updated in meta table
        last_run = await repository.get_last_nightly_run()
        assert last_run is not None

        # Subsequent call should NOT re-run
        executed_second = await run_catchup_if_needed(bot, llm_client=mock_llm)
        assert executed_second is False

    finally:
        await close_db()
