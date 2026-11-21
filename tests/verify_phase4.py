"""Verification script for Phase 4."""

import asyncio
from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set test environment
os.environ["BOT_TOKEN"] = "123456789:ABC_TEST_TOKEN"
os.environ["AI_PROVIDER"] = "gemini"
os.environ["GEMINI_API_KEY"] = "AIza_TEST_KEY"
os.environ["SUMMARY_TIME_HOUR"] = "0"
os.environ["SUMMARY_TIME_MINUTE"] = "1"
os.environ["TIMEZONE"] = "Asia/Tehran"

from bot import main
from bot.database.db import init_db, close_db
from bot.database import repository
from bot.scheduler.jobs import build_scheduler, run_summary_for_topic
from bot.services.llm_client import BaseLLMClient


class MockLLM(BaseLLMClient):
    async def generate(self, system_prompt: str, user_content: str) -> str:
        return (
            "📌 <b>خلاصه تاپیک: آزمایشی</b>\n"
            "📅 <b>تاریخ:</b> ۱۴۰۵/۰۸/۳۰\n\n"
            "🔹 <b>محورهای اصلی بحث</b>\n"
            "• تست موفقیت‌آمیز پایپ‌لاین خلاصه."
        )

    async def close(self) -> None:
        pass


def test_main_import_and_scheduler_trigger() -> None:
    print("[1/3] Verifying main module import and scheduler trigger calculation...")
    assert main is not None
    assert main.create_bot_session is not None

    mock_bot = MagicMock()
    scheduler = build_scheduler(mock_bot)
    job = scheduler.get_job("nightly_summary")
    assert job is not None, "nightly_summary job not found in scheduler"

    # Compute next run time using datetime in Tehran timezone
    tz = ZoneInfo("Asia/Tehran")
    now_tehran = datetime.now(tz)
    next_fire = job.trigger.get_next_fire_time(None, now_tehran)
    assert next_fire is not None
    print(f"  Next fire time computed by APScheduler: {next_fire}")
    assert next_fire.hour == 0, f"Expected hour 0, got {next_fire.hour}"
    assert next_fire.minute == 1, f"Expected minute 1, got {next_fire.minute}"
    print(" Scheduler correctly configured for 00:01 in Asia/Tehran.")


async def test_run_summary_pipeline_db_state() -> None:
    print("[2/3] Verifying run_summary_for_topic against SQLite with mocked LLM & Bot...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = os.path.join(tmp_dir, "test.db")
        await init_db(test_db)

        chat_id = -100555555
        thread_id = 42

        # Ensure group and topic are active
        await repository.ensure_group_settings(chat_id, default_min_messages=5)
        await repository.set_topic_active_status(chat_id, thread_id, is_active=1, name="تست")

        mock_llm = MockLLM()

        # CASE 1: Below threshold (< 5 messages) -> No send, marked summarized
        print("  - Case 1: Testing below threshold behavior (3 messages)...")
        for i in range(3):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=100 + i,
                user_id=1,
                sender_name="تستر",
                text=f"پیام تستی {i}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-21T10:00:00Z",
            )

        mock_bot_fail_send = AsyncMock()
        success, details = await run_summary_for_topic(
            bot=mock_bot_fail_send,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            is_nightly=True,
            llm_client=mock_llm,
        )
        assert success is False
        assert "کمتر از حداقل حد نصاب" in details
        # Verify bot was NEVER called
        assert mock_bot_fail_send.send_message.call_count == 0
        # Verify messages were marked summarized (so they don't leak)
        unsum_after_skip = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(unsum_after_skip) == 0, "Expected below-threshold messages to be marked summarized"
        print("    -> Below threshold verified: No send, marked summarized.")

        # CASE 2: Send fails -> Messages NOT marked summarized
        print("  - Case 2: Testing publish failure (messages should remain unsummarized)...")
        for i in range(6):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=200 + i,
                user_id=1,
                sender_name="تستر",
                text=f"پیام معنادار برای خلاصه {i}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-21T10:00:00Z",
            )

        # Mock bot raising exception during send
        mock_bot_error = AsyncMock()
        mock_bot_error.send_message.side_effect = RuntimeError("Telegram connection timeout")

        success, details = await run_summary_for_topic(
            bot=mock_bot_error,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            is_nightly=True,
            llm_client=mock_llm,
        )
        assert success is False
        # Verify messages are STILL unsummarized
        unsum_after_fail = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(unsum_after_fail) == 6, f"Expected 6 unsummarized messages, found {len(unsum_after_fail)}"
        print("    -> Publish failure verified: Messages were NOT marked summarized.")

        # CASE 3: Send succeeds -> Messages marked summarized
        print("  - Case 3: Testing publish success (messages should be marked summarized)...")
        mock_bot_ok = AsyncMock()
        success, details = await run_summary_for_topic(
            bot=mock_bot_ok,
            chat_id=chat_id,
            thread_id=thread_id,
            mark_summarized=True,
            is_nightly=True,
            llm_client=mock_llm,
        )
        assert success is True
        assert mock_bot_ok.send_message.call_count >= 1
        # Verify messages are now marked summarized
        unsum_after_ok = await repository.fetch_unsummarized_messages(chat_id, thread_id)
        assert len(unsum_after_ok) == 0, f"Expected 0 unsummarized messages, found {len(unsum_after_ok)}"
        print("    -> Publish success verified: Messages successfully marked as summarized.")

        await close_db()

    print("[3/3] Phase 4 resilience and pipeline verification completed successfully.")


async def main_test() -> None:
    test_main_import_and_scheduler_trigger()
    await test_run_summary_pipeline_db_state()
    print("\n>>> ALL PHASE 4 CHECKS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    asyncio.run(main_test())
