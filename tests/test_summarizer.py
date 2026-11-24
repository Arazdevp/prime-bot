"""Tests for summarizer text formatting, headings, spacing, and blue link expansion."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
from aiogram import Bot
import pytest
from bot.database import repository
from bot.database.db import close_db, init_db
from bot.scheduler.jobs import run_summary_for_topic
from bot.services.llm_client import BaseLLMClient, LLMError
from bot.services.summarizer import (
    SYSTEM_PROMPT_CONDENSE,
    SYSTEM_PROMPT_MERGE,
    SYSTEM_PROMPT_SINGLE,
    format_messages,
    reset_llm_pacer,
    sanitize_telegram_html,
    summarize_topic_messages,
)
from bot.utils.text_split import TELEGRAM_MESSAGE_MAX_CHARS, telegram_text_length


def test_format_messages_includes_message_id() -> None:
    raw_msgs = [
        {
            "id": 1,
            "chat_id": -1002458921841,
            "thread_id": 10,
            "message_id": 105,
            "sender_name": "سارا",
            "text": "سلام به همگی، جلسه فردا ساعت ۱۰ است.",
            "media_type": None,
            "reply_to_name": None,
            "reply_to_snippet": None,
            "created_at": "2026-11-21T10:30:00+00:00",
        }
    ]
    transcript = format_messages(raw_msgs)
    assert "[#پیام 105]" in transcript
    assert "سارا" in transcript
    assert "جلسه فردا ساعت ۱۰ است." in transcript


def test_sanitize_converts_html_headings() -> None:
    input_text = (
        "<h1>عنوان تاپیک</h1>"
        "<p>متن مقدمه</p>"
        "<h2>محورهای اصلی</h2>"
        "<p>نکته اول</p>"
        "<h3>عنوان مبحث</h3>"
        "<p>شرح</p>"
    )
    result = sanitize_telegram_html(input_text)

    # <h1>, <h2>, <h3> must not remain in output (preventing Telegram Entity Error)
    assert "<h1>" not in result
    assert "<h2>" not in result
    assert "<h3>" not in result
    assert "<b>عنوان تاپیک</b>" in result
    assert "<b>محورهای اصلی</b>" in result
    assert "<b>عنوان مبحث</b>" in result


def test_sanitize_converts_markdown_headings() -> None:
    input_text = (
        "# عنوان تاپیک\n\n"
        "متن مقدمه\n\n"
        "## سرفصل دوم\n\n"
        "متن سرفصل"
    )
    result = sanitize_telegram_html(input_text)
    assert "<b>عنوان تاپیک</b>" in result
    assert "<b>سرفصل دوم</b>" in result


def test_sanitize_expands_relative_message_links() -> None:
    clean_chat_id = "2458921841"
    input_text = (
        'تصمیم گرفته شد که <a href="#105">جلسه فردا برگزار شود</a> و همچنین '
        '<a href="108">فایل پروژه ارسال گردد</a>.'
    )
    result = sanitize_telegram_html(input_text, clean_chat_id=clean_chat_id)

    assert f'href="https://t.me/c/{clean_chat_id}/105"' in result
    assert f'href="https://t.me/c/{clean_chat_id}/108"' in result
    assert "جلسه فردا برگزار شود" in result


def test_sanitize_preserves_full_telegram_links() -> None:
    input_text = '<a href="https://t.me/c/2458921841/999">پیام اصلی</a>'
    result = sanitize_telegram_html(input_text, clean_chat_id="2458921841")
    assert 'href="https://t.me/c/2458921841/999"' in result
    assert "پیام اصلی" in result


def test_sanitize_preserves_generous_spacing() -> None:
    input_text = "بخش اول\n\n\nبخش دوم\n\n\n\n\n\nبخش سوم"
    result = sanitize_telegram_html(input_text)
    # Consecutive newlines should normalize to 3 newlines (\n\n\n = two empty lines)
    assert "\n\n\n" in result
    assert "\n\n\n\n" not in result


def test_sanitize_href_ampersand_no_double_escape() -> None:
    """Verify & inside href is unescaped and escaped once (never &amp;amp;)."""
    # 1. Raw ampersands in query parameters
    input_text = (
        '<a href="https://example.com/watch?v=abc&t=10s&lang=fa">ویدیو آموزشی</a>'
        ' و متن ساده با علامت & برای آزمایش.'
    )
    result = sanitize_telegram_html(input_text)

    # href must have single &amp;, never &amp;amp;
    assert 'href="https://example.com/watch?v=abc&amp;t=10s&amp;lang=fa"' in result
    assert "&amp;amp;" not in result
    # Text node must have &amp;
    assert "متن ساده با علامت &amp; برای آزمایش." in result

    # 2. Already escaped &amp; in raw href from model
    input_text2 = '<a href="https://example.com/api?a=1&amp;b=2">لینک ای‌پی‌آی</a>'
    result2 = sanitize_telegram_html(input_text2)
    assert 'href="https://example.com/api?a=1&amp;b=2"' in result2
    assert "&amp;amp;" not in result2


def test_sanitize_topic_message_links_non_general_and_general() -> None:
    """Verify 3-part message links for non-General topics and 2-part for General topic."""
    clean_chat_id = "2458921841"

    # Non-General topic: thread_id = 42
    input_non_general = (
        'مبحث اول <a href="#105">لینک پیام نسبی</a> و '
        '<a href="https://t.me/c/2458921841/200">لینک پیام ۲ تایی</a>'
    )
    result_non_general = sanitize_telegram_html(
        input_non_general,
        clean_chat_id=clean_chat_id,
        thread_id=42,
    )
    assert f'href="https://t.me/c/{clean_chat_id}/42/105"' in result_non_general
    assert f'href="https://t.me/c/{clean_chat_id}/42/200"' in result_non_general

    # General topic: thread_id = 1
    input_general = 'مبحث عمومی <a href="#105">پیام عمومی</a>'
    result_general = sanitize_telegram_html(
        input_general,
        clean_chat_id=clean_chat_id,
        thread_id=1,
    )
    assert f'href="https://t.me/c/{clean_chat_id}/105"' in result_general
    assert f"/1/105" not in result_general


def test_prompts_enforce_single_message_budget_and_prioritization() -> None:
    """Verify prompts strictly enforce single-message <= max_chars budget and high-signal selection."""
    for prompt in (SYSTEM_PROMPT_SINGLE, SYSTEM_PROMPT_MERGE):
        assert "{max_chars}" in prompt
        assert "{target_chars}" in prompt
        assert "مگر اینکه" not in prompt
        assert "ترجیحاً" not in prompt
        assert "یک پیام تلگرامی" in prompt
        assert "مهم‌ترین" in prompt or "خیلی مهم" in prompt

    assert "{current_chars}" in SYSTEM_PROMPT_CONDENSE
    assert "{max_chars}" in SYSTEM_PROMPT_CONDENSE
    assert "{target_chars}" in SYSTEM_PROMPT_CONDENSE


@pytest.mark.asyncio
async def test_summarize_topic_messages_triggers_ai_condensation_when_over_budget() -> None:
    """Verify summarize_topic_messages runs SYSTEM_PROMPT_CONDENSE when output exceeds max_rendered_chars."""
    reset_llm_pacer()
    calls: list[tuple[str, str]] = []
    oversized_draft = "📌 <b>خلاصه تاپیک: تست</b>\n\n" + ("• بحث طولانی درباره موضوعات مختلف در گروه. " * 120)
    condensed_final = "📌 <b>خلاصه تاپیک: تست</b>\n\n🔹 <b>محورهای اصلی بحث</b>\n• فقط مهم‌ترین تصمیم‌ها و نکات کلیدی با جملات کامل."

    async def fake_generate(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
        calls.append((system_prompt, user_content))
        if len(calls) == 1:
            return oversized_draft
        return condensed_final

    mock_llm = AsyncMock(spec=BaseLLMClient)
    mock_llm.generate = AsyncMock(side_effect=fake_generate)

    messages = [
        {"id": i, "chat_id": -1002458921841, "thread_id": 10, "message_id": 100 + i, "sender_name": "علی", "text": f"پیام مهم شماره {i}"}
        for i in range(1, 7)
    ]

    with patch("bot.services.summarizer.asyncio.sleep", new_callable=AsyncMock):
        result = await summarize_topic_messages(
            topic_name="تست",
            messages=messages,
            chat_id=-1002458921841,
            thread_id=10,
            min_messages=5,
            llm_client=mock_llm,
            max_rendered_chars=1200,
        )

    assert len(calls) == 2
    assert "پیش‌نویس خلاصه زیر" in calls[1][0]
    assert "1200" in calls[0][0]
    assert result == condensed_final
    assert telegram_text_length(result) <= 1200
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_summarize_topic_messages_skips_condensation_when_only_raw_html_exceeds_4096() -> None:
    """Verify summaries with raw HTML > 4096 due to <a> tags but rendered length <= 4096 need no extra LLM call."""
    reset_llm_pacer()
    links = [
        f'• <a href="https://t.me/c/2458921841/10/{200 + i}">نکته مهم شماره {i} درباره تولید محتوا و استراتژی کانال</a>'
        for i in range(1, 55)
    ]
    rich_html = "📌 <b>خلاصه تاپیک: تست</b>\n\n" + "\n".join(links)
    assert len(rich_html) > TELEGRAM_MESSAGE_MAX_CHARS
    assert telegram_text_length(rich_html) <= TELEGRAM_MESSAGE_MAX_CHARS

    mock_llm = AsyncMock(spec=BaseLLMClient)
    mock_llm.generate = AsyncMock(return_value=rich_html)

    messages = [
        {"id": i, "chat_id": -1002458921841, "thread_id": 10, "message_id": 200 + i, "sender_name": "سارا", "text": f"پیام {i}"}
        for i in range(1, 7)
    ]

    result = await summarize_topic_messages(
        topic_name="تست",
        messages=messages,
        chat_id=-1002458921841,
        thread_id=10,
        min_messages=5,
        llm_client=mock_llm,
    )

    assert mock_llm.generate.call_count == 1
    assert result is not None
    assert telegram_text_length(result) <= TELEGRAM_MESSAGE_MAX_CHARS
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_summarize_topic_messages_condensation_error_falls_back_gracefully() -> None:
    """Verify LLMError during condensation returns best sanitized summary rather than failing the topic."""
    reset_llm_pacer()
    oversized_draft = "📌 <b>خلاصه تاپیک: تست</b>\n\n" + ("• نکته مهم اول در گفتگو. " * 60)
    calls = 0

    async def fail_on_condense(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return oversized_draft
        raise LLMError("Insufficient credits", status_code=402, is_retryable=False)

    mock_llm = AsyncMock(spec=BaseLLMClient)
    mock_llm.generate = AsyncMock(side_effect=fail_on_condense)

    messages = [
        {"id": i, "chat_id": -1001, "thread_id": 1, "message_id": i, "sender_name": "کاربر", "text": f"متن {i}"}
        for i in range(1, 7)
    ]

    result = await summarize_topic_messages(
        topic_name="تست",
        messages=messages,
        chat_id=-1001,
        thread_id=1,
        min_messages=5,
        llm_client=mock_llm,
        max_rendered_chars=600,
    )

    assert calls == 2
    assert result == oversized_draft.strip()
    reset_llm_pacer()


@pytest.mark.asyncio
async def test_run_summary_for_topic_aggregate_reserves_prefix_and_sends_single_message(tmp_path: Path) -> None:
    """Verify aggregate mode reserves header budget and delivers rich HTML summary in 1 single Telegram message."""
    reset_llm_pacer()
    await init_db(str(tmp_path / "agg_single_msg.db"))
    try:
        chat_id = -1002458921841
        thread_id = 3
        agg_thread_id = 8
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.set_group_destination(chat_id, "aggregate", agg_thread_id)
        await repository.upsert_topic(chat_id, thread_id, "The Hub | گفتگوی آزاد", is_active=1)

        for mid in range(1, 5):
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=130000 + mid,
                user_id=10,
                sender_name="آرش",
                text=f"بحث مهم شماره {mid}",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-24T10:00:00+00:00",
            )

        # Build a rich summary with 45 blue links: raw HTML > 5000 chars, rendered text ~3300 chars
        links = [
            f'• <a href="https://t.me/c/2458921841/3/{130000 + i}">نکته بسیار مهم شماره {i} در گفتگوهای امروز هیلز پرایم</a>'
            for i in range(1, 46)
        ]
        rich_summary = "📌 <b>خلاصه تاپیک: The Hub | گفتگوی آزاد</b>\n\n" + "\n".join(links)
        assert len(rich_summary) > TELEGRAM_MESSAGE_MAX_CHARS

        received_system_prompts: list[str] = []

        async def mock_gen(system_prompt: str, user_content: str, allow_truncated: bool = False) -> str:
            received_system_prompts.append(system_prompt)
            return rich_summary

        mock_llm = AsyncMock(spec=BaseLLMClient)
        mock_llm.generate = AsyncMock(side_effect=mock_gen)
        bot = AsyncMock(spec=Bot)

        with patch("bot.services.summarizer.asyncio.sleep", new_callable=AsyncMock):
            ok, _ = await run_summary_for_topic(
                bot=bot,
                chat_id=chat_id,
                thread_id=thread_id,
                mark_summarized=True,
                is_nightly=False,
                llm_client=mock_llm,
            )

        assert ok is True
        # Delivered in ONE single Telegram message (not split despite raw HTML > 4096)
        assert bot.send_message.call_count == 1
        sent_text = bot.send_message.call_args.kwargs["text"]
        assert telegram_text_length(sent_text) <= TELEGRAM_MESSAGE_MAX_CHARS
        assert "تاپیک مبدا: The Hub | گفتگوی آزاد" in sent_text
        # Prompt max_chars reserved space for the aggregate header prefix (< 4096)
        expected_budget = TELEGRAM_MESSAGE_MAX_CHARS - telegram_text_length("📂 <b>تاپیک مبدا: The Hub | گفتگوی آزاد</b>\n\n\n")
        assert f"حداکثر {expected_budget} کاراکتر" in received_system_prompts[0]
    finally:
        await close_db()
        reset_llm_pacer()
