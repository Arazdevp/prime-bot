"""Tests for summarizer text formatting, headings, spacing, and blue link expansion."""

import pytest
from bot.services.summarizer import (
    format_messages,
    sanitize_telegram_html,
)


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
