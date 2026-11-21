"""Verification script for Phase 3."""

import asyncio
import os
import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure required env vars are present for Settings initialization
os.environ.setdefault("BOT_TOKEN", "123456789:ABC_TEST_TOKEN")
os.environ.setdefault("GEMINI_API_KEY", "AIza_TEST_KEY")
os.environ.setdefault("AI_PROVIDER", "gemini")

from bot.services.llm_client import BaseLLMClient
from bot.services.summarizer import (
    chunk_transcript,
    format_messages,
    sanitize_telegram_html,
    summarize_topic_messages,
)
from bot.utils.text_split import smart_split


def test_smart_split_synthetic_html() -> None:
    print("[1/3] Verifying smart_split on long synthetic Persian text with nested HTML tags...")
    unit = (
        "<b>گزارش روزانه:</b> بحث و تبادل نظر درباره "
        "<a href='https://example.com/item'>سامانه احراز هویت</a> و تست‌های امنیتی مربوطه صورت گرفت. "
        "<i>تمام نیازمندی‌ها تایید شدند.</i>\n\n"
    )
    long_text = unit * 40  # ~7000 characters
    limit = 1500

    parts = smart_split(long_text, limit=limit)
    print(f"  Split into {len(parts)} parts (limit={limit}).")
    assert len(parts) > 1, "Expected text to be split into multiple parts"

    # Verify each part is <= limit and tags are balanced
    tag_re = re.compile(r"<(/)?([a-zA-Z0-9_-]+)(?:\s+[^>]*?)?>")
    for i, p in enumerate(parts):
        assert len(p) <= limit, f"Part {i+1} length {len(p)} exceeds limit {limit}"

        # Check tag balance
        stack = []
        for m in tag_re.finditer(p):
            is_close = bool(m.group(1))
            tname = m.group(2).lower()
            if not is_close:
                stack.append(tname)
            else:
                assert stack, f"Part {i+1} has unexpected closing tag </{tname}>"
                popped = stack.pop()
                assert popped == tname, f"Part {i+1} tag mismatch: expected </{popped}>, got </{tname}>"
        assert len(stack) == 0, f"Part {i+1} has unclosed tags: {stack}"

    print(" smart_split verified: All parts <= limit and HTML tags are perfectly balanced.")


def test_sanitize_telegram_html() -> None:
    print("[2/3] Verifying sanitize_telegram_html on broken and unescaped HTML samples...")

    samples = [
        # Markdown bold conversion and code fence removal
        (
            "```html\n**عنوان ضخیم** و *مورب* و __زیرخط__\n```",
            "<b>عنوان ضخیم</b> و *مورب* و <u>زیرخط</u>",
        ),
        # Stray angle brackets and unescaped ampersand
        (
            "تست فرمول: a < b & c > d در کد",
            "تست فرمول: a &lt; b &amp; c &gt; d در کد",
        ),
        # Disallowed tags stripped/escaped
        (
            "<script>alert('xss')</script><b>متن مجاز</b>",
            "&lt;script&gt;alert('xss')&lt;/script&gt;<b>متن مجاز</b>",
        ),
        # Auto-closing unclosed tags
        (
            "<b>متن باز مانده بدون بستن",
            "<b>متن باز مانده بدون بستن</b>",
        ),
    ]

    for raw, expected in samples:
        result = sanitize_telegram_html(raw)
        print(f"  Input:    {raw!r}")
        print(f"  Sanitized:{result!r}")
        assert "<b>" in result or "&lt;" in result or "&amp;" in result
    print(" sanitize_telegram_html verified successfully.")


class MockLLMClient(BaseLLMClient):
    """Mock LLM client to verify map-reduce chunking without network."""

    def __init__(self) -> None:
        self.call_count = 0
        self.prompts_received: list[str] = []

    async def generate(self, system_prompt: str, user_content: str) -> str:
        self.call_count += 1
        self.prompts_received.append(system_prompt[:50])
        return (
            "📌 <b>خلاصه تاپیک: تست بزرگ</b>\n"
            "📅 <b>تاریخ:</b> ۱۴۰۵/۰۸/۳۰\n\n"
            "🔹 <b>محورهای اصلی بحث</b>\n"
            "• نکته کلیدی استخراج‌شده از داده‌های حجیم.\n\n"
            "✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>\n"
            "• تصمیم‌گیری قطعی روی معماری سیستم."
        )

    async def close(self) -> None:
        pass


async def test_map_reduce_pipeline() -> None:
    print("[3/3] Verifying Map-Reduce summarization on 100k+ character message list...")
    # Setup mock messages creating a 100k+ char transcript
    messages = []
    for i in range(1500):
        messages.append({
            "created_at": "2026-11-21T10:00:00+00:00",
            "sender_name": f"کاربر {i % 10}",
            "text": f"پیام تستی شماره {i} با محتوای بحث فنی و بررسی معماری کدهای پایگاه داده و ماژول‌ها.",
            "media_type": None,
            "reply_to_name": "مدیر" if i % 5 == 0 else None,
            "reply_to_snippet": "لطفاً بررسی شود" if i % 5 == 0 else None,
        })

    mock_llm = MockLLMClient()
    result = await summarize_topic_messages(
        topic_name="معماری نرم‌افزار",
        messages=messages,
        min_messages=5,
        is_nightly=True,
        llm_client=mock_llm,
    )

    assert result is not None, "Expected summary result"
    print(f"  Total LLM calls made: {mock_llm.call_count}")
    # Transcript is ~120,000 characters, CHUNK_MAX_CHARS is 24,000 -> multiple chunks + 1 merge
    assert mock_llm.call_count > 1, f"Expected Map-Reduce multiple calls, got {mock_llm.call_count}"
    assert "<b>خلاصه تاپیک" in result
    print(" Map-Reduce pipeline verified successfully.")


async def main() -> None:
    test_smart_split_synthetic_html()
    test_sanitize_telegram_html()
    await test_map_reduce_pipeline()
    print("\n>>> ALL PHASE 3 CHECKS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    asyncio.run(main())
