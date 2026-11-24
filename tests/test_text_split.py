"""Unit tests for text_split module."""

import re
import pytest
from bot.utils.text_split import TELEGRAM_MESSAGE_MAX_CHARS, smart_split, telegram_text_length


def is_tags_balanced(text: str) -> bool:
    """Verify that all opened HTML tags are cleanly and correctly closed in LIFO order."""
    tag_re = re.compile(r"<(/)?([a-zA-Z0-9_-]+)(?:\s+[^>]*?)?>")
    stack: list[str] = []
    for m in tag_re.finditer(text):
        is_close = bool(m.group(1))
        tag_name = m.group(2).lower()
        if not is_close:
            stack.append(tag_name)
        else:
            if not stack:
                return False
            top = stack.pop()
            if top != tag_name:
                return False
    return len(stack) == 0


def test_smart_split_empty_input() -> None:
    """Test smart_split with empty and whitespace inputs."""
    assert smart_split("") == []
    assert smart_split("   ") == []


def test_smart_split_short_text() -> None:
    """Test text shorter than limit returns single part untouched."""
    text = "<b>سلام دنیا</b> این یک متن کوتاه آزمایشی است."
    parts = smart_split(text, limit=100)
    assert len(parts) == 1
    assert parts[0] == text


def test_smart_split_limit_respected() -> None:
    """Test that every part stays strictly within the limit."""
    unit = "این یک جمله تستی به زبان فارسی برای سنجش شکستن هوشمند پیام است.\n"
    text = unit * 50
    limit = 200
    parts = smart_split(text, limit=limit)
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= limit, f"Part exceeds limit ({len(p)} > {limit})"


def test_smart_split_tags_reopened_and_closed() -> None:
    """Test that open HTML tags at split boundaries are closed and re-opened."""
    inner = "محتوای بسیار طولانی درون تگ ضخیم و لینک اختصاصی تلگرام. " * 30
    text = f"<b>مقدمه {inner} <a href='https://example.com/test'>لینک مهم درون متن</a> پایان</b>"
    limit = 250
    parts = smart_split(text, limit=limit)
    assert len(parts) > 1

    for idx, p in enumerate(parts):
        assert len(p) <= limit, f"Part {idx+1} exceeds limit: {len(p)}"
        assert is_tags_balanced(p), f"Part {idx+1} has unbalanced tags: {p}"

    # First part must start with <b>
    assert parts[0].startswith("<b>")
    # Last part must end with </b>
    assert parts[-1].endswith("</b>")


def test_smart_split_very_long_single_line() -> None:
    """Test handling of a very long continuous line without spaces or newlines."""
    long_line = "یک‌کلمه_بسیار_طولانی_" * 50
    limit = 100
    parts = smart_split(long_line, limit=limit)
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= limit


def test_telegram_text_length_rendered_utf16_and_entities() -> None:
    """Verify telegram_text_length ignores HTML tags, unescapes entities, and counts astral emoji as 2 UTF-16 units."""
    # '📌' is U+1F4CC -> 2 UTF-16 code units; '<b>' and '<a href=...>' are excluded; '&amp;' -> '&' (1 unit)
    sample = '📌 <b>عنوان &amp; نکته</b>: <a href="https://t.me/c/2458921841/2/132479">لینک</a>'
    # Visible rendered text: "📌 عنوان & نکته: لینک" -> 2 + 1 + 5 + 1 + 1 + 1 + 4 + 1 + 1 + 4 = 21 UTF-16 units
    assert telegram_text_length(sample) == 21
    assert telegram_text_length("") == 0


def test_smart_split_rendered_keeps_rich_html_links_in_single_message() -> None:
    """Verify a summary with 40 blue links (raw HTML > 5000 chars, rendered <= 4096) stays 1 single message."""
    lines = [
        f'• نکته مهم شماره {i}: <a href="https://t.me/c/2458921841/2/{132400 + i}">توضیح تصمیم گرفته شده درباره بخش {i} در جلسه امروز</a>'
        for i in range(1, 41)
    ]
    rich_summary = "\n".join(lines)
    assert len(rich_summary) > TELEGRAM_MESSAGE_MAX_CHARS
    assert telegram_text_length(rich_summary) <= TELEGRAM_MESSAGE_MAX_CHARS

    parts = smart_split(rich_summary, limit=TELEGRAM_MESSAGE_MAX_CHARS, count_rendered=True)
    assert len(parts) == 1
    assert parts[0] == rich_summary


def test_smart_split_rendered_exceeding_limit_balances_tags_and_entities() -> None:
    """Verify count_rendered=True splits oversized HTML cleanly within rendered limit and balances tags."""
    inner = 'متن تست &amp; نکته با لینک <a href="https://t.me/c/2458921841/2/999">پیام کلیدی</a> و توضیح 📌. ' * 12
    text = f"<b>{inner}</b>"
    limit = 200
    parts = smart_split(text, limit=limit, count_rendered=True)

    assert len(parts) > 1
    for idx, p in enumerate(parts):
        assert telegram_text_length(p) <= limit, f"Part {idx+1} exceeded rendered limit: {telegram_text_length(p)}"
        assert is_tags_balanced(p), f"Part {idx+1} has unbalanced tags: {p}"
        assert not p.endswith("&a") and not p.endswith("&am") and not p.endswith("&amp")
        assert p.strip() not in ("<b></b>", "<b>   </b>")
