"""Unit tests for text_split module."""

import re
import pytest
from bot.utils.text_split import smart_split


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
