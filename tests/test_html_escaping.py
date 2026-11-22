"""Unit tests for HTML escaping helper and dynamic string formatting."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiogram import Bot
from bot.scheduler.jobs import notify_admins_failure
from bot.utils.html_utils import escape_html


def test_escape_html_basic():
    """Verify escape_html handles special characters including R&D <x>."""
    assert escape_html("R&D <x>") == "R&amp;D &lt;x&gt;"
    assert escape_html("Team 'Alpha' & \"Beta\"") == "Team &#x27;Alpha&#x27; &amp; &quot;Beta&quot;"
    assert escape_html("Clean text 123") == "Clean text 123"
    assert escape_html(None) == ""
    assert escape_html(12345) == "12345"
    assert escape_html("") == ""


def test_escape_html_nested_tags():
    """Verify HTML tags in dynamic strings are converted to entities."""
    raw = "<script>alert('xss')</script>"
    escaped = escape_html(raw)
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&lt;/script&gt;" in escaped


@pytest.mark.asyncio
async def test_notify_admins_failure_escapes_dynamic_inputs():
    """Verify notify_admins_failure escapes chat titles, topic names, and error messages."""
    mock_bot = AsyncMock(spec=Bot)
    mock_bot.send_message = AsyncMock()

    with patch("bot.scheduler.jobs.get_settings") as mock_settings:
        mock_settings.return_value.ADMIN_IDS = [111, 222]

        await notify_admins_failure(
            bot=mock_bot,
            chat_title="گروه <تست & توسعه>",
            topic_name="R&D <x>",
            error_type="CustomError<API>",
            error_message="خطای 400: <unclosed tag> & param=test",
        )

        assert mock_bot.send_message.call_count == 2
        for call in mock_bot.send_message.call_args_list:
            text = call.kwargs["text"]
            # Assert dynamic strings were properly escaped
            assert "گروه &lt;تست &amp; توسعه&gt;" in text
            assert "R&amp;D &lt;x&gt;" in text
            assert "CustomError&lt;API&gt;" in text
            assert "&lt;unclosed tag&gt; &amp; param=test" in text
            # Ensure raw unescaped fragments are NOT present
            assert "R&D <x>" not in text
            assert "<unclosed tag>" not in text
            # Ensure legitimate Telegram HTML tags remain
            assert "<b>" in text and "</b>" in text
            assert "<code>" in text and "</code>" in text
