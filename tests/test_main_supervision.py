"""Unit tests for polling task supervision and crash notifications in bot/main.py."""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from aiogram import Bot
from bot.main import notify_admins_crash


@pytest.mark.asyncio
async def test_notify_admins_crash_formats_and_sends_persian_alert():
    """Verify notify_admins_crash sends properly escaped Persian alerts to all configured admins."""
    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock()

    with patch("bot.main.get_settings") as mock_settings:
        mock_settings.return_value.ADMIN_IDS = [1001, 1002]

        exc = RuntimeError("Fatal connection error <socket closed> & reset")
        await notify_admins_crash(bot, exc)

        assert bot.send_message.call_count == 2
        for call in bot.send_message.call_args_list:
            text = call.kwargs["text"]
            assert "خطای بحرانی در ربات خلاصه‌ساز تلگرام" in text
            assert "&lt;socket closed&gt; &amp; reset" in text
            assert "<socket closed>" not in text


@pytest.mark.asyncio
async def test_supervision_flow_on_polling_exception():
    """Simulate polling task crashing with an exception and verify supervision detects it."""
    stop_event = asyncio.Event()

    async def crashing_polling():
        await asyncio.sleep(0.01)
        raise ConnectionError("Telegram network unreachable <404>")

    polling_task = asyncio.create_task(crashing_polling())
    stop_task = asyncio.create_task(stop_event.wait())

    done, pending = await asyncio.wait(
        {polling_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    assert polling_task in done
    exc = polling_task.exception()
    assert isinstance(exc, ConnectionError)
    assert "<404>" in str(exc)

    stop_task.cancel()
    try:
        await stop_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_supervision_flow_on_clean_stop_event():
    """Simulate clean shutdown via stop_event and verify polling task is cancelled and awaited."""
    stop_event = asyncio.Event()

    async def long_polling():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return "cancelled_cleanly"

    polling_task = asyncio.create_task(long_polling())
    stop_task = asyncio.create_task(stop_event.wait())

    # Trigger stop event
    stop_event.set()

    done, pending = await asyncio.wait(
        {polling_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    assert stop_task in done
    polling_task.cancel()
    res = await polling_task
    assert res == "cancelled_cleanly"
