"""Unit tests for AdminFilter TTL caching and router-level filter isolation."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiogram import Bot
from aiogram.types import Chat, ChatMemberAdministrator, ChatMemberMember, Message, User
from bot.filters import (
    AdminFilter,
    ForumGroupFilter,
    clear_admin_cache,
    get_cached_admin_status,
    set_cached_admin_status,
)
from bot.handlers.admin import router as admin_router


def _build_msg(chat_id: int, user_id: int | None, sender_chat_id: int | None = None) -> Message:
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"

    if user_id is not None:
        user = MagicMock(spec=User)
        user.id = user_id
        msg.from_user = user
    else:
        msg.from_user = None

    if sender_chat_id is not None:
        s_chat = MagicMock(spec=Chat)
        s_chat.id = sender_chat_id
        msg.sender_chat = s_chat
    else:
        msg.sender_chat = None

    return msg


def test_router_level_filter_isolation():
    """Verify router-level filter contains only ForumGroupFilter and NOT AdminFilter."""
    observer_filter_types = [
        type(f.callback if hasattr(f, "callback") else f)
        for f in admin_router.message._handler.filters
    ]
    assert ForumGroupFilter in observer_filter_types
    assert AdminFilter not in observer_filter_types

    # And verify all 13 command handlers on admin_router have AdminFilter
    assert len(admin_router.message.handlers) == 13
    for handler in admin_router.message.handlers:
        handler_filter_types = [
            type(f.callback if hasattr(f, "callback") else f)
            for f in handler.filters
        ]
        assert AdminFilter in handler_filter_types


@pytest.mark.asyncio
async def test_admin_filter_ttl_cache_hit():
    """Verify AdminFilter caches get_chat_member results for 5 minutes."""
    clear_admin_cache()
    chat_id = -100111
    user_id = 5555

    msg = _build_msg(chat_id, user_id)
    bot = AsyncMock(spec=Bot)
    admin_member = MagicMock(spec=ChatMemberAdministrator)
    admin_member.status = "administrator"
    bot.get_chat_member = AsyncMock(return_value=admin_member)

    filt = AdminFilter(ttl=300.0)

    with patch("bot.filters.get_settings") as mock_settings:
        mock_settings.return_value.ADMIN_IDS = []

        # First lookup: misses cache, calls bot.get_chat_member
        res1 = await filt(msg, bot)
        assert res1 is True
        assert bot.get_chat_member.call_count == 1

        # Second lookup: hits cache, does NOT call bot.get_chat_member
        res2 = await filt(msg, bot)
        assert res2 is True
        assert bot.get_chat_member.call_count == 1


@pytest.mark.asyncio
async def test_admin_filter_ttl_cache_expiry():
    """Verify cached admin status expires after TTL."""
    clear_admin_cache()
    chat_id = -100222
    user_id = 6666

    msg = _build_msg(chat_id, user_id)
    bot = AsyncMock(spec=Bot)
    admin_member = MagicMock(spec=ChatMemberAdministrator)
    admin_member.status = "administrator"
    bot.get_chat_member = AsyncMock(return_value=admin_member)

    filt = AdminFilter(ttl=300.0)

    with patch("bot.filters.get_settings") as mock_settings, patch("time.monotonic") as mock_time:
        mock_settings.return_value.ADMIN_IDS = []
        mock_time.return_value = 1000.0

        # First call at t=1000.0 (expires at 1300.0)
        await filt(msg, bot)
        assert bot.get_chat_member.call_count == 1

        # Call at t=1100.0 (still valid)
        mock_time.return_value = 1100.0
        await filt(msg, bot)
        assert bot.get_chat_member.call_count == 1

        # Call at t=1301.0 (expired)
        mock_time.return_value = 1301.0
        await filt(msg, bot)
        assert bot.get_chat_member.call_count == 2


@pytest.mark.asyncio
async def test_admin_filter_bypasses():
    """Anonymous admins and global ADMIN_IDS bypass cache and API calls."""
    clear_admin_cache()
    chat_id = -100333
    bot = AsyncMock(spec=Bot)
    bot.get_chat_member = AsyncMock()

    filt = AdminFilter()

    # 1. Anonymous admin
    anon_msg = _build_msg(chat_id, user_id=None, sender_chat_id=chat_id)
    assert await filt(anon_msg, bot) is True
    bot.get_chat_member.assert_not_called()

    # 2. Config ADMIN_IDS
    admin_user_msg = _build_msg(chat_id, user_id=999)
    with patch("bot.filters.get_settings") as mock_settings:
        mock_settings.return_value.ADMIN_IDS = [999]
        assert await filt(admin_user_msg, bot) is True
        bot.get_chat_member.assert_not_called()


@pytest.mark.asyncio
async def test_admin_filter_api_failure_does_not_cache():
    """Network or permission errors during get_chat_member should return False and not cache."""
    clear_admin_cache()
    chat_id = -100444
    user_id = 7777

    msg = _build_msg(chat_id, user_id)
    bot = AsyncMock(spec=Bot)
    bot.get_chat_member = AsyncMock(side_effect=Exception("Telegram connection timeout"))

    filt = AdminFilter()
    with patch("bot.filters.get_settings") as mock_settings:
        mock_settings.return_value.ADMIN_IDS = []
        res = await filt(msg, bot)
        assert res is False
        assert get_cached_admin_status(chat_id, user_id) is None
