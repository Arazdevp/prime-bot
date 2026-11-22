"""Custom aiogram filters for admin checks and forum supergroups."""

import time
from aiogram import Bot
from aiogram.filters import Filter
from aiogram.types import CallbackQuery, Message
from bot.config import get_settings

# In-memory TTL cache: (chat_id, user_id) -> (is_admin, expiry_timestamp)
_admin_cache: dict[tuple[int, int], tuple[bool, float]] = {}
ADMIN_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes


def clear_admin_cache() -> None:
    """Clear all entries from the admin lookup TTL cache (useful for test isolation)."""
    _admin_cache.clear()


def get_cached_admin_status(chat_id: int, user_id: int) -> bool | None:
    """Return cached admin status if present and unexpired; otherwise None."""
    entry = _admin_cache.get((chat_id, user_id))
    if entry is None:
        return None
    is_admin, expiry = entry
    if time.monotonic() < expiry:
        return is_admin
    _admin_cache.pop((chat_id, user_id), None)
    return None


def set_cached_admin_status(
    chat_id: int, user_id: int, is_admin: bool, ttl: float = ADMIN_CACHE_TTL_SECONDS
) -> None:
    """Store admin status in the TTL cache."""
    _admin_cache[(chat_id, user_id)] = (is_admin, time.monotonic() + ttl)


class ForumGroupFilter(Filter):
    """Filter ensuring the message originates from an allowed supergroup."""

    async def __call__(self, message: Message, bot: Bot) -> bool:
        if not message.chat:
            return False

        # Must be a supergroup
        if message.chat.type != "supergroup":
            return False

        settings = get_settings()
        if settings.ALLOWED_CHAT_IDS:
            if message.chat.id not in settings.ALLOWED_CHAT_IDS:
                return False

        return True


class AdminFilter(Filter):
    """Filter allowing bot admins, group creators, chat administrators, or anonymous admins."""

    def __init__(self, ttl: float = ADMIN_CACHE_TTL_SECONDS) -> None:
        self.ttl = ttl

    async def __call__(self, message: Message, bot: Bot) -> bool:
        if not message.chat:
            return False

        settings = get_settings()

        # 1. Check anonymous admin (Telegram sends message from sender_chat == chat)
        if message.sender_chat and message.sender_chat.id == message.chat.id:
            return True

        # If there is no user, cannot check user-based admin status
        if not message.from_user:
            return False

        user_id = message.from_user.id

        # 2. Check global admin IDs from config (.env)
        if user_id in settings.ADMIN_IDS:
            return True

        # 3. Check TTL cache for (chat_id, user_id)
        cached = get_cached_admin_status(message.chat.id, user_id)
        if cached is not None:
            return cached

        # 4. Check group administrator / creator status in the current chat
        try:
            member = await bot.get_chat_member(chat_id=message.chat.id, user_id=user_id)
            is_admin = member.status in ("creator", "administrator")
            set_cached_admin_status(message.chat.id, user_id, is_admin, ttl=self.ttl)
            return is_admin
        except Exception:
            # If unable to fetch chat member (e.g. network/permissions), default to False without caching
            return False


class PrivateChatFilter(Filter):
    """Filter allowing only updates originating from private chats."""

    async def __call__(self, event: Message | CallbackQuery) -> bool:
        chat = getattr(event, "chat", None)
        if chat is None:
            msg = getattr(event, "message", None)
            chat = getattr(msg, "chat", None)
        return bool(chat and chat.type == "private")
