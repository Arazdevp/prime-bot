"""Custom aiogram filters for admin checks and forum supergroups."""

from aiogram import Bot
from aiogram.filters import Filter
from aiogram.types import Message
from bot.config import get_settings


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

        # 3. Check group administrator / creator status in the current chat
        try:
            member = await bot.get_chat_member(chat_id=message.chat.id, user_id=user_id)
            if member.status in ("creator", "administrator"):
                return True
        except Exception:
            # If unable to fetch chat member (e.g. network/permissions), default to False
            return False

        return False
