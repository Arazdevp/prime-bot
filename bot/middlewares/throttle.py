"""Per-user private chat throttling middleware."""

import time
from typing import Any, Awaitable, Callable
from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject


class PrivateThrottleMiddleware(BaseMiddleware):
    """Rate-limit private chat updates to at most 1 update per rate_limit_seconds per user.

    Excess updates in private chats are silently dropped.
    Non-private updates (groups/supergroups) pass through untouched.
    """

    def __init__(self, rate_limit_seconds: float = 1.0) -> None:
        super().__init__()
        self.rate_limit_seconds = rate_limit_seconds
        self._last_seen: dict[int, float] = {}

    def clear(self) -> None:
        """Clear tracked user timestamps (useful for test isolation)."""
        self._last_seen.clear()

    def _prune_stale(self, now: float) -> None:
        """Prevent unbounded memory growth by removing old entries when map is large."""
        if len(self._last_seen) <= 5000:
            return
        cutoff = now - max(60.0, self.rate_limit_seconds * 10)
        stale_users = [uid for uid, ts in self._last_seen.items() if ts < cutoff]
        for uid in stale_users:
            self._last_seen.pop(uid, None)

    @staticmethod
    def _is_private_event(event: TelegramObject) -> bool:
        if isinstance(event, Message):
            return bool(event.chat and event.chat.type == "private")
        if isinstance(event, CallbackQuery):
            msg = event.message
            chat = getattr(msg, "chat", None)
            return bool(chat and chat.type == "private")
        chat = getattr(event, "chat", None)
        if chat is None:
            msg = getattr(event, "message", None)
            chat = getattr(msg, "chat", None)
        return bool(chat and getattr(chat, "type", None) == "private")

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not self._is_private_event(event):
            return await handler(event, data)

        from_user = getattr(event, "from_user", None)
        if from_user is None:
            return await handler(event, data)

        user_id = from_user.id
        now = time.monotonic()
        last = self._last_seen.get(user_id)
        if last is not None and (now - last) < self.rate_limit_seconds:
            return None

        self._last_seen[user_id] = now
        self._prune_stale(now)
        return await handler(event, data)
