"""Role resolution and per-group permission checks sharing the AdminFilter TTL cache."""

from enum import Enum
import logging
import time
from typing import Any
from aiogram import Bot
from bot.config import get_settings
from bot.database import repository
from bot.filters import (
    ADMIN_CACHE_TTL_SECONDS,
    get_cached_admin_status,
    set_cached_admin_status,
)

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """Access role of a user interacting with the bot in private chat."""

    GLOBAL_ADMIN = "global_admin"
    GROUP_ADMIN = "group_admin"
    STRANGER = "stranger"


def is_global_admin(user_id: int) -> bool:
    """Return True if user_id is listed in configured global ADMIN_IDS."""
    return user_id in get_settings().ADMIN_IDS


async def is_user_group_admin(
    bot: Bot,
    chat_id: int,
    user_id: int,
    ttl: float = ADMIN_CACHE_TTL_SECONDS,
) -> bool:
    """Check whether user_id is a creator or administrator in chat_id using the shared TTL cache."""
    if is_global_admin(user_id):
        return True

    cached = get_cached_admin_status(chat_id, user_id)
    if cached is not None:
        return cached

    try:
        member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
        is_admin = member.status in ("creator", "administrator")
        set_cached_admin_status(chat_id, user_id, is_admin, ttl=ttl)
        return is_admin
    except Exception as exc:
        logger.debug(
            "Failed checking admin status for user %s in chat %s: %s",
            user_id,
            chat_id,
            exc,
        )
        return False


def _filter_allowed_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict groups to ALLOWED_CHAT_IDS when configured."""
    allowed = get_settings().ALLOWED_CHAT_IDS
    if not allowed:
        return groups
    allowed_set = set(allowed)
    return [g for g in groups if g["chat_id"] in allowed_set]


async def get_manageable_groups(bot: Bot, user_id: int) -> list[dict[str, Any]]:
    """Return active known groups that user_id is authorized to manage."""
    active_groups = _filter_allowed_groups(await repository.list_active_known_groups())
    if not active_groups:
        return []

    if is_global_admin(user_id):
        return active_groups

    manageable: list[dict[str, Any]] = []
    for group in active_groups:
        if await is_user_group_admin(bot, group["chat_id"], user_id):
            manageable.append(group)
    return manageable


async def resolve_user_role(bot: Bot, user_id: int) -> tuple[UserRole, list[dict[str, Any]]]:
    """Determine private-chat UserRole and return the list of manageable groups."""
    if is_global_admin(user_id):
        groups = await get_manageable_groups(bot, user_id)
        return UserRole.GLOBAL_ADMIN, groups

    groups = await get_manageable_groups(bot, user_id)
    if groups:
        return UserRole.GROUP_ADMIN, groups

    return UserRole.STRANGER, []


async def can_manage_group(bot: Bot, user_id: int, chat_id: int) -> bool:
    """Re-verify whether user_id is permitted to manage a specific active group."""
    settings = get_settings()
    if settings.ALLOWED_CHAT_IDS and chat_id not in settings.ALLOWED_CHAT_IDS:
        return False

    group = await repository.get_known_group(chat_id)
    if group is None or group.get("left_at") is not None:
        return False

    if is_global_admin(user_id):
        return True

    return await is_user_group_admin(bot, chat_id, user_id)


# Separate 5-minute TTL cache for general group membership checks (used for DM timer features)
_member_cache: dict[tuple[int, int], tuple[bool, float]] = {}
MEMBER_CACHE_TTL_SECONDS: float = 300.0


def clear_member_cache() -> None:
    """Clear the group-membership TTL cache (useful for test isolation)."""
    _member_cache.clear()


async def is_active_group_member(
    bot: Bot,
    user_id: int,
    ttl: float = MEMBER_CACHE_TTL_SECONDS,
) -> bool:
    """Return True if user_id is a global admin, owns an active timer, or is a member of an active allowed group."""
    if is_global_admin(user_id):
        return True

    try:
        if await repository.count_active_timers_for_user(user_id) > 0:
            return True
    except Exception:
        pass

    active_groups = _filter_allowed_groups(await repository.list_active_known_groups())
    if not active_groups:
        return False

    now = time.monotonic()
    for group in active_groups:
        chat_id = int(group["chat_id"])
        cached = _member_cache.get((chat_id, user_id))
        if cached is not None and now < cached[1]:
            if cached[0]:
                return True
            continue

        try:
            member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
            is_member = member.status in ("creator", "administrator", "member", "restricted")
            _member_cache[(chat_id, user_id)] = (is_member, now + ttl)
            if len(_member_cache) > 5000:
                expired = [k for k, (_, exp) in _member_cache.items() if now >= exp]
                for k in expired:
                    _member_cache.pop(k, None)
            if is_member:
                return True
        except Exception as exc:
            logger.debug(
                "Failed checking membership for user %s in chat %s: %s",
                user_id,
                chat_id,
                exc,
            )
            continue

    return False
