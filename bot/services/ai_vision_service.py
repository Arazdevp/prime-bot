"""Multimodal Vision (GIF/photo/sticker) and on-demand Telegram profile/bio inspection for AI chat."""

import base64
import io
import logging
import re
import time
from typing import Any
from aiogram import Bot
from aiogram.types import Message
from bot.database import repository

logger = logging.getLogger(__name__)

MAX_VISION_FILE_BYTES = 1_500_000
MAX_GEMINI_GIF_MP4_BYTES = 1_200_000
PROFILE_CACHE_TTL_SECONDS = 300.0

_ai_chat_vision_enabled_cache: bool = True
_profile_cache: dict[int, tuple[float, dict[str, Any]]] = {}

_PROFILE_TARGET_KEYWORDS: tuple[str, ...] = (
    "پروفایل",
    "پروف",
    "اواتار",
    "آواتار",
    "عکس پروف",
    "بیوگرافی",
    "بیو",
    "استایل",
    "تیپ",
    "profile",
    "pfp",
    "avatar",
    "bio",
    "style",
)

_POSSESSIVE_PROFILE_KEYWORDS: tuple[str, ...] = (
    "پروفایلم",
    "پروفایلش",
    "پروفایلت",
    "پروفم",
    "پروفش",
    "اواتارم",
    "اواتارش",
    "آواتارم",
    "آواتارش",
    "بیوم",
    "بیوش",
    "بیوگرافیم",
    "بیوگرافیش",
    "استایلم",
    "استایلش",
    "تیپم",
    "تیپش",
    "عکسم",
    "عکسش",
)

_INSPECTION_ACTION_KEYWORDS: tuple[str, ...] = (
    "ببین",
    "ببینی",
    "بخون",
    "بخونی",
    "نگاه",
    "نمره",
    "امتیاز",
    "چطوره",
    "چطور",
    "چند میدی",
    "چند از",
    "نظرت",
    "نظر",
    "تفسیر",
    "تحلیل",
    "بررسی",
    "قضاوت",
    "قشنگه",
    "خوبه",
    "بده",
    "بگو",
    "چیه",
    "چی نوشته",
    "rate",
    "check",
    "roast",
    "review",
    "score",
    "look",
)

_FIRST_PERSON_CUES: tuple[str, ...] = (
    "پروفایلم",
    "پروفم",
    "اواتارم",
    "آواتارم",
    "بیوم",
    "بیوگرافیم",
    "استایلم",
    "تیپم",
    "عکسم",
    "مال من",
    "خودم",
    "منو",
    "به من",
    "my ",
)

_THIRD_PERSON_CUES: tuple[str, ...] = (
    "پروفایلش",
    "پروفش",
    "اواتارش",
    "آواتارش",
    "بیوش",
    "بیوگرافیش",
    "استایلش",
    "تیپش",
    "عکسش",
    "ایشون",
    "طرف",
    "اون",
    "اینو",
    "his ",
    "her ",
    "their ",
)


def reset_ai_vision_caches() -> None:
    """Reset in-memory AI Vision toggle and profile cache."""
    global _ai_chat_vision_enabled_cache
    _ai_chat_vision_enabled_cache = True
    _profile_cache.clear()


async def init_ai_vision_state() -> bool:
    """Load persisted AI chat Vision toggle from SQLite."""
    global _ai_chat_vision_enabled_cache
    _ai_chat_vision_enabled_cache = await repository.get_ai_chat_vision_enabled()
    return _ai_chat_vision_enabled_cache


def is_ai_chat_vision_enabled_cached() -> bool:
    """Fast O(1) read of the cached AI chat Vision & profile inspection toggle."""
    return _ai_chat_vision_enabled_cache


async def set_ai_chat_vision_enabled_state(enabled: bool) -> bool:
    """Persist AI chat Vision & profile inspection toggle and update cache."""
    global _ai_chat_vision_enabled_cache
    await repository.set_ai_chat_vision_enabled(bool(enabled))
    _ai_chat_vision_enabled_cache = bool(enabled)
    return _ai_chat_vision_enabled_cache


def _valid_str(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return None


def _valid_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _valid_photo_list(obj: Any) -> list[Any]:
    if not isinstance(obj, (list, tuple)):
        return []
    return [p for p in obj if _valid_str(getattr(p, "file_id", None)) is not None]


def _pick_photo_file_id(photos: list[Any], *, preferred_max_width: int = 1024, max_bytes: int = MAX_VISION_FILE_BYTES) -> str | None:
    """Pick the sharpest photo size within width and byte budgets."""
    if not photos:
        return None
    sized: list[tuple[int, str]] = []
    for p in photos:
        fid = _valid_str(getattr(p, "file_id", None))
        if not fid:
            continue
        fsize = _valid_int(getattr(p, "file_size", None))
        if fsize is not None and fsize > max_bytes:
            continue
        width = _valid_int(getattr(p, "width", None)) or 0
        sized.append((width, fid))
    if not sized:
        return None
    within = [item for item in sized if item[0] <= preferred_max_width]
    if within:
        within.sort(key=lambda x: x[0], reverse=True)
        return within[0][1]
    sized.sort(key=lambda x: x[0])
    return sized[0][1]


def _detect_mime_from_bytes(raw: bytes, fallback: str = "image/jpeg") -> str:
    """Detect image/video MIME type from magic header bytes."""
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if len(raw) >= 4 and raw[:4] == b"GIF8":
        return "image/gif"
    if len(raw) >= 8 and raw[4:8] == b"ftyp":
        return "video/mp4"
    return fallback


async def download_telegram_file_b64(
    bot: Bot,
    file_id: str,
    *,
    fallback_mime: str = "image/jpeg",
    max_bytes: int = MAX_VISION_FILE_BYTES,
) -> dict[str, str] | None:
    """Download a Telegram file by file_id into memory and encode as base64 attachment dict."""
    clean_fid = _valid_str(file_id)
    if not clean_fid:
        return None
    try:
        buf = io.BytesIO()
        await bot.download(clean_fid, destination=buf)
        raw = buf.getvalue()
        if not raw:
            tg_file = await bot.get_file(clean_fid)
            file_path = _valid_str(getattr(tg_file, "file_path", None))
            if not file_path:
                return None
            buf2 = io.BytesIO()
            await bot.download_file(file_path, destination=buf2)
            raw = buf2.getvalue()
        if not raw or len(raw) > max_bytes:
            return None
        mime = _detect_mime_from_bytes(raw, fallback=fallback_mime)
        return {
            "mime_type": mime,
            "data_b64": base64.b64encode(raw).decode("ascii"),
        }
    except Exception as exc:
        logger.debug("Failed downloading Telegram media '%s': %s", clean_fid, exc)
        return None


def has_supported_visual_input(message: Message) -> bool:
    """Return True if message contains a real photo, animation, sticker, video, or video_note."""
    if _valid_photo_list(getattr(message, "photo", None)):
        return True
    for attr in ("animation", "sticker", "video", "video_note"):
        media_obj = getattr(message, attr, None)
        if media_obj is not None and _valid_str(getattr(media_obj, "file_id", None)) is not None:
            return True
    return False


def has_visual_photo(message: Message) -> bool:
    """Return True if message contains at least one real Telegram PhotoSize."""
    return bool(_valid_photo_list(getattr(message, "photo", None)))


async def _extract_from_single_message(
    bot: Bot,
    msg: Message,
    *,
    provider: str,
    is_reply: bool,
) -> tuple[dict[str, str] | None, str | None]:
    """Extract a single visual attachment and descriptive Persian context note from msg."""
    prefix = "در پاسخ به پیامی که شامل " if is_reply else "کاربر یک "
    suffix = " بوده است" if is_reply else " فرستاده است"

    # 1. Photo
    photos = _valid_photo_list(getattr(msg, "photo", None))
    if photos:
        fid = _pick_photo_file_id(photos)
        if fid:
            att = await download_telegram_file_b64(bot, fid, fallback_mime="image/jpeg")
            if att:
                return att, f"{prefix}عکس{suffix}؛ تصویر پیوست شده تا دقیقاً متناسب با محتوای داخل عکس جواب بدهی."

    # 2. GIF / Animation
    anim = getattr(msg, "animation", None)
    if anim is not None and _valid_str(getattr(anim, "file_id", None)):
        anim_fid = _valid_str(getattr(anim, "file_id", None))
        anim_size = _valid_int(getattr(anim, "file_size", None))
        anim_name = _valid_str(getattr(anim, "file_name", None))
        thumb = getattr(anim, "thumbnail", None) or getattr(anim, "thumb", None)
        thumb_fid = _valid_str(getattr(thumb, "file_id", None)) if thumb is not None else None

        att: dict[str, str] | None = None
        if provider == "gemini" and anim_fid and (anim_size is None or anim_size <= MAX_GEMINI_GIF_MP4_BYTES):
            att = await download_telegram_file_b64(
                bot,
                anim_fid,
                fallback_mime="video/mp4",
                max_bytes=MAX_GEMINI_GIF_MP4_BYTES,
            )
        if att is None and thumb_fid:
            att = await download_telegram_file_b64(bot, thumb_fid, fallback_mime="image/jpeg")
        if att is not None:
            name_hint = f" (نام فایل: {anim_name[:60]})" if anim_name else ""
            return att, (
                f"{prefix}گیف (GIF){name_hint}{suffix}؛ تصویر/ویدیو آن پیوست شده تا دقیقاً متناسب با صحنه، متن و حس‌وحال گیف جواب بدهی."
            )

    # 3. Sticker
    stk = getattr(msg, "sticker", None)
    if stk is not None and _valid_str(getattr(stk, "file_id", None)):
        stk_fid = _valid_str(getattr(stk, "file_id", None))
        is_animated = bool(getattr(stk, "is_animated", False)) is True
        is_video = bool(getattr(stk, "is_video", False)) is True
        stk_emoji = _valid_str(getattr(stk, "emoji", None))
        thumb = getattr(stk, "thumbnail", None) or getattr(stk, "thumb", None)
        thumb_fid = _valid_str(getattr(thumb, "file_id", None)) if thumb is not None else None

        att = None
        if not is_animated and not is_video and stk_fid:
            att = await download_telegram_file_b64(bot, stk_fid, fallback_mime="image/webp")
        if att is None and thumb_fid:
            att = await download_telegram_file_b64(bot, thumb_fid, fallback_mime="image/webp")
        if att is not None:
            emoji_hint = f" (ایموجی استیکر: {stk_emoji})" if stk_emoji else ""
            return att, f"{prefix}استیکر{emoji_hint}{suffix}؛ تصویر استیکر پیوست شده تا متناسب با حس‌وحالش جواب بدهی."

    # 4. Video / Video Note (thumbnail inspection)
    for attr, label in (("video_note", "ویدیومسیج"), ("video", "ویدیو")):
        vobj = getattr(msg, attr, None)
        if vobj is not None and _valid_str(getattr(vobj, "file_id", None)):
            thumb = getattr(vobj, "thumbnail", None) or getattr(vobj, "thumb", None)
            thumb_fid = _valid_str(getattr(thumb, "file_id", None)) if thumb is not None else None
            if thumb_fid:
                att = await download_telegram_file_b64(bot, thumb_fid, fallback_mime="image/jpeg")
                if att is not None:
                    return att, f"{prefix}{label}{suffix}؛ فریم پیش‌نمایش آن پیوست شده است."

    return None, None


async def extract_visual_attachment_from_message(
    bot: Bot,
    message: Message,
    *,
    provider: str = "9router",
    include_reply: bool = True,
) -> tuple[dict[str, str] | None, str | None]:
    """Extract visual media from message (or its replied message) for Vision-enabled LLMs."""
    att, note = await _extract_from_single_message(bot, message, provider=provider, is_reply=False)
    if att is not None:
        return att, note

    if include_reply:
        reply = getattr(message, "reply_to_message", None)
        if reply is not None and not getattr(reply, "forum_topic_created", None):
            att, note = await _extract_from_single_message(bot, reply, provider=provider, is_reply=True)
            if att is not None:
                return att, note

    return None, None


def _normalize_profile_query(text: str) -> str:
    cleaned = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("آ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ة", "ه")
        .replace("ۀ", "ه")
        .replace("\u200c", " ")
        .lower()
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_profile_inspection_intent(
    text: str | None,
    *,
    has_replied_human: bool = False,
) -> tuple[bool, bool]:
    """Detect whether the user wants the bot to inspect a profile/bio/avatar/style.

    Returns (should_inspect, inspect_replied_user).
    """
    if not text or not text.strip():
        return False, False
    norm = _normalize_profile_query(text)
    if not norm:
        return False, False

    has_target_kw = any(kw in norm for kw in _PROFILE_TARGET_KEYWORDS)
    if not has_target_kw:
        return False, False

    has_possessive = any(kw in norm for kw in _POSSESSIVE_PROFILE_KEYWORDS)
    has_action = any(kw in norm for kw in _INSPECTION_ACTION_KEYWORDS)
    if not (has_possessive or has_action):
        return False, False

    if has_replied_human:
        has_third = any(cue in norm for cue in _THIRD_PERSON_CUES)
        has_first = any(cue in norm for cue in _FIRST_PERSON_CUES)
        if has_third or not has_first:
            return True, True

    return True, False


async def fetch_user_profile_context(
    bot: Bot,
    target_user: Any,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Fetch a Telegram user's bio and latest profile photo (cached 5 minutes)."""
    user_id = _valid_int(getattr(target_user, "id", None))
    fallback_name = (
        _valid_str(getattr(target_user, "full_name", None))
        or _valid_str(getattr(target_user, "first_name", None))
        or "کاربر"
    )
    fallback_username = _valid_str(getattr(target_user, "username", None))

    if user_id is None or user_id <= 0:
        return {
            "context_text": f"- نام: {fallback_name}\n- بیوگرافی و عکس پروفایل در دسترس نبود.",
            "attachment": None,
            "has_bio": False,
            "has_photo": False,
        }

    current = now if now is not None else time.monotonic()
    cached = _profile_cache.get(user_id)
    if cached is not None and (current - cached[0]) < PROFILE_CACHE_TTL_SECONDS:
        return dict(cached[1])

    bio: str | None = None
    username = fallback_username
    display_name = fallback_name

    try:
        chat_info = await bot.get_chat(user_id)
        if chat_info is not None:
            raw_bio = _valid_str(getattr(chat_info, "bio", None))
            if raw_bio:
                bio = raw_bio[:500]
            uname = _valid_str(getattr(chat_info, "username", None))
            if uname:
                username = uname.lstrip("@")
            fn = _valid_str(getattr(chat_info, "first_name", None))
            ln = _valid_str(getattr(chat_info, "last_name", None))
            combined = " ".join(part for part in (fn, ln) if part).strip()
            if combined:
                display_name = combined
    except Exception as exc:
        logger.debug("Could not fetch get_chat(%s) for profile bio: %s", user_id, exc)

    photo_att: dict[str, str] | None = None
    try:
        photos_resp = await bot.get_user_profile_photos(user_id=user_id, limit=1)
        photos_matrix = getattr(photos_resp, "photos", None) if photos_resp is not None else None
        if isinstance(photos_matrix, (list, tuple)) and photos_matrix:
            first_album = photos_matrix[0]
            valid_sizes = _valid_photo_list(first_album)
            fid = _pick_photo_file_id(valid_sizes, preferred_max_width=640)
            if fid:
                photo_att = await download_telegram_file_b64(bot, fid, fallback_mime="image/jpeg")
    except Exception as exc:
        logger.debug("Could not fetch profile photo for user %s: %s", user_id, exc)

    lines = [f"- نام در تلگرام: {display_name}"]
    if username:
        lines.append(f"- آیدی: @{username}")
    if bio:
        lines.append(f"- بیوگرافی (Bio): «{bio}»")
    else:
        lines.append("- بیوگرافی (Bio): خالی یا به‌خاطر تنظیمات حریم خصوصی کاربر برای ربات قابل مشاهده نیست.")

    if photo_att is not None:
        lines.append("- عکس پروفایل: تصویر پروفایل کاربر پیوست شده است (به استایل، وایب و جزئیات واقعی داخل عکس دقت کن و نظر/نمره واقعی بده).")
    else:
        lines.append("- عکس پروفایل: عکس پروفایلی ندارد یا به دلیل حریم خصوصی تلگرام برای ربات پنهان است.")

    result = {
        "context_text": "\n".join(lines),
        "attachment": photo_att,
        "has_bio": bool(bio),
        "has_photo": photo_att is not None,
    }
    _profile_cache[user_id] = (current, dict(result))
    return result
