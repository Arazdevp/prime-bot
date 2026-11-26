"""Music Downloader service: topic detection, media extraction, indexing, and DM delivery."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message
from bot.database import repository
from bot.utils.general_topic import resolve_thread_id
from bot.utils.html_utils import escape_html

logger = logging.getLogger(__name__)

MUSIC_TOPIC_KEYWORDS: tuple[str, ...] = (
    "muzik",
    "musik",
    "music",
    "موزیک",
    "موظیک",
    "آهنگ",
    "اهنگ",
)

AUDIO_FILE_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".wav",
    ".aac",
    ".opus",
    ".wma",
    ".alac",
)


def is_music_topic_name(name: str | None) -> bool:
    """Return True if a forum topic name matches common music topic keywords."""
    if not name:
        return False
    lowered = name.strip().lower()
    if not lowered:
        return False
    return any(kw in lowered for kw in MUSIC_TOPIC_KEYWORDS)


def _clean_opt_str(value: Any) -> str | None:
    """Normalize optional string metadata."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def extract_music_media(message: Message) -> dict[str, Any] | None:
    """Extract normalized audio/music metadata from a Telegram message, or None if not music."""
    caption = _clean_opt_str(getattr(message, "caption", None))

    audio = getattr(message, "audio", None)
    if audio is not None and getattr(audio, "file_id", None):
        return {
            "file_id": audio.file_id,
            "file_unique_id": _clean_opt_str(getattr(audio, "file_unique_id", None)),
            "media_type": "audio",
            "title": _clean_opt_str(getattr(audio, "title", None)),
            "performer": _clean_opt_str(getattr(audio, "performer", None)),
            "file_name": _clean_opt_str(getattr(audio, "file_name", None)),
            "caption": caption,
            "duration": getattr(audio, "duration", None),
            "file_size": getattr(audio, "file_size", None),
            "mime_type": _clean_opt_str(getattr(audio, "mime_type", None)),
        }

    document = getattr(message, "document", None)
    if document is not None and getattr(document, "file_id", None):
        mime = (_clean_opt_str(getattr(document, "mime_type", None)) or "").lower()
        file_name = _clean_opt_str(getattr(document, "file_name", None))
        ext = Path(file_name).suffix.lower() if file_name else ""
        if mime.startswith("audio/") or ext in AUDIO_FILE_EXTENSIONS:
            return {
                "file_id": document.file_id,
                "file_unique_id": _clean_opt_str(getattr(document, "file_unique_id", None)),
                "media_type": "document",
                "title": None,
                "performer": None,
                "file_name": file_name,
                "caption": caption,
                "duration": None,
                "file_size": getattr(document, "file_size", None),
                "mime_type": mime or None,
            }

    voice = getattr(message, "voice", None)
    if voice is not None and getattr(voice, "file_id", None):
        return {
            "file_id": voice.file_id,
            "file_unique_id": _clean_opt_str(getattr(voice, "file_unique_id", None)),
            "media_type": "voice",
            "title": None,
            "performer": None,
            "file_name": None,
            "caption": caption,
            "duration": getattr(voice, "duration", None),
            "file_size": getattr(voice, "file_size", None),
            "mime_type": _clean_opt_str(getattr(voice, "mime_type", None)),
        }

    video = getattr(message, "video", None)
    if video is not None and getattr(video, "file_id", None):
        return {
            "file_id": video.file_id,
            "file_unique_id": _clean_opt_str(getattr(video, "file_unique_id", None)),
            "media_type": "video",
            "title": None,
            "performer": None,
            "file_name": _clean_opt_str(getattr(video, "file_name", None)),
            "caption": caption,
            "duration": getattr(video, "duration", None),
            "file_size": getattr(video, "file_size", None),
            "mime_type": _clean_opt_str(getattr(video, "mime_type", None)),
        }

    return None


async def should_capture_music_in_topic(
    chat_id: int,
    thread_id: int,
    topic_name_hint: str | None = None,
) -> bool:
    """Return True if music indexing is enabled for chat_id and thread_id matches config or topic name."""
    cfg = await repository.get_music_settings(chat_id)
    if cfg is not None:
        if int(cfg.get("enabled", 1)) == 0:
            return False
        configured_tid = cfg.get("thread_id")
        if configured_tid is not None:
            return int(configured_tid) == thread_id

    if is_music_topic_name(topic_name_hint):
        return True

    stored_name = await repository.get_topic_name(chat_id, thread_id)
    return is_music_topic_name(stored_name)


def _extract_sender_info(message: Message) -> tuple[int | None, str]:
    """Extract sender user_id and display name consistently with listener.py."""
    sender_chat = getattr(message, "sender_chat", None)
    from_user = getattr(message, "from_user", None)
    chat = getattr(message, "chat", None)

    if sender_chat is not None:
        if chat is not None and sender_chat.id == chat.id:
            return sender_chat.id, "ادمین ناشناس"
        return sender_chat.id, (getattr(sender_chat, "title", None) or "کانال/گروه")

    if from_user is not None:
        full_name = getattr(from_user, "full_name", None) or getattr(from_user, "first_name", None) or "کاربر"
        return from_user.id, full_name

    return None, "کاربر"


async def maybe_capture_music_message(message: Message) -> bool:
    """Index a qualifying music/audio message from a music topic into SQLite.

    Fast-paths out with zero DB hits for non-music messages.
    """
    chat = getattr(message, "chat", None)
    if chat is None or not getattr(message, "message_id", None):
        return False

    from_user = getattr(message, "from_user", None)
    sender_chat = getattr(message, "sender_chat", None)
    if from_user is not None and getattr(from_user, "is_bot", False) and sender_chat is None:
        return False

    media = extract_music_media(message)
    if media is None:
        return False

    chat_id = int(chat.id)
    thread_id = resolve_thread_id(message)
    topic_name_hint: str | None = None

    reply = getattr(message, "reply_to_message", None)
    if reply is not None:
        ftc = getattr(reply, "forum_topic_created", None)
        if ftc is not None:
            topic_name_hint = _clean_opt_str(getattr(ftc, "name", None))
            if thread_id == 1 and getattr(reply, "message_id", None):
                thread_id = int(reply.message_id)

    if not await should_capture_music_in_topic(chat_id, thread_id, topic_name_hint=topic_name_hint):
        return False

    sender_user_id, sender_name = _extract_sender_info(message)
    msg_date = getattr(message, "date", None)
    if msg_date is not None:
        created_at = (
            msg_date.astimezone(timezone.utc).isoformat()
            if msg_date.tzinfo
            else msg_date.replace(tzinfo=timezone.utc).isoformat()
        )
    else:
        created_at = datetime.now(timezone.utc).isoformat()

    inserted_id = await repository.insert_music_track(
        chat_id=chat_id,
        thread_id=thread_id,
        message_id=int(message.message_id),
        file_id=media["file_id"],
        file_unique_id=media["file_unique_id"],
        media_type=media["media_type"],
        title=media["title"],
        performer=media["performer"],
        file_name=media["file_name"],
        caption=media["caption"],
        duration=media["duration"],
        file_size=media["file_size"],
        mime_type=media["mime_type"],
        sender_user_id=sender_user_id,
        sender_name=sender_name,
        created_at=created_at,
    )
    if inserted_id is not None:
        logger.debug(
            "Indexed music track #%s (msg=%s, type=%s) in chat %s thread %s",
            inserted_id,
            message.message_id,
            media["media_type"],
            chat_id,
            thread_id,
        )
        return True
    return False


def format_track_label(track: dict[str, Any], max_len: int = 34) -> str:
    """Format a concise human-readable track label for inline buttons and lists."""
    performer = _clean_opt_str(track.get("performer"))
    title = _clean_opt_str(track.get("title"))
    file_name = _clean_opt_str(track.get("file_name"))

    if performer and title:
        raw = f"{performer} - {title}"
    elif title:
        raw = title
    elif file_name:
        raw = Path(file_name).stem or file_name
    else:
        mtype = track.get("media_type") or "audio"
        prefix = {"voice": "ویس", "video": "ویدیو"}.get(mtype, "موزیک")
        raw = f"{prefix} #{track.get('id', '?')}"

    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1] + "…"


def build_track_caption(track: dict[str, Any]) -> str:
    """Build an HTML-escaped Persian caption for a delivered music file."""
    label = escape_html(format_track_label(track, max_len=120))
    sender = escape_html(_clean_opt_str(track.get("sender_name")) or "عضو گروه")
    lines = [
        f"🎵 <b>{label}</b>",
        f"👤 ارسال‌شده توسط: {sender}",
    ]
    raw_caption = _clean_opt_str(track.get("caption"))
    if raw_caption:
        snippet = raw_caption[:400] + ("…" if len(raw_caption) > 400 else "")
        lines.append(f"📝 {escape_html(snippet)}")
    lines.append("💾 <i>فایل بدون محدودیت آماده ذخیره شدن است.</i>")

    full_caption = "\n".join(lines)
    if len(full_caption) > 1000:
        return full_caption[:999] + "…"
    return full_caption


async def _dispatch_media(
    bot: Bot,
    user_id: int,
    track: dict[str, Any],
    caption: str | None,
    parse_mode: str | None,
) -> None:
    """Send the stored Telegram file_id to user_id without content protection."""
    file_id = str(track["file_id"])
    media_type = str(track.get("media_type") or "audio")

    if media_type == "audio":
        await bot.send_audio(
            chat_id=user_id,
            audio=file_id,
            caption=caption,
            parse_mode=parse_mode,
            protect_content=False,
        )
    elif media_type == "voice":
        await bot.send_voice(
            chat_id=user_id,
            voice=file_id,
            caption=caption,
            parse_mode=parse_mode,
            protect_content=False,
        )
    elif media_type == "video":
        await bot.send_video(
            chat_id=user_id,
            video=file_id,
            caption=caption,
            parse_mode=parse_mode,
            protect_content=False,
        )
    else:
        await bot.send_document(
            chat_id=user_id,
            document=file_id,
            caption=caption,
            parse_mode=parse_mode,
            protect_content=False,
        )


async def send_music_track_to_user(
    bot: Bot,
    user_id: int,
    track: dict[str, Any],
) -> tuple[bool, str | None]:
    """Send an indexed music track to user_id in private chat and increment download_count."""
    caption = build_track_caption(track)
    try:
        await _dispatch_media(bot, user_id, track, caption, "HTML")
    except TelegramBadRequest as exc:
        err_lower = str(exc).lower()
        if "can't parse entities" in err_lower or "parsing" in err_lower:
            try:
                plain_label = format_track_label(track, max_len=120)
                await _dispatch_media(bot, user_id, track, f"🎵 {plain_label}", None)
            except Exception as retry_exc:
                logger.warning("Failed sending music track #%s to %s: %s", track.get("id"), user_id, retry_exc)
                return False, "❌ ارسال فایل موزیک به پیوی ناموفق بود. ممکن است فایل روی سرور تلگرام منقضی شده باشد."
        else:
            logger.warning("TelegramBadRequest sending music track #%s to %s: %s", track.get("id"), user_id, exc)
            return False, "❌ نتونستم این فایل موزیک رو بفرستم. ممکنه فایل حذف یا منقضی شده باشه."
    except Exception as exc:
        logger.warning("Unexpected error sending music track #%s to %s: %s", track.get("id"), user_id, exc)
        return False, "❌ خطایی در ارسال فایل موزیک پیش اومد. یه لحظه دیگه دوباره امتحان کن."

    track_id = track.get("id")
    if track_id is not None:
        try:
            await repository.increment_music_download_count(int(track_id))
        except Exception as exc:
            logger.debug("Failed incrementing download_count for track %s: %s", track_id, exc)

    return True, None
