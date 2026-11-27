"""Zero-token reactions, media bank management, and sticker/GIF/video_note replies for AI chat."""

from dataclasses import dataclass
import logging
import random
import re
import time
from typing import Any
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, ReactionTypeEmoji
from bot.config import get_settings
from bot.database import repository
from bot.database.db import is_db_initialized
from bot.services.ai_memory_service import is_allowed_ai_chat_topic
from bot.utils.general_topic import resolve_thread_id

logger = logging.getLogger(__name__)

# Strictly allowed reaction emojis per user requirement (thumbs up, thumbs down, grinning)
ALLOWED_AI_REACTIONS: tuple[str, ...] = ("👍", "👎", "😁")

_NEGATIVE_EMOJI_SET: frozenset[str] = frozenset(
    {
        "👎",
        "😡",
        "🤬",
        "😤",
        "😒",
        "🙄",
        "😢",
        "😭",
        "😞",
        "😔",
        "💔",
        "🤢",
        "🤮",
        "💩",
        "❌",
        "🚫",
        "😠",
        "😩",
        "😫",
        "😕",
        "😟",
        "☹",
        "🙁",
        "😣",
        "😖",
        "🤦",
    }
)

_LAUGH_EMOJI_SET: frozenset[str] = frozenset(
    {
        "😂",
        "🤣",
        "😁",
        "😆",
        "😄",
        "😃",
        "😀",
        "😅",
        "😜",
        "🤪",
        "😝",
        "😛",
        "😏",
        "😎",
        "🥳",
        "🤡",
        "💀",
        "☠",
        "😹",
        "😺",
        "😸",
        "🤭",
        "😋",
        "🤩",
        "😉",
    }
)

_HYPE_EMOJI_SET: frozenset[str] = frozenset(
    {
        "🔥",
        "❤️",
        "❤",
        "😍",
        "🥰",
        "🤩",
        "🥳",
        "🚀",
        "💥",
        "⚡",
        "💯",
        "🎉",
        "🎊",
        "🏆",
        "👑",
        "🫶",
        "💖",
        "💘",
        "💕",
        "✨",
        "🌟",
    }
)

_AGREE_EMOJI_SET: frozenset[str] = frozenset(
    {
        "👍",
        "👌",
        "🤝",
        "👏",
        "🙏",
        "✅",
        "☑️",
        "✔️",
        "💪",
        "🫡",
        "🤞",
    }
)

# Match common Unicode emoji codepoints (excluding plain digits/ASCII punctuation)
_EMOJI_CHAR_RE = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"
    r"\U00002600-\U000027BF"
    r"\U00002300-\U000023FF"
    r"\U0001F1E6-\U0001F1FF"
    r"\u2B50\u2B55\u2764\u2763\u2705\u274C\u2757\u2753\u203C\u2049\u3030\u303D\u3297\u3299\u00A9\u00AE"
    r"]"
)
_EMOJI_STRIP_MODIFIERS_RE = re.compile(r"[\uFE0E\uFE0F\u200D\u20E3\U0001F3FB-\U0001F3FF\U000E0020-\U000E007F]")
_LETTER_OR_DIGIT_RE = re.compile(r"[^\W_]", re.UNICODE)
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_PUNCT_AND_SYM_RE = re.compile(r"[^\w\s]", re.UNICODE)
_REPEAT_CHAR_RE = re.compile(r"(.)\1+")
_LAUGH_WORD_RE = re.compile(
    r"^(?:[هخحپ]{2,}|(?:ها|حه|هع|که|هی){2,}|(?:ha|he|hi){2,}|lo+l+|lma+o+|lmfa+o+|rofl+|xd+)$"
)

_GREETING_BLOCK_PHRASES: frozenset[str] = frozenset(
    {
        "سلام",
        "سلام علیکم",
        "درود",
        "خوبی",
        "چطوری",
        "چه خبر",
        "صبح بخیر",
        "عصر بخیر",
        "hi",
        "hello",
        "hey",
    }
)


def _normalize_phrase_text(text: str) -> str:
    """Normalize Persian/Arabic/Latin characters and strip punctuation/emojis for exact matching."""
    cleaned = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("آ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ة", "ه")
        .replace("ۀ", "ه")
        .replace("ؤ", "و")
        .replace("\u200c", " ")
    )
    cleaned = _DIACRITICS_RE.sub("", cleaned)
    cleaned = cleaned.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    cleaned = _EMOJI_CHAR_RE.sub(" ", cleaned)
    cleaned = _EMOJI_STRIP_MODIFIERS_RE.sub("", cleaned)
    cleaned = cleaned.lower()
    cleaned = _PUNCT_AND_SYM_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _collapse_repeats(text: str) -> str:
    """Collapse consecutive duplicate letters so stretched words ('مرسیییی') match cleanly."""
    return _REPEAT_CHAR_RE.sub(r"\1", text)


_RAW_NEGATIVE_PHRASES: tuple[str, ...] = (
    "نه",
    "نخیر",
    "اصلا",
    "ابدا",
    "عمرا",
    "به هیچ وجه",
    "غلطه",
    "اشتباهه",
    "اشتباه",
    "موافق نیستم",
    "مخالفم",
    "قبول ندارم",
    "بد بود",
    "بده",
    "افتضاح",
    "افتضاحه",
    "مزخرف",
    "مزخرفه",
    "چرت",
    "چرته",
    "بیخود",
    "بیخوده",
    "الکی",
    "الکیه",
    "دروغ",
    "دروغه",
    "ناراحتم",
    "حیف",
    "متاسفم",
    "اه",
    "اه اه",
    "ایش",
    "بدم اومد",
    "دوست نداشتم",
    "خوشم نیومد",
    "nope",
    "nah",
    "no",
    "bad",
    "trash",
    "disagree",
    "wrong",
    "meh",
)

_RAW_LAUGH_PHRASES: tuple[str, ...] = (
    "جالب بود",
    "باحال بود",
    "عجب",
    "پشمام",
    "جر",
    "جر خوردم",
    "دهنت",
    "دهنت سرویس",
    "سم",
    "سم بود",
    "اسید",
    "وای",
    "ای بابا",
    "خنده دار بود",
    "بامزه بود",
    "haha",
    "hehe",
    "lol",
    "lmao",
    "xd",
)

_RAW_POSITIVE_PHRASES: tuple[str, ...] = (
    "افرین",
    "باریکلا",
    "ایول",
    "ایولا",
    "دمت گرم",
    "دستت درد نکنه",
    "دست شما درد نکنه",
    "دست مریزاد",
    "خسته نباشی",
    "خسته نباشید",
    "ماشالله",
    "ماشالا",
    "بنازم",
    "عالی",
    "عالیه",
    "عالی بود",
    "فوق العاده",
    "بی نظیر",
    "بینظیر",
    "زنده باد",
    "کارت درسته",
    "موفق باشی",
    "موفق باشید",
    "خوب بود",
    "خوبه",
    "باحاله",
    "باحال",
    "قشنگ بود",
    "جالبه",
    "نایس",
    "گاد",
    "بمب",
    "خفن",
    "خفنه",
    "پرفکت",
    "دقیقا",
    "دقیق",
    "اره",
    "اری",
    "بله",
    "بلله",
    "قطعا",
    "حتما",
    "موافقم",
    "درسته",
    "درست میگی",
    "راست میگی",
    "حق",
    "حقه",
    "حرف حق",
    "کاملا",
    "صد در صد",
    "صددرصد",
    "100 درصد",
    "واقعا",
    "معلومه",
    "البته",
    "طبیعیه",
    "منطقیه",
    "تایید",
    "تاییده",
    "مرسی",
    "ممنون",
    "ممنونم",
    "ممنونتم",
    "متشکرم",
    "تشکر",
    "سپاس",
    "سپاسگزارم",
    "مچکر",
    "مچکرم",
    "قربونت",
    "قربانت",
    "فدات",
    "فداتشم",
    "نوکرم",
    "چاکرم",
    "ارادت",
    "عشقی",
    "عشق منی",
    "عزیزی",
    "لطف کردی",
    "لطفا",
    "خواهش",
    "خواهش میکنم",
    "قابلی نداره",
    "زنده باشی",
    "سلامت باشی",
    "مخلصم",
    "مخلص",
    "کوچیکتم",
    "اوکی",
    "اوکیه",
    "باشه",
    "باش",
    "حله",
    "حل",
    "چشم",
    "قبوله",
    "ردیفه",
    "گرفتم",
    "فهمیدم",
    "متوجه شدم",
    "خداحافظ",
    "بای",
    "فعلا",
    "شب بخیر",
    "شب خوش",
    "nice",
    "great",
    "cool",
    "awesome",
    "good",
    "perfect",
    "gg",
    "wp",
    "bravo",
    "yes",
    "yep",
    "yeah",
    "yup",
    "true",
    "exactly",
    "facts",
    "agreed",
    "agree",
    "indeed",
    "right",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "tysm",
    "merci",
    "mersi",
    "mamnoon",
    "mamnon",
    "ok",
    "okay",
    "k",
    "kk",
    "okey",
    "alright",
    "got it",
    "gotcha",
    "noted",
    "sure",
    "fine",
    "bye",
    "gn",
    "bashe",
    "halle",
)

_RAW_VOCATIVES: tuple[str, ...] = (
    "رفیق",
    "داش",
    "داداش",
    "عزیز",
    "عزیزم",
    "گل",
    "گلم",
    "استاد",
    "بزرگوار",
    "سلطان",
    "عشق",
    "پسر",
    "دختر",
    "حاجی",
    "مشتی",
    "جونم",
    "جان",
    "برادر",
    "خواهر",
    "bro",
    "dude",
    "man",
    "mate",
    "boss",
)

_NEGATIVE_PHRASES: frozenset[str] = frozenset(_normalize_phrase_text(p) for p in _RAW_NEGATIVE_PHRASES)
_LAUGH_PHRASES: frozenset[str] = frozenset(_normalize_phrase_text(p) for p in _RAW_LAUGH_PHRASES)
_POSITIVE_PHRASES: frozenset[str] = frozenset(_normalize_phrase_text(p) for p in _RAW_POSITIVE_PHRASES)
_VOCATIVE_WORDS: frozenset[str] = frozenset(_normalize_phrase_text(p) for p in _RAW_VOCATIVES)

_COLLAPSED_NEGATIVE: frozenset[str] = frozenset(_collapse_repeats(p) for p in _NEGATIVE_PHRASES)
_COLLAPSED_LAUGH: frozenset[str] = frozenset(_collapse_repeats(p) for p in _LAUGH_PHRASES)
_COLLAPSED_POSITIVE: frozenset[str] = frozenset(_collapse_repeats(p) for p in _POSITIVE_PHRASES)
_COLLAPSED_VOCATIVES: frozenset[str] = frozenset(_collapse_repeats(p) for p in _VOCATIVE_WORDS)

# Directives emitted by LLM inside replies
_REACT_DIRECTIVE_RE = re.compile(r"\[REACT\s*:\s*([^\]]+?)\]", re.IGNORECASE)
_MEDIA_DIRECTIVE_RE = re.compile(
    r"\[(STICKER|GIF|ANIMATION|VIDEONOTE|VIDEO_NOTE)(?:\s*:\s*([a-zA-Z_-]+))?\]",
    re.IGNORECASE,
)

# Explicit short media request patterns (zero-token fast-path)
_EXPLICIT_STICKER_RE = re.compile(
    r"(?:استیکر|استیکری|sticker)",
    re.IGNORECASE,
)
_EXPLICIT_GIF_RE = re.compile(
    r"(?:گیف|گیفی|انیمیشن|gif)",
    re.IGNORECASE,
)
_EXPLICIT_VIDEONOTE_RE = re.compile(
    r"(?:ویدیو\s*مسیج|ویدیو\s*نوت|فیلم\s*گرد|video\s*note|videonote|video\s*message)",
    re.IGNORECASE,
)
_EXPLICIT_REQUEST_VERB_RE = re.compile(
    r"(?:بده|بفرست|بزار|بذار|میفرستی|میدی|بفرستی|بزاری|بذاری|send|give|drop|share)",
    re.IGNORECASE,
)

# Runtime caches for media bank & anti-repeat/cooldowns
_ai_chat_media_autocollect_cache: bool = True
_ai_chat_media_counts_cache: dict[str, int] | None = None
_last_sent_media_id: dict[tuple[int, int], int] = {}
_media_only_reply_cooldown: dict[tuple[int, int], float] = {}
MEDIA_ONLY_REPLY_COOLDOWN_SECONDS = 15.0


@dataclass
class ParsedAiDirectives:
    """Parsed text and optional reaction/media directives from an LLM output."""

    text: str
    reaction: str | None = None
    media_type: str | None = None
    media_mood: str = "general"


def reset_ai_media_caches() -> None:
    """Reset in-memory AI media auto-collect, counts, and cooldown caches."""
    global _ai_chat_media_autocollect_cache, _ai_chat_media_counts_cache
    _ai_chat_media_autocollect_cache = True
    _ai_chat_media_counts_cache = None
    _last_sent_media_id.clear()
    _media_only_reply_cooldown.clear()


def invalidate_ai_media_counts_cache() -> None:
    """Invalidate cached media counts so next read queries SQLite."""
    global _ai_chat_media_counts_cache
    _ai_chat_media_counts_cache = None


async def init_ai_media_state() -> None:
    """Load persisted AI chat media auto-collect flag and counts from SQLite."""
    global _ai_chat_media_autocollect_cache, _ai_chat_media_counts_cache
    _ai_chat_media_autocollect_cache = await repository.get_ai_chat_media_autocollect()
    _ai_chat_media_counts_cache = await repository.count_ai_chat_media_by_type()


def is_ai_chat_media_autocollect_cached() -> bool:
    """Fast O(1) read of passive sticker/GIF auto-collect toggle."""
    return _ai_chat_media_autocollect_cache


async def set_ai_chat_media_autocollect_state(enabled: bool) -> bool:
    """Persist passive sticker/GIF auto-collect toggle and update cache."""
    global _ai_chat_media_autocollect_cache
    await repository.set_ai_chat_media_autocollect(bool(enabled))
    _ai_chat_media_autocollect_cache = bool(enabled)
    return _ai_chat_media_autocollect_cache


async def get_ai_chat_media_counts_cached() -> dict[str, int]:
    """Return cached media counts keyed by media_type, refreshing lazily when needed."""
    global _ai_chat_media_counts_cache
    if _ai_chat_media_counts_cache is not None:
        return dict(_ai_chat_media_counts_cache)
    if not is_db_initialized():
        return {mtype: 0 for mtype in repository.AI_CHAT_MEDIA_TYPES}
    try:
        counts = await repository.count_ai_chat_media_by_type()
        _ai_chat_media_counts_cache = counts
        return dict(counts)
    except Exception as exc:
        logger.debug("Could not fetch AI chat media counts: %s", exc)
        return {mtype: 0 for mtype in repository.AI_CHAT_MEDIA_TYPES}


def clamp_ai_reaction(emoji: str | None) -> str:
    """Clamp any emoji candidate strictly into ('👍', '👎', '😁')."""
    cleaned = str(emoji or "").strip()
    if cleaned in ALLOWED_AI_REACTIONS:
        return cleaned
    for ch in cleaned:
        if ch in _NEGATIVE_EMOJI_SET:
            return "👎"
    for ch in cleaned:
        if ch in _LAUGH_EMOJI_SET:
            return "😁"
    return "👍"


def mood_to_reaction(mood: str | None) -> str:
    """Map an AI media mood to one of the 3 allowed reaction emojis."""
    cleaned = (mood or "").strip().lower()
    if cleaned == "disagree":
        return "👎"
    if cleaned == "laugh":
        return "😁"
    return "👍"


def classify_emoji_sentiment(text: str) -> str:
    """Classify emoji string sentiment strictly into '👍', '👎', or '😁'."""
    if not text:
        return "👍"
    for ch in text:
        if ch in _NEGATIVE_EMOJI_SET:
            return "👎"
    for ch in text:
        if ch in _LAUGH_EMOJI_SET:
            return "😁"
    return "👍"


def is_emoji_only_text(text: str | None) -> bool:
    """Return True if text contains >= 1 emoji and no alphabetic letters, digits, or question marks."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped or "?" in stripped or "؟" in stripped:
        return False
    if not _EMOJI_CHAR_RE.search(stripped):
        return False
    without_emojis = _EMOJI_CHAR_RE.sub("", stripped)
    without_emojis = _EMOJI_STRIP_MODIFIERS_RE.sub("", without_emojis)
    without_emojis = _PUNCT_AND_SYM_RE.sub("", without_emojis)
    without_emojis = re.sub(r"\s+", "", without_emojis)
    if without_emojis:
        return False
    return not _LETTER_OR_DIGIT_RE.search(without_emojis)


def _match_phrase_category(phrase: str) -> str | None:
    """Return 'negative', 'laugh', 'positive', 'vocative', or None for a normalized phrase."""
    if _LAUGH_WORD_RE.match(phrase):
        return "laugh"
    if phrase in _NEGATIVE_PHRASES:
        return "negative"
    if phrase in _LAUGH_PHRASES:
        return "laugh"
    if phrase in _POSITIVE_PHRASES:
        return "positive"
    if phrase in _VOCATIVE_WORDS:
        return "vocative"

    collapsed = _collapse_repeats(phrase)
    if _LAUGH_WORD_RE.match(collapsed):
        return "laugh"
    if collapsed in _COLLAPSED_NEGATIVE:
        return "negative"
    if collapsed in _COLLAPSED_LAUGH:
        return "laugh"
    if collapsed in _COLLAPSED_POSITIVE:
        return "positive"
    if collapsed in _COLLAPSED_VOCATIVES:
        return "vocative"
    return None


def classify_zero_token_reaction(text: str | None) -> str | None:
    """Return '👍', '👎', or '😁' for emoji-only or zero-reply messages; otherwise None.

    Runs completely offline with zero LLM tokens. Questions and greetings always return None.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped or "?" in stripped or "؟" in stripped:
        return None

    if is_emoji_only_text(stripped):
        return classify_emoji_sentiment(stripped)

    emojis_found = _EMOJI_CHAR_RE.findall(stripped)
    emoji_hint: str | None = classify_emoji_sentiment("".join(emojis_found)) if emojis_found else None

    norm = _normalize_phrase_text(stripped)
    if not norm or norm in _GREETING_BLOCK_PHRASES:
        return None

    words = norm.split()
    if not words or len(words) > 4:
        return None

    direct_cat = _match_phrase_category(norm)
    if direct_cat in ("negative", "laugh", "positive"):
        if direct_cat == "negative" or emoji_hint == "👎":
            return "👎"
        if direct_cat == "laugh" or emoji_hint == "😁":
            return "😁"
        return "👍"
    if direct_cat == "vocative":
        return None

    # Greedy segmentation for short combinations like 'آفرین دقیقا' or 'مرسی داش'
    categories: list[str] = []
    idx = 0
    while idx < len(words):
        matched_cat: str | None = None
        matched_span = 0
        for span in (3, 2, 1):
            if idx + span > len(words):
                continue
            chunk = " ".join(words[idx : idx + span])
            if chunk in _GREETING_BLOCK_PHRASES:
                return None
            cat = _match_phrase_category(chunk)
            if cat is not None:
                matched_cat = cat
                matched_span = span
                break
        if matched_cat is None:
            return None
        categories.append(matched_cat)
        idx += matched_span

    non_vocatives = [c for c in categories if c != "vocative"]
    if not non_vocatives:
        return None

    if "negative" in non_vocatives or emoji_hint == "👎":
        return "👎"
    if "laugh" in non_vocatives or emoji_hint == "😁":
        return "😁"
    return "👍"


def detect_explicit_media_request(text: str | None) -> str | None:
    """Detect short direct user requests for a sticker, GIF, or video message (zero tokens)."""
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    words = stripped.split()
    if len(words) > 5:
        return None
    if not _EXPLICIT_REQUEST_VERB_RE.search(stripped):
        return None
    if _EXPLICIT_VIDEONOTE_RE.search(stripped):
        return "video_note"
    if _EXPLICIT_GIF_RE.search(stripped):
        return "animation"
    if _EXPLICIT_STICKER_RE.search(stripped):
        return "sticker"
    return None


def _canonical_media_type(tag_name: str) -> str | None:
    upper = (tag_name or "").strip().upper()
    if upper == "STICKER":
        return "sticker"
    if upper in ("GIF", "ANIMATION"):
        return "animation"
    if upper in ("VIDEONOTE", "VIDEO_NOTE"):
        return "video_note"
    return None


def canonical_media_tag(media_type: str, mood: str = "general") -> str:
    """Return canonical directive tag string for a media_type and mood."""
    safe_mood = mood if mood in repository.AI_CHAT_MEDIA_MOODS else "general"
    if media_type == "sticker":
        return f"[STICKER:{safe_mood}]"
    if media_type == "animation":
        return f"[GIF:{safe_mood}]"
    if media_type == "video_note":
        return f"[VIDEONOTE:{safe_mood}]"
    return ""


def parse_ai_chat_directives(raw_text: str | None) -> ParsedAiDirectives:
    """Extract and strip [REACT:...] and [STICKER/GIF/VIDEONOTE:...] tags from an LLM reply."""
    text = str(raw_text or "").strip()
    if not text:
        return ParsedAiDirectives(text="")

    reaction: str | None = None
    media_type: str | None = None
    media_mood = "general"

    def _collect_react(match: re.Match[str]) -> str:
        nonlocal reaction
        if reaction is None:
            reaction = clamp_ai_reaction(match.group(1))
        return ""

    def _collect_media(match: re.Match[str]) -> str:
        nonlocal media_type, media_mood
        if media_type is None:
            mtype = _canonical_media_type(match.group(1))
            if mtype is not None:
                media_type = mtype
                raw_mood = (match.group(2) or "general").strip().lower()
                media_mood = raw_mood if raw_mood in repository.AI_CHAT_MEDIA_MOODS else "general"
        return ""

    cleaned = _REACT_DIRECTIVE_RE.sub(_collect_react, text)
    cleaned = _MEDIA_DIRECTIVE_RE.sub(_collect_media, cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return ParsedAiDirectives(
        text=cleaned,
        reaction=reaction,
        media_type=media_type,
        media_mood=media_mood,
    )


def format_directive_history_placeholder(parsed: ParsedAiDirectives) -> str:
    """Format clean text saved to rolling conversation history without raw directive syntax."""
    if parsed.text:
        return parsed.text
    if parsed.media_type == "sticker":
        return "[ارسال استیکر]"
    if parsed.media_type == "animation":
        return "[ارسال گیف]"
    if parsed.media_type == "video_note":
        return "[ارسال ویدیو مسیج]"
    if parsed.reaction:
        return f"[واکنش {parsed.reaction}]"
    return ""


def infer_mood_from_emoji(emoji: str | None) -> str:
    """Infer an AI chat media mood ('laugh', 'disagree', 'hype', 'agree', or 'general') from emoji."""
    if not emoji or not isinstance(emoji, str):
        return "general"
    for ch in emoji:
        if ch in _LAUGH_EMOJI_SET:
            return "laugh"
    for ch in emoji:
        if ch in _NEGATIVE_EMOJI_SET:
            return "disagree"
    for ch in emoji:
        if ch in _HYPE_EMOJI_SET:
            return "hype"
    for ch in emoji:
        if ch in _AGREE_EMOJI_SET:
            return "agree"
    return "general"


def extract_ai_chat_media(message: Message) -> dict[str, Any] | None:
    """Safely extract sticker, animation (GIF), or video_note metadata from a Telegram Message.

    Explicitly validates that file_id and file_unique_id are non-empty strings so MagicMock(spec=Message)
    objects without configured media attributes return None.
    """
    sticker = getattr(message, "sticker", None)
    if sticker is not None:
        fid = getattr(sticker, "file_id", None)
        fuid = getattr(sticker, "file_unique_id", None)
        if isinstance(fid, str) and fid.strip() and isinstance(fuid, str) and fuid.strip():
            raw_emoji = getattr(sticker, "emoji", None)
            emoji = raw_emoji.strip() if isinstance(raw_emoji, str) and raw_emoji.strip() else None
            return {
                "media_type": "sticker",
                "file_id": fid.strip(),
                "file_unique_id": fuid.strip(),
                "emoji": emoji,
            }

    animation = getattr(message, "animation", None)
    if animation is not None:
        fid = getattr(animation, "file_id", None)
        fuid = getattr(animation, "file_unique_id", None)
        if isinstance(fid, str) and fid.strip() and isinstance(fuid, str) and fuid.strip():
            return {
                "media_type": "animation",
                "file_id": fid.strip(),
                "file_unique_id": fuid.strip(),
                "emoji": None,
            }

    video_note = getattr(message, "video_note", None)
    if video_note is not None:
        fid = getattr(video_note, "file_id", None)
        fuid = getattr(video_note, "file_unique_id", None)
        if isinstance(fid, str) and fid.strip() and isinstance(fuid, str) and fuid.strip():
            return {
                "media_type": "video_note",
                "file_id": fid.strip(),
                "file_unique_id": fuid.strip(),
                "emoji": None,
            }

    return None


async def maybe_capture_ai_chat_media(message: Message) -> bool:
    """Passively index stickers and GIFs from the free-chat topic into ai_chat_media."""
    if not is_ai_chat_media_autocollect_cached():
        return False

    chat = getattr(message, "chat", None)
    if chat is None or getattr(chat, "type", None) != "supergroup":
        return False

    from_user = getattr(message, "from_user", None)
    if from_user is not None and getattr(from_user, "is_bot", False) and not getattr(message, "sender_chat", None):
        return False

    media = extract_ai_chat_media(message)
    if media is None or media["media_type"] not in ("sticker", "animation"):
        return False

    settings = get_settings()
    if settings.ALLOWED_CHAT_IDS and chat.id not in settings.ALLOWED_CHAT_IDS:
        return False

    thread_id = resolve_thread_id(message)
    if not await is_allowed_ai_chat_topic(chat.id, thread_id):
        return False

    mood = infer_mood_from_emoji(media.get("emoji"))
    added_by = getattr(from_user, "id", None) if from_user is not None else None
    await repository.upsert_ai_chat_media(
        media_type=media["media_type"],
        file_id=media["file_id"],
        file_unique_id=media["file_unique_id"],
        mood=mood,
        emoji=media.get("emoji"),
        source="auto",
        added_by=added_by,
        chat_id=chat.id,
    )
    invalidate_ai_media_counts_cache()
    return True


def should_reply_with_media_to_media(
    chat_id: int,
    thread_id: int,
    *,
    now: float | None = None,
) -> bool:
    """Enforce a short cooldown so consecutive media messages don't spam back-to-back media replies."""
    current = now if now is not None else time.monotonic()
    key = (chat_id, thread_id)
    last_ts = _media_only_reply_cooldown.get(key)
    if last_ts is not None and (current - last_ts) < MEDIA_ONLY_REPLY_COOLDOWN_SECONDS:
        return False
    _media_only_reply_cooldown[key] = current
    return True


async def send_ai_reaction(
    bot: Bot,
    chat_id: int,
    message_id: int,
    emoji: str,
) -> bool:
    """Send a single Telegram message reaction strictly clamped to ('👍', '👎', '😁')."""
    safe_emoji = clamp_ai_reaction(emoji)
    try:
        await bot.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[ReactionTypeEmoji(emoji=safe_emoji)],
        )
        return True
    except Exception as exc:
        logger.debug("Failed setting AI reaction %s on chat %s msg %s: %s", safe_emoji, chat_id, message_id, exc)
        return False


async def _pick_media_candidates(media_type: str, mood: str) -> list[dict[str, Any]]:
    """Fetch ordered candidate media items matching mood, falling back to general or any."""
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    queries: list[str | None] = [mood]
    if mood != "general":
        queries.append("general")
    queries.append(None)

    for q_mood in queries:
        rows = await repository.list_ai_chat_media_candidates(media_type, mood=q_mood, limit=8)
        for row in rows:
            rid = int(row["id"])
            if rid not in seen_ids:
                seen_ids.add(rid)
                candidates.append(row)
        if len(candidates) >= 4:
            break
    return candidates


def _choose_candidate(candidates: list[dict[str, Any]], chat_id: int, thread_id: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    last_id = _last_sent_media_id.get((chat_id, thread_id))
    pool = [c for c in candidates if int(c["id"]) != last_id] if len(candidates) > 1 and last_id is not None else candidates
    # Prefer top least-used items with slight randomness so replies stay fresh
    top_slice = pool[: min(3, len(pool))]
    return random.choice(top_slice)


async def _send_single_media_item(
    bot: Bot,
    *,
    chat_id: int,
    send_thread_id: int | None,
    media_type: str,
    file_id: str,
    reply_to_message_id: int | None,
) -> None:
    if media_type == "sticker":
        await bot.send_sticker(
            chat_id=chat_id,
            sticker=file_id,
            message_thread_id=send_thread_id,
            reply_to_message_id=reply_to_message_id,
        )
    elif media_type == "animation":
        await bot.send_animation(
            chat_id=chat_id,
            animation=file_id,
            message_thread_id=send_thread_id,
            reply_to_message_id=reply_to_message_id,
        )
    elif media_type == "video_note":
        await bot.send_video_note(
            chat_id=chat_id,
            video_note=file_id,
            message_thread_id=send_thread_id,
            reply_to_message_id=reply_to_message_id,
        )
    else:
        raise ValueError(f"Unsupported media_type: {media_type}")


async def send_ai_chat_media_reply(
    bot: Bot,
    *,
    chat_id: int,
    thread_id: int,
    send_thread_id: int | None,
    media_type: str,
    mood: str = "general",
    reply_to_message_id: int | None = None,
) -> bool:
    """Send a sticker, GIF (animation), or video_note from the bank; returns True if sent."""
    if media_type not in repository.AI_CHAT_MEDIA_TYPES:
        return False

    candidates = await _pick_media_candidates(media_type, mood)
    if not candidates:
        return False

    action_name = {
        "sticker": "choose_sticker",
        "animation": "upload_video",
        "video_note": "record_video_note",
    }.get(media_type, "typing")
    try:
        await bot.send_chat_action(
            chat_id=chat_id,
            action=action_name,
            message_thread_id=send_thread_id,
        )
    except Exception:
        pass

    tried_ids: set[int] = set()
    for _attempt in range(min(2, len(candidates))):
        remaining = [c for c in candidates if int(c["id"]) not in tried_ids]
        chosen = _choose_candidate(remaining, chat_id, thread_id)
        if chosen is None:
            break
        item_id = int(chosen["id"])
        tried_ids.add(item_id)
        file_id = str(chosen["file_id"])

        try:
            await _send_single_media_item(
                bot,
                chat_id=chat_id,
                send_thread_id=send_thread_id,
                media_type=media_type,
                file_id=file_id,
                reply_to_message_id=reply_to_message_id,
            )
            _last_sent_media_id[(chat_id, thread_id)] = item_id
            await repository.mark_ai_chat_media_used(item_id)
            return True
        except TelegramBadRequest as exc:
            err_lower = str(exc).lower()
            if "message to be replied not found" in err_lower or "reply" in err_lower:
                try:
                    await _send_single_media_item(
                        bot,
                        chat_id=chat_id,
                        send_thread_id=send_thread_id,
                        media_type=media_type,
                        file_id=file_id,
                        reply_to_message_id=None,
                    )
                    _last_sent_media_id[(chat_id, thread_id)] = item_id
                    await repository.mark_ai_chat_media_used(item_id)
                    return True
                except Exception as retry_exc:
                    logger.warning("Retry without reply_to failed for media #%s: %s", item_id, retry_exc)
            else:
                logger.warning("Telegram rejected AI media #%s (%s): %s", item_id, media_type, exc)
                if chosen.get("source") == "auto":
                    try:
                        await repository.delete_ai_chat_media_by_id(item_id)
                        invalidate_ai_media_counts_cache()
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Unexpected error sending AI media #%s (%s): %s", item_id, media_type, exc)
            break

    return False
