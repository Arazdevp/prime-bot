"""Summarization service handling message formatting, map-reduce chunking, and HTML sanitization."""

import asyncio
from datetime import datetime
import hashlib
import html
import logging
import re
import time
from zoneinfo import ZoneInfo
from bot.config import get_settings
from bot.database import repository
from bot.database.db import is_db_initialized
from bot.services.llm_client import BaseLLMClient, LLMError, LLMTruncationError, get_llm_client
from bot.utils.dates import format_date_heading, get_tehran_now
from bot.utils.link_parser import get_clean_chat_id
from bot.utils.retry import async_retry
from bot.utils.text_split import TELEGRAM_MESSAGE_MAX_CHARS, telegram_text_length

logger = logging.getLogger(__name__)

# Shared LLM serialization, pacing, and throttle-cooldown state across topics, chunks, and previews
_llm_pace_lock = asyncio.Lock()
_next_allowed_call_ts: float = 0.0
_rate_limit_until_ts: float = 0.0
_adaptive_extra_delay: float = 0.0
_adaptive_delay_expires_ts: float = 0.0
_ADAPTIVE_DELAY_STEP_SECONDS = 10.0
_ADAPTIVE_DELAY_MAX_SECONDS = 30.0
_ADAPTIVE_DELAY_TTL_SECONDS = 900.0
_MAX_REDUCE_ROUNDS = 5
SUMMARY_PROMPT_TARGET_CHARS = 3500
_MAX_CONDENSE_ATTEMPTS = 2


def reset_llm_pacer() -> None:
    """Reset in-memory LLM pacing, adaptive backoff, and cooldown timestamps (useful for test isolation)."""
    global _next_allowed_call_ts, _rate_limit_until_ts, _adaptive_extra_delay, _adaptive_delay_expires_ts
    _next_allowed_call_ts = 0.0
    _rate_limit_until_ts = 0.0
    _adaptive_extra_delay = 0.0
    _adaptive_delay_expires_ts = 0.0


def _get_effective_chunk_delay(chunk_delay: float) -> float:
    """Compute effective inter-request delay including any active adaptive in-flight throttle buffer."""
    global _adaptive_extra_delay, _adaptive_delay_expires_ts
    if _adaptive_extra_delay > 0 and time.monotonic() >= _adaptive_delay_expires_ts:
        _adaptive_extra_delay = 0.0
        _adaptive_delay_expires_ts = 0.0
    base = max(0.0, float(chunk_delay))
    return base + _adaptive_extra_delay


def _record_rate_limit_cooldown(delay_seconds: float) -> None:
    """Record a shared cooldown deadline when the LLM API returns HTTP 429 or transient 402."""
    global _rate_limit_until_ts
    if delay_seconds > 0:
        deadline = time.monotonic() + float(delay_seconds)
        if deadline > _rate_limit_until_ts:
            _rate_limit_until_ts = deadline


def _clear_expired_or_resolved_cooldown() -> None:
    """Clear the throttle cooldown deadline after an LLM call succeeds."""
    global _rate_limit_until_ts
    _rate_limit_until_ts = 0.0


def _record_in_flight_throttle(delay_seconds: float) -> None:
    """Record cooldown and adaptively widen inter-request spacing after OpenRouter in-flight 402."""
    global _adaptive_extra_delay, _adaptive_delay_expires_ts
    _record_rate_limit_cooldown(delay_seconds)
    _adaptive_extra_delay = min(
        _ADAPTIVE_DELAY_MAX_SECONDS,
        _adaptive_extra_delay + _ADAPTIVE_DELAY_STEP_SECONDS,
    )
    _adaptive_delay_expires_ts = time.monotonic() + _ADAPTIVE_DELAY_TTL_SECONDS
    logger.warning(
        "OpenRouter in-flight credit limit triggered; adding +%.1fs adaptive delay between LLM calls for %.0fs.",
        _adaptive_extra_delay,
        _ADAPTIVE_DELAY_TTL_SECONDS,
    )


async def _wait_for_llm_slot_locked(chunk_delay: float) -> None:
    """Wait for any active throttle cooldown or inter-chunk pacing slot (caller MUST hold _llm_pace_lock)."""
    effective_delay = _get_effective_chunk_delay(chunk_delay)
    now = time.monotonic()
    wait_until = max(_rate_limit_until_ts, _next_allowed_call_ts if effective_delay > 0 else 0.0)
    if wait_until > now:
        sleep_for = wait_until - now
        logger.debug("Pacing LLM call: sleeping %.2fs before next request.", sleep_for)
        await asyncio.sleep(sleep_for)


async def _wait_for_llm_slot(chunk_delay: float) -> None:
    """Wait for any active rate-limit cooldown or inter-chunk pacing slot before calling LLM."""
    global _next_allowed_call_ts
    async with _llm_pace_lock:
        await _wait_for_llm_slot_locked(chunk_delay)
        effective_delay = _get_effective_chunk_delay(chunk_delay)
        if effective_delay > 0:
            _next_allowed_call_ts = time.monotonic() + effective_delay


def _compute_chunk_hash(chat_id: int, thread_id: int, link_prefix: str, chunk_text: str) -> str:
    """Compute a deterministic SHA-256 cache key for a Map-Reduce transcript chunk."""
    payload = f"{chat_id}:{thread_id}:{link_prefix}\n{chunk_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

SYSTEM_PROMPT_SINGLE = """شما یک دستیار هوشمند و تحلیل‌گر حرفه‌ای برای گروه‌های تلگرام هستید. وظیفه شما خواندن متن گفتگوهای روزانه یک تاپیک و تولید یک خلاصه دقیق، گزینشی، ساختاریافته، بسیار خوانا و سلیس به زبان فارسی است که فقط مهم‌ترین و ارزشمندترین صحبت‌ها را پوشش دهد.

قوانین حیاتی:
۱. فقط و فقط از اطلاعات موجود در پیام‌ها استفاده کنید. هرگز از خود چیزی حدس نزنید یا اضافه نکنید.
۲. اولویت‌بندی سخت‌گیرانه: فقط صحبت‌های خیلی مهم، تصمیم‌های جدی، پرسش‌ها و پاسخ‌های کلیدی، راه‌حل‌های کاربردی و اخبار مهم را انتخاب کنید. خوش‌وبش، احوالپرسی، شوخی، تعارفات، حاشیه‌ها، بحث‌های فرعی کم‌اهمیت و پیام‌های تکراری را کاملاً حذف کنید.
۳. اسامی افراد، پیوندها (لینک‌ها) و اعداد/مبالغ مهم را با نهایت دقت و بدون تحریف حفظ کنید.
۴. قانون پیوند آبی به پیام‌های کلیدی:
در متن ورودی، هر پیام با شناسه مانند [#پیام ۱۲۳] مشخص شده است. هر زمان که یک تصمیم نهایی، خبر مهم، پاسخ کلیدی، راه‌حل یا سؤال بی‌پاسخ مطرح شد، یک عبارت کوتاه و معنادار را به پیام اصلی پیوند دهید تا در تلگرام به رنگ آبی و قابل کلیک باشد:
<a href="https://t.me/c/{clean_chat_id}/MESSAGE_ID">عبارت کلیدی نکته مهم</a>
(فقط عبارت کلیدی مرتبط را لینک کنید؛ از چسباندن زنجیره‌ای شماره‌های خام پیام مثل (۱۲۳، ۱۲۴، ۱۲۵) یا لینک کردن کل پاراگراف خودداری کنید).
۵. قانون فاصله‌گذاری و خوانایی:
بین هر سرفصل و مبحث، حتماً دو خط خالی (فاصله باز) قرار دهید تا متن کاملاً خوانا، تفکیک‌شده و چشم‌نواز باشد.
۶. صرفاً از تگ‌های مجاز HTML تلگرام استفاده کنید: <b>, <i>, <code>, <blockquote>, <a href="...">. هیچ‌گونه مارک‌داون (مانند ** یا #) استفاده نکنید. کاراکترهای < و > و & در متن ساده را حتماً Escape کنید.
۷. ساختار خروجی شما باید دقیقاً به شکل زیر با هدینگ‌های مشخص و دو خط خالی بین بخش‌ها باشد (بخش‌های بدون محتوا را حذف کنید):

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}


🔹 <b>محورهای اصلی بحث</b>
▫️ <b>عنوان مبحث اول:</b>
• شرح موجز گفتگو و <a href="https://t.me/c/{clean_chat_id}/ID">نکته کلیدی به رنگ آبی</a>...


✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• <a href="https://t.me/c/{clean_chat_id}/ID">تصمیم اتخاذ شده</a> یا نتیجه نهایی (در صورت عدم وجود حذف شود)


🔗 <b>لینک‌ها و منابع مهم</b>
• پیوندها به صورت <a href="...">عنوان لینک</a> یا توضیحات مختصر (در صورت عدم وجود حذف شود)


❓ <b>سؤالات بی‌پاسخ</b>
• <a href="https://t.me/c/{clean_chat_id}/ID">سؤال مهم بی‌پاسخ</a> (در صورت عدم وجود حذف شود)

۸. قانون قطعی یک‌پیامی (حداکثر {max_chars} کاراکتر):
کل خروجی شما باید **حتماً در یک پیام تلگرامی واحد** جا بگیرد؛ طول متن قابل مشاهده (بدون احتساب آدرس تگ‌های HTML) باید **حداکثر {max_chars} کاراکتر** باشد (هدف‌گذاری ایده‌آل حدود {target_chars} کاراکتر است و به هیچ وجه از {max_chars} کاراکتر عبور نکنید).
متن به هیچ عنوان نباید نصفه‌نیمه یا ناتمام بماند؛ اگر حجم گفتگوها زیاد است، به‌جای زیاد کردن متن، فقط مهم‌ترین مباحث را گزینش کنید و بسیار موجز و کامل بنویسید تا همه جملات و بخش‌ها در همین یک پیام به پایان برسند.
۹. هیچ‌گونه مقدمه، نتیجه‌گیری، سلام، خداحافظی یا اشاره به این‌که شما هوش مصنوعی هستید اضافه نکنید."""

SYSTEM_PROMPT_CHUNK = """شما یک دستیار هوشمند خلاصه‌سازی هستید. یک بخش از پیام‌های روزانه یک تاپیک تلگرام به همراه شناسه پیام‌ها به شما داده می‌شود. وظیفه شما استخراج فقط صحبت‌های خیلی مهم، مباحث کلیدی، اسامی مهم، تصمیمات، لینک‌ها و سؤالات اصلی مطرح‌شده در این بخش است.

قوانین:
۱. متن خلاصه را بسیار موجز، دقیق و به زبان فارسی سلیس بنویسید.
۲. احوالپرسی، شوخی، پیام‌های تکراری، حاشیه‌های کم‌اهمیت و گفتگوهای فاقد بار اطلاعاتی را کاملاً حذف کنید.
۳. برای هر نکته یا تصمیم مهم، یک عبارت کوتاه را با شناسه پیام آن پیوند دهید (<a href="https://t.me/c/{clean_chat_id}/ID">عبارت کلیدی</a>) و از نوشتن لیست شماره‌های خام پیام خودداری کنید.
۴. از تگ‌های HTML مجاز تلگرام استفاده کنید.
۵. بین هر مبحث فاصله مناسب قرار دهید.
۶. خروجی این بخش را فشرده و حدود ۱۰۰۰ تا ۲۰۰۰ کاراکتر نگه دارید تا در مرحله ترکیب نهایی به‌راحتی در یک پیام تلگرامی جا بگیرد."""

SYSTEM_PROMPT_MERGE = """شما یک دستیار تحلیل‌گر ارشد هستید. به شما خلاصه‌های بخش‌های مختلف از گفتگوهای یک تاپیک تلگرام داده می‌شود. وظیفه شما ترکیب، گزینش، یکپارچه‌سازی و حذف موارد تکراری و کم‌اهمیت برای تولید یک خلاصه نهایی جامع و موجز در قالب یک پیام تلگرامی واحد به زبان فارسی است.

قوانین حیاتی:
۱. ساختار نهایی باید دقیقاً و الزاماً مطابق این الگو با هدینگ‌های مشخص و دو خط خالی بین هر بخش باشد:

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}


🔹 <b>محورهای اصلی بحث</b>
▫️ <b>مبحث:</b>
• نکات اصلی و خیلی مهم گفتگو...


✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• تصمیم‌ها... (در صورت عدم وجود حذف شود)


🔗 <b>لینک‌ها و منابع مهم</b>
• پیوندها... (در صورت عدم وجود حذف شود)


❓ <b>سؤالات بی‌پاسخ</b>
• سؤالات بی‌جواب... (در صورت عدم وجود حذف شود)

۲. گزینش صحبت‌های خیلی مهم و پیوند آبی: فقط مهم‌ترین موضوعات، تصمیم‌ها و پاسخ‌های کلیدی را نگه دارید، موارد جزئی یا تکراری را حذف کنید و هر نکته اساسی را با تگ <a href="https://t.me/c/{clean_chat_id}/ID">عبارت کلیدی</a> پیوند دهید تا به رنگ آبی نمایش داده شود (از فهرست‌کردن شماره‌های خام پیام پرهیز کنید).
۳. فاصله‌گذاری: حتماً بین هر بخش دو خط خالی قرار دهید تا متن بسیار خوانا و باز باشد.
۴. فقط از تگ‌های مجاز تلگرام (<b>, <i>, <code>, <blockquote>, <a href="...">) استفاده کنید و هیچ مارک‌داونی به کار نبرید.
۵. قانون قطعی یک‌پیامی (حداکثر {max_chars} کاراکتر):
کل خروجی باید **حتماً در یک پیام تلگرامی واحد** جا بگیرد؛ طول متن قابل مشاهده (بدون احتساب آدرس تگ‌های HTML) باید **حداکثر {max_chars} کاراکتر** باشد (هدف‌گذاری ایده‌آل حدود {target_chars} کاراکتر است و هرگز از {max_chars} کاراکتر فراتر نروید).
هرگز متن را نصفه‌نیمه رها نکنید؛ با حذف جزئیات کم‌اهمیت و ادغام موارد مشابه، خلاصه‌ای کامل، تمام‌شده و دقیق بنویسید.
۶. هیچ مقدمه، مؤخره یا توضیحی فراتر از قالب ساختاریافته فوق اضافه نکنید."""

SYSTEM_PROMPT_CONDENSE = """شما یک ویراستار ارشد خلاصه‌سازی تلگرام هستید. پیش‌نویس خلاصه زیر {current_chars} کاراکتر است و از سقف یک پیام تلگرام ({max_chars} کاراکتر) بیشتر شده است.

وظیفه شما بازنویسی و فشرده‌سازی این خلاصه است به‌طوری‌که **حتماً در یک پیام تلگرامی واحد** (حدود {target_chars} کاراکتر و اکیداً حداکثر {max_chars} کاراکتر متن قابل مشاهده، بدون احتساب آدرس تگ‌های HTML) جا بگیرد و هیچ جمله‌ای نصفه‌نیمه نماند.

قوانین بازنویسی:
۱. فقط مهم‌ترین صحبت‌ها، تصمیم‌های کلیدی، لینک‌های ضروری و سؤالات مهم بی‌پاسخ را نگه دارید و نکات کم‌اهمیت، جزئیات تکراری و حاشیه‌ها را حذف یا ادغام کنید.
۲. قالب استاندارد زیر را حفظ کنید (بخش‌های خالی را حذف کنید):
📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}
(به همراه 🔹 محورهای اصلی بحث، ✅ تصمیم‌ها و جمع‌بندی‌ها، 🔗 لینک‌ها و منابع مهم، ❓ سؤالات بی‌پاسخ در صورت وجود)
۳. تگ‌های مجاز HTML تلگرام (<b>, <i>, <code>, <blockquote>, <a href="...">) و لینک‌های آبی <a href="https://t.me/c/{clean_chat_id}/ID">...</a> مهم را سالم نگه دارید و بدون مارک‌داون بنویسید.
۴. هرگز هیچ اطلاعاتی از خود اضافه نکنید و هیچ مقدمه یا توضیح اضافه‌ای درباره کوتاه‌سازی ننویسید."""

ALLOWED_HTML_TAGS = {
    "b",
    "strong",
    "i",
    "em",
    "u",
    "ins",
    "s",
    "strike",
    "del",
    "code",
    "pre",
    "blockquote",
    "a",
}

MEDIA_TYPE_MARKERS = {
    "photo": "[عکس] ",
    "video": "[ویدیو] ",
    "document": "[فایل] ",
    "voice": "[ویس] ",
    "audio": "[صوت] ",
    "video_note": "[پیام ویدیویی] ",
}


def format_messages(messages: list[dict]) -> str:
    """Format raw message records into chronological Persian transcripts with message IDs."""
    tehran_tz = ZoneInfo("Asia/Tehran")
    formatted_lines: list[str] = []

    for msg in messages:
        raw_created = msg.get("created_at", "")
        try:
            dt = datetime.fromisoformat(raw_created)
            tehran_dt = dt.astimezone(tehran_tz) if dt.tzinfo else dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tehran_tz)
            time_str = tehran_dt.strftime("%H:%M")
        except Exception:
            time_str = "00:00"

        msg_id = msg.get("message_id")
        id_str = f"[#پیام {msg_id}] " if msg_id else ""

        sender_name = msg.get("sender_name") or "کاربر"
        reply_name = msg.get("reply_to_name")
        reply_snippet = msg.get("reply_to_snippet")
        reply_str = ""
        if reply_name and reply_snippet:
            clean_snippet = reply_snippet.replace("\n", " ").strip()
            reply_str = f" ↩️(پاسخ به «{reply_name}»: «{clean_snippet}»)"

        media_type = msg.get("media_type")
        media_prefix = MEDIA_TYPE_MARKERS.get(media_type, "")

        text = re.sub(r"\n{3,}", "\n\n", (msg.get("text") or "").strip())
        formatted_lines.append(f"{id_str}[{time_str}] {sender_name}{reply_str}: {media_prefix}{text}")

    return "\n".join(formatted_lines)


def chunk_transcript(transcript: str, max_chars: int = 24000) -> list[str]:
    """Split a long transcript on line boundaries into chunks not exceeding max_chars."""
    lines = transcript.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        # If a single line is longer than max_chars, hard-truncate with ellipsis
        if len(line) > max_chars:
            line = line[: max_chars - 3] + "…"

        line_len = len(line) + 1  # include newline char
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _apply_inline_markdown_outside_tags(text: str) -> str:
    """Convert **bold** and __underline__ only in plain text outside HTML tags and code blocks."""
    segments = re.split(r"(<[^>]+>)", text)
    in_code_depth = 0
    result: list[str] = []

    for seg in segments:
        if not seg:
            continue
        if seg.startswith("<") and seg.endswith(">"):
            tag_match = re.match(r"^<\s*(/)?\s*(code|pre)(?:\s|>)", seg, flags=re.IGNORECASE)
            if tag_match:
                if tag_match.group(1):
                    in_code_depth = max(0, in_code_depth - 1)
                else:
                    in_code_depth += 1
            result.append(seg)
        elif in_code_depth > 0:
            result.append(seg)
        else:
            converted = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", seg)
            converted = re.sub(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)", r"<u>\1</u>", converted)
            result.append(converted)

    return "".join(result)


def sanitize_telegram_html(
    text: str,
    clean_chat_id: str | None = None,
    thread_id: int | None = None,
) -> str:
    """Clean model output, convert HTML/markdown headings, escape stray characters, and balance allowed tags."""
    if not text:
        return ""

    # 1. Strip markdown code fences
    cleaned = re.sub(r"^```(?:html)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 2. Convert HTML headings <h1>..<h6> to Telegram-friendly styled bold headings with generous spacing
    cleaned = re.sub(r"<h1>(.*?)</h1>", r"\n\n\n📌 <b>\1</b>\n\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<h2>(.*?)</h2>", r"\n\n\n🔹 <b>\1</b>\n\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<h3>(.*?)</h3>", r"\n\n▫️ <b>\1</b>\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<h[4-6]>(.*?)</h[4-6]>", r"\n\n▪️ <b>\1</b>\n", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # 2b. Convert common non-Telegram structural HTML tags into clean newlines/bullets
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(?:p|div)>", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<(?:p|div)(?:\s+[^>]*)?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?(?:ul|ol)(?:\s+[^>]*)?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<li(?:\s+[^>]*)?>\s*", "• ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</li>", "\n", cleaned, flags=re.IGNORECASE)

    # 3. Convert Markdown headings #..### to styled bold headings with double newlines
    cleaned = re.sub(r"(?m)^#\s+(.+)$", r"\n\n📌 <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^##\s+(.+)$", r"\n\n🔹 <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^###\s+(.+)$", r"\n\n▫️ <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^#{4,6}\s+(.+)$", r"\n\n▪️ <b>\1</b>\n", cleaned)

    # 4. Convert other common markdown patterns ONLY outside HTML tags and code blocks
    cleaned = _apply_inline_markdown_outside_tags(cleaned)

    # 5. Normalize multiple empty lines to at most two empty lines (3 newlines)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)

    # Helper: escape ampersands and angle brackets ONLY in text nodes
    def _sanitize_text_segment(segment: str) -> str:
        # Escape lone ampersands not already part of valid entities
        segment = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", segment)
        # Escape stray angle brackets
        segment = segment.replace("<", "&lt;").replace(">", "&gt;")
        return segment

    # 6. Tokenize and sanitize tags
    tag_pattern = re.compile(r"<(/)?([a-zA-Z0-9_-]+)(?:\s+([^>]*))?>")
    tokens: list[str] = []
    last_idx = 0
    open_stack: list[str] = []

    for m in tag_pattern.finditer(cleaned):
        # Escape stray angle brackets and lone ampersands in text before tag
        if m.start() > last_idx:
            segment = cleaned[last_idx : m.start()]
            tokens.append(_sanitize_text_segment(segment))

        is_closing = bool(m.group(1))
        tag_name = m.group(2).lower()
        attributes = m.group(3) or ""

        if tag_name not in ALLOWED_HTML_TAGS:
            # Escape disallowed tag
            disallowed_tag = _sanitize_text_segment(m.group(0))
            tokens.append(disallowed_tag)
        else:
            if is_closing:
                # Find matching opening tag in stack
                if tag_name in open_stack:
                    while open_stack:
                        top = open_stack.pop()
                        tokens.append(f"</{top}>")
                        if top == tag_name:
                            break
            else:
                if tag_name == "a":
                    # Avoid nested <a> tags (rejected by Telegram HTML parser)
                    if "a" in open_stack:
                        last_idx = m.end()
                        continue
                    # Extract href attribute safely
                    href_match = re.search(r'href=[\"\']([^\"\']+)[\"\']', attributes)
                    raw_href = href_match.group(1).strip() if href_match else ""
                    # Auto-expand relative message IDs (#123, 123, /123, msg:123)
                    num_match = re.match(r"^(?:#|/|msg:)?(\d+)$", raw_href)
                    if num_match and clean_chat_id:
                        if thread_id and thread_id != 1:
                            href = f"https://t.me/c/{clean_chat_id}/{thread_id}/{num_match.group(1)}"
                        else:
                            href = f"https://t.me/c/{clean_chat_id}/{num_match.group(1)}"
                    else:
                        # Check if href is 2-part t.me/c/{clean_chat_id}/{msg_id} when non-General thread_id is set
                        m2 = (
                            re.match(rf"^(?:https?://)?(?:www\.)?t\.me/c/{re.escape(clean_chat_id)}/(\d+)$", raw_href)
                            if clean_chat_id and thread_id and thread_id != 1
                            else None
                        )
                        if m2:
                            href = f"https://t.me/c/{clean_chat_id}/{thread_id}/{m2.group(1)}"
                        else:
                            # In href: unescape first to prevent double-escaping (&amp; -> &), then escape once
                            href = html.unescape(raw_href)
                            if href.lower().startswith("www."):
                                href = f"https://{href}"

                    # Only emit <a> for valid Telegram-supported URL schemes without whitespace
                    if (
                        href
                        and not any(ch.isspace() for ch in href)
                        and re.match(r"^(?:https?://|tg://|mailto:)", href, flags=re.IGNORECASE)
                    ):
                        tokens.append(f'<a href="{html.escape(href, quote=True)}">')
                        open_stack.append("a")
                else:
                    tokens.append(f"<{tag_name}>")
                    open_stack.append(tag_name)

        last_idx = m.end()

    # Remainder text after last tag
    if last_idx < len(cleaned):
        segment = cleaned[last_idx:]
        tokens.append(_sanitize_text_segment(segment))

    # Close any unclosed tags at the end
    while open_stack:
        tokens.append(f"</{open_stack.pop()}>")

    return "".join(tokens).strip()


def _batch_summaries_for_reduce(
    summaries: list[str],
    max_chars: int,
    sep: str = "\n\n---\n\n",
) -> list[list[str]]:
    """Group partial summaries into batches whose joined length fits within max_chars."""
    safe_limit = max(1000, max_chars)

    def _pack(items: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for item in items:
            trimmed = item if len(item) <= safe_limit else item[: safe_limit - 1] + "…"
            add_len = len(trimmed) if not current else len(sep) + len(trimmed)
            if current and current_len + add_len > safe_limit:
                batches.append(current)
                current = [trimmed]
                current_len = len(trimmed)
            else:
                current.append(trimmed)
                current_len += add_len
        if current:
            batches.append(current)
        return batches

    batches = _pack(summaries)
    # Ensure hierarchical reduce always shrinks item count even if individual items exceed half max_chars
    if len(summaries) > 1 and len(batches) >= len(summaries):
        half_cap = max(500, (safe_limit - len(sep)) // 2)
        capped = [s if len(s) <= half_cap else s[: half_cap - 1] + "…" for s in summaries]
        batches = _pack(capped)
    return batches


async def _generate_with_retry(
    llm_client: BaseLLMClient,
    system_prompt: str,
    user_content: str,
    max_retries: int = 3,
    throttle_retries: int | None = None,
    chunk_delay: float = 0.0,
) -> str:
    """Execute a strictly serialized LLM call with pacing, throttle-aware retry, and truncation recovery."""
    global _next_allowed_call_ts

    async def _call_once(allow_truncated: bool) -> str:
        return await llm_client.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            allow_truncated=allow_truncated,
        )

    # Hold _llm_pace_lock for the full request lifecycle so this bot NEVER has >1 in-flight LLM request
    async with _llm_pace_lock:
        await _wait_for_llm_slot_locked(chunk_delay)
        try:
            try:
                result = await async_retry(
                    lambda: _call_once(allow_truncated=False),
                    retries=max_retries,
                    throttle_retries=throttle_retries,
                    on_rate_limit=_record_rate_limit_cooldown,
                    on_in_flight_budget=_record_in_flight_throttle,
                )
                _clear_expired_or_resolved_cooldown()
                return result
            except LLMTruncationError as exc:
                logger.warning(
                    "LLM response was truncated (token limit reached). Retrying generation once with allow_truncated=True..."
                )
                now = time.monotonic()
                if _rate_limit_until_ts > now:
                    await asyncio.sleep(_rate_limit_until_ts - now)
                try:
                    result = await async_retry(
                        lambda: _call_once(allow_truncated=True),
                        retries=max_retries,
                        throttle_retries=throttle_retries,
                        on_rate_limit=_record_rate_limit_cooldown,
                        on_in_flight_budget=_record_in_flight_throttle,
                    )
                    _clear_expired_or_resolved_cooldown()
                    return result
                except LLMTruncationError as exc_persisted:
                    logger.warning(
                        "LLM response truncation persisted on retry (%d chars). Returning truncated output for sanitization.",
                        len(exc_persisted.truncated_text),
                    )
                    _clear_expired_or_resolved_cooldown()
                    return exc_persisted.truncated_text
                except LLMError as retry_exc:
                    if exc.truncated_text:
                        logger.warning(
                            "Retry after truncation failed (%s); falling back to initial truncated output (%d chars).",
                            retry_exc.message,
                            len(exc.truncated_text),
                        )
                        return exc.truncated_text
                    raise
        finally:
            effective_delay = _get_effective_chunk_delay(chunk_delay)
            if effective_delay > 0:
                _next_allowed_call_ts = max(_next_allowed_call_ts, time.monotonic() + effective_delay)


async def _condense_to_single_message(
    summary_html: str,
    topic_name: str,
    date_heading: str,
    link_prefix: str,
    clean_chat_id: str,
    target_thread: int | None,
    max_rendered_chars: int,
    llm_client: BaseLLMClient,
    max_retries: int,
    throttle_retries: int,
    chunk_delay: float,
) -> str:
    """Ensure sanitized summary fits in a single Telegram message via AI condensation when needed."""
    best = summary_html
    best_len = telegram_text_length(best)
    if best_len <= max_rendered_chars:
        return best

    logger.warning(
        "Summary for topic '%s' rendered length (%d) exceeds single-message limit (%d). "
        "Running AI condensation pass.",
        topic_name,
        best_len,
        max_rendered_chars,
    )

    attempt_targets = [
        min(max(400, int(max_rendered_chars * 0.85)), max(350, max_rendered_chars - 250)),
        min(max(350, int(max_rendered_chars * 0.72)), max(300, max_rendered_chars - 450)),
    ]

    for attempt in range(1, _MAX_CONDENSE_ATTEMPTS + 1):
        target_chars = attempt_targets[min(attempt - 1, len(attempt_targets) - 1)]
        system_prompt = (
            SYSTEM_PROMPT_CONDENSE.replace("{topic_name}", topic_name)
            .replace("{date_heading}", date_heading)
            .replace("{clean_chat_id}", link_prefix)
            .replace("{current_chars}", str(best_len))
            .replace("{max_chars}", str(max_rendered_chars))
            .replace("{target_chars}", str(target_chars))
        )
        user_content = (
            f"پیش‌نویس خلاصه تاپیک «{topic_name}» ({best_len} کاراکتر):\n\n"
            f"{best}\n\n"
            f"لطفاً با حفظ مهم‌ترین صحبت‌ها و حذف موارد فرعی، این خلاصه را کامل و بدون جمله‌های ناتمام در حدود {target_chars} کاراکتر (اکیداً حداکثر {max_rendered_chars} کاراکتر) بازنویسی کنید."
        )

        try:
            raw_condensed = await _generate_with_retry(
                llm_client=llm_client,
                system_prompt=system_prompt,
                user_content=user_content,
                max_retries=max_retries,
                throttle_retries=throttle_retries,
                chunk_delay=chunk_delay,
            )
        except LLMError as exc:
            logger.warning(
                "AI condensation attempt %d/%d failed for topic '%s': %s. Using best candidate (%d chars).",
                attempt,
                _MAX_CONDENSE_ATTEMPTS,
                topic_name,
                exc,
                best_len,
            )
            return best

        sanitized = sanitize_telegram_html(raw_condensed, clean_chat_id=clean_chat_id, thread_id=target_thread)
        if not sanitized:
            continue

        cand_len = telegram_text_length(sanitized)
        if cand_len < best_len:
            best = sanitized
            best_len = cand_len

        if best_len <= max_rendered_chars:
            logger.info(
                "AI condensation for topic '%s' succeeded on attempt %d (%d chars <= %d).",
                topic_name,
                attempt,
                best_len,
                max_rendered_chars,
            )
            return best

    logger.warning(
        "Summary for topic '%s' remained %d chars (> %d) after %d condensation attempts; returning shortest candidate.",
        topic_name,
        best_len,
        max_rendered_chars,
        _MAX_CONDENSE_ATTEMPTS,
    )
    return best


async def summarize_topic_messages(
    topic_name: str,
    messages: list[dict],
    chat_id: int | None = None,
    thread_id: int | None = None,
    min_messages: int = 5,
    is_nightly: bool = True,
    llm_client: BaseLLMClient | None = None,
    use_chunk_cache: bool = False,
    max_rendered_chars: int = TELEGRAM_MESSAGE_MAX_CHARS,
) -> str | None:
    """Summarize a list of messages for a topic.

    Returns None if messages count is below threshold.
    Performs Map-Reduce chunking if transcript length exceeds CHUNK_MAX_CHARS.
    When use_chunk_cache=True and SQLite is initialized, completed Map chunks are persisted
    so interrupted runs resume without re-paying for already summarized chunks.
    Enforces single-message rendered character budget via prompt limits and automatic AI condensation.
    """
    if len(messages) < min_messages:
        logger.info(
            "Topic '%s' has %s messages (below threshold %s). Skipping summarization.",
            topic_name,
            len(messages),
            min_messages,
        )
        return None

    settings = get_settings()
    if llm_client is None:
        llm_client = get_llm_client(settings)

    target_chat = chat_id or (messages[0].get("chat_id") if messages else None)
    clean_chat_id = get_clean_chat_id(target_chat) if target_chat else ""
    target_thread = thread_id if thread_id is not None else (messages[0].get("thread_id") if messages else None)

    transcript = format_messages(messages)

    # Derive heading date from the earliest included message in Asia/Tehran
    if messages:
        try:
            first_created = messages[0].get("created_at") or ""
            first_dt = datetime.fromisoformat(first_created)
            tehran_tz = ZoneInfo(settings.TIMEZONE)
            tehran_dt = (
                first_dt.astimezone(tehran_tz)
                if first_dt.tzinfo
                else first_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tehran_tz)
            )
        except Exception:
            tehran_dt = get_tehran_now()
    else:
        tehran_dt = get_tehran_now()

    date_heading = format_date_heading(tehran_dt)

    # Prefix used in system prompts for message hyperlinks
    if target_thread and target_thread != 1:
        link_prefix = f"{clean_chat_id}/{target_thread}"
    else:
        link_prefix = clean_chat_id or "chat"

    raw_chunk_delay = getattr(settings, "AI_CHUNK_DELAY_SECONDS", 6.0)
    chunk_delay = float(raw_chunk_delay) if isinstance(raw_chunk_delay, (int, float)) else 6.0

    raw_throttle_retries = getattr(settings, "AI_THROTTLE_MAX_RETRIES", 6)
    cfg_throttle_retries = (
        int(raw_throttle_retries)
        if isinstance(raw_throttle_retries, int) and not isinstance(raw_throttle_retries, bool)
        else 6
    )
    throttle_retries = max(settings.AI_MAX_RETRIES, cfg_throttle_retries) if settings.AI_MAX_RETRIES > 0 else 0

    effective_max = max(500, int(max_rendered_chars))
    prompt_target = min(SUMMARY_PROMPT_TARGET_CHARS, max(400, effective_max - 400))

    # Check if single call or map-reduce
    if len(transcript) <= settings.CHUNK_MAX_CHARS:
        system_prompt = (
            SYSTEM_PROMPT_SINGLE.replace("{topic_name}", topic_name)
            .replace("{date_heading}", date_heading)
            .replace("{clean_chat_id}", link_prefix)
            .replace("{max_chars}", str(effective_max))
            .replace("{target_chars}", str(prompt_target))
        )
        user_content = (
            f"متن گفتگوهای تاپیک «{topic_name}»:\n\n"
            f"{transcript}\n\n"
            f"لطفاً فقط مهم‌ترین صحبت‌ها را با دو خط فاصله و پیوندهای آبی خلاصه کنید؛ خروجی باید کامل و حداکثر {effective_max} کاراکتر باشد تا در یک پیام تلگرامی جا بگیرد."
        )

        raw_output = await _generate_with_retry(
            llm_client=llm_client,
            system_prompt=system_prompt,
            user_content=user_content,
            max_retries=settings.AI_MAX_RETRIES,
            throttle_retries=throttle_retries,
            chunk_delay=chunk_delay,
        )
        sanitized = sanitize_telegram_html(raw_output, clean_chat_id=clean_chat_id, thread_id=target_thread)
        return await _condense_to_single_message(
            summary_html=sanitized,
            topic_name=topic_name,
            date_heading=date_heading,
            link_prefix=link_prefix,
            clean_chat_id=clean_chat_id,
            target_thread=target_thread,
            max_rendered_chars=effective_max,
            llm_client=llm_client,
            max_retries=settings.AI_MAX_RETRIES,
            throttle_retries=throttle_retries,
            chunk_delay=chunk_delay,
        )

    else:
        # Map-Reduce pipeline with pacing, chunk checkpointing, safety-block resilience, and hierarchical reduce
        chunks = chunk_transcript(transcript, max_chars=settings.CHUNK_MAX_CHARS)
        logger.info(
            "Topic '%s' transcript length %s exceeds limit %s. Running Map-Reduce over %s chunks.",
            topic_name,
            len(transcript),
            settings.CHUNK_MAX_CHARS,
            len(chunks),
        )

        can_cache_chunks = bool(
            use_chunk_cache
            and target_chat is not None
            and target_thread is not None
            and is_db_initialized()
        )
        chunk_prompt = SYSTEM_PROMPT_CHUNK.replace("{clean_chat_id}", link_prefix)
        partial_summaries: list[str] = []
        skipped_chunks = 0
        for i, chunk in enumerate(chunks, start=1):
            chunk_hash: str | None = None
            if can_cache_chunks:
                chunk_hash = _compute_chunk_hash(int(target_chat), int(target_thread), link_prefix, chunk)
                try:
                    cached_partial = await repository.get_cached_map_chunk(
                        int(target_chat),
                        int(target_thread),
                        chunk_hash,
                    )
                    if cached_partial:
                        logger.info(
                            "Reusing cached map summary for chunk %s/%s of topic '%s'.",
                            i,
                            len(chunks),
                            topic_name,
                        )
                        partial_summaries.append(cached_partial.strip())
                        continue
                except Exception as cache_err:
                    logger.debug("Chunk cache read skipped for topic '%s': %s", topic_name, cache_err)

            chunk_user_content = (
                f"بخش {i} از {len(chunks)} پیام‌های تاپیک «{topic_name}»:\n\n"
                f"{chunk}\n\n"
                "فقط مهم‌ترین صحبت‌ها، تصمیمات و مباحث کلیدی این بخش را با شناسه‌های پیام استخراج کنید."
            )

            try:
                partial_summary = await _generate_with_retry(
                    llm_client=llm_client,
                    system_prompt=chunk_prompt,
                    user_content=chunk_user_content,
                    max_retries=settings.AI_MAX_RETRIES,
                    throttle_retries=throttle_retries,
                    chunk_delay=chunk_delay,
                )
                cleaned_partial = partial_summary.strip()
                partial_summaries.append(cleaned_partial)
                if can_cache_chunks and chunk_hash is not None:
                    try:
                        await repository.save_cached_map_chunk(
                            int(target_chat),
                            int(target_thread),
                            chunk_hash,
                            cleaned_partial,
                        )
                    except Exception as cache_err:
                        logger.debug("Chunk cache write skipped for topic '%s': %s", topic_name, cache_err)
            except LLMError as exc:
                # Skip individual safety/content-blocked chunks so one edgy message doesn't break the whole topic
                if len(chunks) > 1 and exc.status_code == 200 and not exc.is_retryable:
                    skipped_chunks += 1
                    logger.warning(
                        "Skipping chunk %s/%s for topic '%s' due to non-retryable content/safety response: %s",
                        i,
                        len(chunks),
                        topic_name,
                        exc.message,
                    )
                    continue
                raise

        if not partial_summaries:
            raise LLMError(
                f"All {len(chunks)} chunks failed content/safety checks for topic '{topic_name}'",
                status_code=200,
                is_retryable=False,
            )

        if skipped_chunks:
            logger.warning(
                "Completed map phase for topic '%s' with %s/%s chunks skipped due to content filters.",
                topic_name,
                skipped_chunks,
                len(chunks),
            )

        # Hierarchical reduce if combined partial summaries exceed CHUNK_MAX_CHARS
        sep = "\n\n---\n\n"
        reduce_round = 0
        while len(sep.join(partial_summaries)) > settings.CHUNK_MAX_CHARS and reduce_round < _MAX_REDUCE_ROUNDS:
            reduce_round += 1
            batches = _batch_summaries_for_reduce(partial_summaries, max_chars=settings.CHUNK_MAX_CHARS, sep=sep)
            logger.info(
                "Topic '%s' partial summaries (%s chars) exceed limit %s. Running hierarchical reduce round %s (%s batches).",
                topic_name,
                len(sep.join(partial_summaries)),
                settings.CHUNK_MAX_CHARS,
                reduce_round,
                len(batches),
            )
            next_level: list[str] = []
            for b_idx, batch in enumerate(batches, start=1):
                batch_user_content = (
                    f"خلاصه‌های میانی دسته {b_idx} از {len(batches)} برای تاپیک «{topic_name}»:\n\n"
                    + sep.join(batch)
                    + "\n\nلطفاً این خلاصه‌ها را به‌صورت فشرده ادغام کنید، موارد تکراری را حذف نمایید و شناسه‌های پیام‌های کلیدی را حفظ کنید."
                )
                reduced = await _generate_with_retry(
                    llm_client=llm_client,
                    system_prompt=chunk_prompt,
                    user_content=batch_user_content,
                    max_retries=settings.AI_MAX_RETRIES,
                    throttle_retries=throttle_retries,
                    chunk_delay=chunk_delay,
                )
                next_level.append(reduced.strip())
            partial_summaries = next_level

        combined_partials = sep.join(partial_summaries)
        if len(combined_partials) > settings.CHUNK_MAX_CHARS:
            logger.warning(
                "Topic '%s' merged partials (%s chars) still exceed CHUNK_MAX_CHARS (%s) after %s reduce rounds; trimming.",
                topic_name,
                len(combined_partials),
                settings.CHUNK_MAX_CHARS,
                reduce_round,
            )
            combined_partials = combined_partials[: settings.CHUNK_MAX_CHARS - 1] + "…"

        # Final merge step
        merge_system_prompt = (
            SYSTEM_PROMPT_MERGE.replace("{topic_name}", topic_name)
            .replace("{date_heading}", date_heading)
            .replace("{clean_chat_id}", link_prefix)
            .replace("{max_chars}", str(effective_max))
            .replace("{target_chars}", str(prompt_target))
        )
        merge_user_content = (
            f"خلاصه‌های {len(partial_summaries)} بخش تاپیک «{topic_name}»:\n\n"
            + combined_partials
            + f"\n\nاین بخش‌ها را ترکیب کرده و خلاصه نهایی کامل و یک‌پیامی (حداکثر {effective_max} کاراکتر) را با دو خط فاصله و لینک آبی ارائه دهید."
        )

        merged_output = await _generate_with_retry(
            llm_client=llm_client,
            system_prompt=merge_system_prompt,
            user_content=merge_user_content,
            max_retries=settings.AI_MAX_RETRIES,
            throttle_retries=throttle_retries,
            chunk_delay=chunk_delay,
        )
        sanitized = sanitize_telegram_html(merged_output, clean_chat_id=clean_chat_id, thread_id=target_thread)
        return await _condense_to_single_message(
            summary_html=sanitized,
            topic_name=topic_name,
            date_heading=date_heading,
            link_prefix=link_prefix,
            clean_chat_id=clean_chat_id,
            target_thread=target_thread,
            max_rendered_chars=effective_max,
            llm_client=llm_client,
            max_retries=settings.AI_MAX_RETRIES,
            throttle_retries=throttle_retries,
            chunk_delay=chunk_delay,
        )
