"""Summarization service handling message formatting, map-reduce chunking, and HTML sanitization."""

from datetime import datetime
import html
import logging
import re
from zoneinfo import ZoneInfo
from bot.config import get_settings
from bot.services.llm_client import BaseLLMClient, LLMTruncationError, get_llm_client
from bot.utils.dates import format_date_heading, get_tehran_now
from bot.utils.link_parser import get_clean_chat_id
from bot.utils.retry import async_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SINGLE = """شما یک دستیار هوشمند و تحلیل‌گر حرفه‌ای برای گروه‌های تلگرام هستید. وظیفه شما خواندن متن گفتگوهای روزانه یک تاپیک و تولید یک خلاصه دقیق، ساختاریافته، بسیار خوانا، سلیس و بدون حاشیه به زبان فارسی است.

قوانین حیاتی:
۱. فقط و فقط از اطلاعات موجود در پیام‌ها استفاده کنید. هرگز از خود چیزی حدس نزنید یا اضافه نکنید.
۲. خوش‌وبش، احوالپرسی، تعارفات و پیام‌های بی‌محتوا را کاملاً نادیده بگیرید.
۳. اسامی افراد، پیوندها (لینک‌ها) و اعداد/مبالغ مهم را با نهایت دقت و بدون تحریف حفظ کنید.
۴. قانون پیوند آبی به پیام‌های کلیدی:
در متن ورودی، هر پیام با شناسه مانند [#پیام ۱۲۳] مشخص شده است. هر زمان که یک تصمیم نهایی، خبر مهم، پاسخ کلیدی، راه‌حل، یا سؤال بی‌پاسخ مطرح شد، عبارت خلاصه آن نکته را حتماً به پیام اصلی پیوند دهید تا در تلگرام به رنگ آبی و قابل کلیک باشد:
<a href="https://t.me/c/{clean_chat_id}/MESSAGE_ID">متن نکته مهم یا تصمیم</a>
(فقط عبارت کلیدی مرتبط را لینک کنید و از لینک کردن کل پاراگراف خودداری نمایید).
۵. قانون فاصله‌گذاری و خوانایی:
بین هر سرفصل و مبحث، حتماً دو خط خالی (فاصله باز) قرار دهید تا متن به هیچ عنوان فشرده نباشد و کاملاً خوانا، تفکیک‌شده و چشم‌نواز باشد.
۶. صرفاً از تگ‌های مجاز HTML تلگرام استفاده کنید: <b>, <i>, <code>, <blockquote>, <a href="...">. هیچ‌گونه مارک‌داون (مانند ** یا #) استفاده نکنید. کاراکترهای < و > و & در متن ساده را حتماً Escape کنید.
۷. ساختار خروجی شما باید دقیقاً به شکل زیر با هدینگ‌های مشخص و دو خط خالی بین بخش‌ها باشد (بخش‌های بدون محتوا را حذف کنید):

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}


🔹 <b>محورهای اصلی بحث</b>
▫️ <b>عنوان مبحث اول:</b>
• شرح گفتگو و <a href="https://t.me/c/{clean_chat_id}/ID">نکته کلیدی به رنگ آبی</a>...


✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• <a href="https://t.me/c/{clean_chat_id}/ID">تصمیم اتخاذ شده</a> یا نتیجه نهایی (در صورت عدم وجود حذف شود)


🔗 <b>لینک‌ها و منابع مهم</b>
• پیوندها به صورت <a href="...">عنوان لینک</a> یا توضیحات مختصر (در صورت عدم وجود حذف شود)


❓ <b>سؤالات بی‌پاسخ</b>
• <a href="https://t.me/c/{clean_chat_id}/ID">سؤال مهم بی‌پاسخ</a> (در صورت عدم وجود حذف شود)

۸. طول کل خروجی معمولاً باید زیر ۳۵۰۰ کاراکتر باشد مگر اینکه حجم محتوای معنادار واقعاً زیاد باشد.
۹. هیچ‌گونه مقدمه، نتیجه‌گیری، سلام، خداحافظی یا اشاره به این‌که شما هوش مصنوعی هستید اضافه نکنید."""

SYSTEM_PROMPT_CHUNK = """شما یک دستیار هوشمند خلاصه‌سازی هستید. یک بخش از پیام‌های روزانه یک تاپیک تلگرام به همراه شناسه پیام‌ها به شما داده می‌شود. وظیفه شما استخراج نکات کلیدی، مباحث معنادار، اسامی مهم، تصمیمات، لینک‌ها و سؤالات مطرح‌شده در این بخش است.

قوانین:
۱. متن خلاصه را موجز، دقیق و به زبان فارسی سلیس بنویسید.
۲. احوالپرسی و پیام‌های فاقد بار اطلاعاتی را کاملاً حذف کنید.
۳. برای هر نکته یا تصمیم مهم، شناسه پیام آن را حفظ کنید (یا با تگ <a href="https://t.me/c/{clean_chat_id}/ID"> پیوند دهید).
۴. از تگ‌های HTML مجاز تلگرام استفاده کنید.
۵. بین هر مبحث فاصله مناسب قرار دهید."""

SYSTEM_PROMPT_MERGE = """شما یک دستیار تحلیل‌گر ارشد هستید. به شما خلاصه‌های بخش‌های مختلف از گفتگوهای یک تاپیک تلگرام داده می‌شود. وظیفه شما ترکیب، یکپارچه‌سازی و حذف موارد تکراری برای تولید خلاصه نهایی و جامع روزانه به زبان فارسی است.

قوانین حیاتی:
۱. ساختار نهایی باید دقیقاً و الزاماً مطابق این الگو با هدینگ‌های مشخص و دو خط خالی بین هر بخش باشد:

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}


🔹 <b>محورهای اصلی بحث</b>
▫️ <b>مبحث:</b>
• نکات اصلی گفتگو...


✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• تصمیم‌ها... (در صورت عدم وجود حذف شود)


🔗 <b>لینک‌ها و منابع مهم</b>
• پیوندها... (در صورت عدم وجود حذف شود)


❓ <b>سؤالات بی‌پاسخ</b>
• سؤالات بی‌جواب... (در صورت عدم وجود حذف شود)

۲. قانون لینک آبی: هر تصمیم مهم، نکته اساسی یا راه‌حل را با تگ <a href="https://t.me/c/{clean_chat_id}/ID">متن مهم</a> پیوند دهید تا به رنگ آبی و قابل کلیک نمایش داده شود.
۳. فاصله‌گذاری: حتماً بین هر بخش دو خط خالی قرار دهید تا متن بسیار خوانا و باز باشد.
۴. فقط از تگ‌های مجاز تلگرام (<b>, <i>, <code>, <blockquote>, <a href="...">) استفاده کنید و هیچ مارک‌داونی به کار نبرید.
۵. طول خروجی ترجیحاً زیر ۳۵۰۰ کاراکتر باشد.
۶. هیچ مقدمه، مؤخره یا توضیحی فراتر از قالب ساختاریافته فوق اضافه نکنید."""

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

        text = (msg.get("text") or "").strip()
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

    # 3. Convert Markdown headings #..### to styled bold headings with double newlines
    cleaned = re.sub(r"(?m)^#\s+(.+)$", r"\n\n📌 <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^##\s+(.+)$", r"\n\n🔹 <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^###\s+(.+)$", r"\n\n▫️ <b>\1</b>\n", cleaned)
    cleaned = re.sub(r"(?m)^#{4,6}\s+(.+)$", r"\n\n▪️ <b>\1</b>\n", cleaned)

    # 4. Convert other common markdown patterns
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"<u>\1</u>", cleaned)

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
                    # Extract href attribute safely
                    href_match = re.search(r'href=[\"\']([^\"\']+)[\"\']', attributes)
                    raw_href = href_match.group(1).strip() if href_match else "#"
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

                    tokens.append(f'<a href="{html.escape(href, quote=True)}">')
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


async def _generate_with_retry(
    llm_client: BaseLLMClient,
    system_prompt: str,
    user_content: str,
    max_retries: int = 3,
) -> str:
    """Execute LLM call with retry on network/server errors, and retry-once on token truncation."""
    async def _call_once(allow_truncated: bool) -> str:
        return await llm_client.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            allow_truncated=allow_truncated,
        )

    try:
        return await async_retry(
            lambda: _call_once(allow_truncated=False),
            retries=max_retries,
        )
    except LLMTruncationError as exc:
        logger.warning(
            "LLM response was truncated (token limit reached). Retrying generation once with allow_truncated=True..."
        )
        try:
            return await async_retry(
                lambda: _call_once(allow_truncated=True),
                retries=max_retries,
            )
        except LLMTruncationError as exc_persisted:
            logger.warning(
                "LLM response truncation persisted on retry (%d chars). Returning truncated output for sanitization.",
                len(exc_persisted.truncated_text),
            )
            return exc_persisted.truncated_text


async def summarize_topic_messages(
    topic_name: str,
    messages: list[dict],
    chat_id: int | None = None,
    thread_id: int | None = None,
    min_messages: int = 5,
    is_nightly: bool = True,
    llm_client: BaseLLMClient | None = None,
) -> str | None:
    """Summarize a list of messages for a topic.

    Returns None if messages count is below threshold.
    Performs Map-Reduce chunking if transcript length exceeds CHUNK_MAX_CHARS.
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
    date_heading = format_date_heading(get_tehran_now(), is_nightly=is_nightly)

    # Prefix used in system prompts for message hyperlinks
    if target_thread and target_thread != 1:
        link_prefix = f"{clean_chat_id}/{target_thread}"
    else:
        link_prefix = clean_chat_id or "chat"

    # Check if single call or map-reduce
    if len(transcript) <= settings.CHUNK_MAX_CHARS:
        system_prompt = (
            SYSTEM_PROMPT_SINGLE.replace("{topic_name}", topic_name)
            .replace("{date_heading}", date_heading)
            .replace("{clean_chat_id}", link_prefix)
        )
        user_content = (
            f"متن گفتگوهای تاپیک «{topic_name}»:\n\n"
            f"{transcript}\n\n"
            "لطفاً خلاصه جامع و ساختاریافته را به زبان فارسی طبق دستورالعمل با دو خط فاصله بین بخش‌ها و پیوندهای آبی به پیام‌های مهم ارائه دهید."
        )

        raw_output = await _generate_with_retry(
            llm_client=llm_client,
            system_prompt=system_prompt,
            user_content=user_content,
            max_retries=settings.AI_MAX_RETRIES,
        )
        return sanitize_telegram_html(raw_output, clean_chat_id=clean_chat_id, thread_id=target_thread)

    else:
        # Map-Reduce pipeline
        chunks = chunk_transcript(transcript, max_chars=settings.CHUNK_MAX_CHARS)
        logger.info(
            "Topic '%s' transcript length %s exceeds limit %s. Running Map-Reduce over %s chunks.",
            topic_name,
            len(transcript),
            settings.CHUNK_MAX_CHARS,
            len(chunks),
        )

        chunk_prompt = SYSTEM_PROMPT_CHUNK.replace("{clean_chat_id}", link_prefix)
        partial_summaries: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_user_content = (
                f"بخش {i} از {len(chunks)} پیام‌های تاپیک «{topic_name}»:\n\n"
                f"{chunk}\n\n"
                "نکات کلیدی، تصمیمات و مباحث معنادار این بخش را به همراه شناسه‌های پیام استخراج کنید."
            )

            partial_summary = await _generate_with_retry(
                llm_client=llm_client,
                system_prompt=chunk_prompt,
                user_content=chunk_user_content,
                max_retries=settings.AI_MAX_RETRIES,
            )
            partial_summaries.append(partial_summary.strip())

        # Merge step
        merge_system_prompt = (
            SYSTEM_PROMPT_MERGE.replace("{topic_name}", topic_name)
            .replace("{date_heading}", date_heading)
            .replace("{clean_chat_id}", link_prefix)
        )
        merge_user_content = (
            f"خلاصه‌های استخراج‌شده از {len(partial_summaries)} بخش مختلف تاپیک «{topic_name}»:\n\n"
            + "\n\n---\n\n".join(partial_summaries)
            + "\n\nلطفاً این بخش‌ها را ترکیب کرده و خلاصه نهایی جامع را با دو خط فاصله و لینک‌های آبی به پیام‌ها ارائه دهید."
        )

        merged_output = await _generate_with_retry(
            llm_client=llm_client,
            system_prompt=merge_system_prompt,
            user_content=merge_user_content,
            max_retries=settings.AI_MAX_RETRIES,
        )
        return sanitize_telegram_html(merged_output, clean_chat_id=clean_chat_id, thread_id=target_thread)
