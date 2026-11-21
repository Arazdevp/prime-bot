"""Summarization service handling message formatting, map-reduce chunking, and HTML sanitization."""

from datetime import datetime
import html
import logging
import re
from zoneinfo import ZoneInfo
from bot.config import get_settings
from bot.services.llm_client import BaseLLMClient, get_llm_client
from bot.utils.dates import format_date_heading, get_tehran_now
from bot.utils.retry import async_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SINGLE = """شما یک دستیار هوشمند و تحلیل‌گر حرفه‌ای برای گروه‌های تلگرام هستید. وظیفه شما خواندن متن گفتگوهای روزانه یک تاپیک و تولید یک خلاصه دقیق، ساختاریافته، سلیس و بدون حاشیه به زبان فارسی است.

قوانین حیاتی:
۱. فقط و فقط از اطلاعات موجود در پیام‌ها استفاده کنید. هرگز از خود چیزی حدس نزنید یا اضافه نکنید.
۲. خوش‌وبش، احوالپرسی، تعارفات و پیام‌های بی‌محتوا را کاملاً نادیده بگیرید.
۳. اسامی افراد، پیوندها (لینک‌ها) و اعداد/مبالغ مهم را با نهایت دقت و بدون تحریف حفظ کنید.
۴. صرفاً از تگ‌های مجاز HTML تلگرام استفاده کنید: <b>, <i>, <code>, <blockquote>, <a href="...">. هیچ‌گونه مارک‌داون (مانند ** یا #) استفاده نکنید. کاراکترهای < و > و & در متن ساده را حتماً Escape کنید.
۵. ساختار خروجی شما باید دقیقاً به شکل زیر باشد (بخش‌های بدون محتوا را حذف کنید):

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}

🔹 <b>محورهای اصلی بحث</b>
• نکته اول به صورت موجز و روان
• نکته دوم...

✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• تصمیم‌های اتخاذ شده یا نتایج نهایی (در صورت عدم وجود این بخش را حذف کنید)

🔗 <b>لینک‌ها و منابع مهم</b>
• پیوندها به صورت <a href="...">عنوان لینک</a> یا توضیحات مختصر (در صورت عدم وجود این بخش را حذف کنید)

❓ <b>سؤالات بی‌پاسخ</b>
• سؤالات مهمی که در گفتگوها مطرح شده اما بی‌پاسخ مانده‌اند (در صورت عدم وجود این بخش را حذف کنید)

۶. طول کل خروجی معمولاً باید زیر ۳۰۰۰ کاراکتر باشد مگر اینکه حجم محتوای معنادار واقعاً زیاد باشد.
۷. هیچ‌گونه مقدمه، نتیجه‌گیری، سلام، خداحافظی یا اشاره به این‌که شما هوش مصنوعی هستید اضافه نکنید."""

SYSTEM_PROMPT_CHUNK = """شما یک دستیار هوشمند خلاصه‌سازی هستید. یک بخش از پیام‌های روزانه یک تاپیک تلگرام به شما داده می‌شود. وظیفه شما استخراج نکات کلیدی، مباحث معنادار، اسامی مهم، تصمیمات، لینک‌ها و سؤالات مطرح‌شده در این بخش است.

قوانین:
۱. متن خلاصه را موجز، دقیق و به زبان فارسی سلیس بنویسید.
۲. احوالپرسی و پیام‌های فاقد بار اطلاعاتی را کاملاً حذف کنید.
۳. از تگ‌های HTML مجاز تلگرام استفاده کنید.
۴. از حدس زدن پرهیز کنید و لینک‌ها و اسامی را دقیقاً قید نمایید.
۵. مقدمه یا توضیحی اضافه نکنید و صرفاً نکات استخراج‌شده را بنویسید."""

SYSTEM_PROMPT_MERGE = """شما یک دستیار تحلیل‌گر ارشد هستید. به شما خلاصه‌های بخش‌های مختلف از گفتگوهای یک تاپیک تلگرام داده می‌شود. وظیفه شما ترکیب، یکپارچه‌سازی و حذف موارد تکراری برای تولید خلاصه نهایی و جامع روزانه به زبان فارسی است.

قوانین حیاتی:
۱. ساختار نهایی باید دقیقاً و الزاماً مطابق این الگو باشد:

📌 <b>خلاصه تاپیک: {topic_name}</b>
{date_heading}

🔹 <b>محورهای اصلی بحث</b>
• نکات اصلی بحث...

✅ <b>تصمیم‌ها و جمع‌بندی‌ها</b>
• تصمیم‌ها... (در صورت عدم وجود حذف شود)

🔗 <b>لینک‌ها و منابع مهم</b>
• لینک‌ها... (در صورت عدم وجود حذف شود)

❓ <b>سؤالات بی‌پاسخ</b>
• سؤالات بی‌جواب... (در صورت عدم وجود حذف شود)

۲. فقط از تگ‌های HTML مجاز تلگرام (<b>, <i>, <code>, <blockquote>, <a href="...">) استفاده کنید و هیچ مارک‌داونی به کار نبرید.
۳. طول خروجی ترجیحاً زیر ۳۰۰۰ کاراکتر باشد.
۴. هیچ مقدمه، مؤخره یا توضیحی فراتر از قالب ساختاریافته فوق اضافه نکنید."""

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
    """Format raw message records into chronological Persian transcripts."""
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
        formatted_lines.append(f"[{time_str}] {sender_name}{reply_str}: {media_prefix}{text}")

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


def sanitize_telegram_html(text: str) -> str:
    """Clean model output, convert markdown, escape stray characters, and balance allowed HTML tags."""
    if not text:
        return ""

    # 1. Strip markdown code fences
    cleaned = re.sub(r"^```(?:html)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 2. Convert common markdown patterns to HTML
    cleaned = re.sub(r"#{1,6}\s+(.+)", r"<b>\1</b>", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"<u>\1</u>", cleaned)

    # 3. Escape lone ampersands not already part of valid entities
    cleaned = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", cleaned)

    # 4. Tokenize and sanitize tags
    tag_pattern = re.compile(r"<(/)?([a-zA-Z0-9_-]+)(?:\s+([^>]*))?>")
    tokens: list[str] = []
    last_idx = 0
    open_stack: list[str] = []

    for m in tag_pattern.finditer(cleaned):
        # Escape stray angle brackets in text before tag
        if m.start() > last_idx:
            segment = cleaned[last_idx : m.start()]
            segment = segment.replace("<", "&lt;").replace(">", "&gt;")
            tokens.append(segment)

        is_closing = bool(m.group(1))
        tag_name = m.group(2).lower()
        attributes = m.group(3) or ""

        if tag_name not in ALLOWED_HTML_TAGS:
            # Escape disallowed tag
            disallowed_tag = m.group(0).replace("<", "&lt;").replace(">", "&gt;")
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
                    href = href_match.group(1) if href_match else "#"
                    tokens.append(f'<a href="{html.escape(href)}">')
                else:
                    tokens.append(f"<{tag_name}>")
                open_stack.append(tag_name)

        last_idx = m.end()

    # Remainder text after last tag
    if last_idx < len(cleaned):
        segment = cleaned[last_idx:]
        segment = segment.replace("<", "&lt;").replace(">", "&gt;")
        tokens.append(segment)

    # Close any unclosed tags at the end
    while open_stack:
        tokens.append(f"</{open_stack.pop()}>")

    return "".join(tokens).strip()


async def summarize_topic_messages(
    topic_name: str,
    messages: list[dict],
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

    transcript = format_messages(messages)
    date_heading = format_date_heading(get_tehran_now(), is_nightly=is_nightly)

    # Check if single call or map-reduce
    if len(transcript) <= settings.CHUNK_MAX_CHARS:
        system_prompt = SYSTEM_PROMPT_SINGLE.replace("{topic_name}", topic_name).replace(
            "{date_heading}", date_heading
        )
        user_content = (
            f"متن گفتگوهای تاپیک «{topic_name}»:\n\n"
            f"{transcript}\n\n"
            "لطفاً خلاصه جامع و ساختاریافته را به زبان فارسی طبق دستورالعمل ارائه دهید."
        )

        async def _call_single() -> str:
            return await llm_client.generate(system_prompt=system_prompt, user_content=user_content)

        raw_output = await async_retry(_call_single, retries=settings.AI_MAX_RETRIES)
        return sanitize_telegram_html(raw_output)

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

        partial_summaries: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_user_content = (
                f"بخش {i} از {len(chunks)} پیام‌های تاپیک «{topic_name}»:\n\n"
                f"{chunk}\n\n"
                "نکات کلیدی و مباحث معنادار این بخش را استخراج کنید."
            )

            async def _call_chunk(c_content: str = chunk_user_content) -> str:
                return await llm_client.generate(
                    system_prompt=SYSTEM_PROMPT_CHUNK,
                    user_content=c_content,
                )

            partial_summary = await async_retry(_call_chunk, retries=settings.AI_MAX_RETRIES)
            partial_summaries.append(partial_summary.strip())

        # Merge step
        merge_system_prompt = SYSTEM_PROMPT_MERGE.replace("{topic_name}", topic_name).replace(
            "{date_heading}", date_heading
        )
        merge_user_content = (
            f"خلاصه‌های استخراج‌شده از {len(partial_summaries)} بخش مختلف تاپیک «{topic_name}»:\n\n"
            + "\n\n---\n\n".join(partial_summaries)
            + "\n\nلطفاً این بخش‌ها را ترکیب کرده و خلاصه نهایی جامع را ارائه دهید."
        )

        async def _call_merge() -> str:
            return await llm_client.generate(
                system_prompt=merge_system_prompt,
                user_content=merge_user_content,
            )

        merged_output = await async_retry(_call_merge, retries=settings.AI_MAX_RETRIES)
        return sanitize_telegram_html(merged_output)
