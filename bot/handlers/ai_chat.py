"""Interactive AI chat handlers and /manage_ai owner-only control panel."""

import json
import logging
import re
import time
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, Filter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup, default_state
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from bot.config import (
    format_ai_provider_label,
    get_settings,
    is_9router_base_url,
    is_openai_compatible_provider,
    resolve_openai_compat_config,
)
from bot.database import repository
from bot.handlers.panel import safe_answer_callback, safe_edit_message
from bot.keyboards.ai_chat import (
    AiChatCB,
    build_ai_media_done_kb,
    build_ai_media_mood_kb,
    build_ai_media_panel_kb,
    build_ai_memory_view_kb,
    build_ai_prompt_cancel_kb,
    build_ai_prompt_view_kb,
    build_ai_quota_cancel_kb,
    build_ai_quota_kb,
    build_manage_ai_kb,
)
from bot.services.access import is_global_admin
from bot.services.ai_chat_service import (
    check_and_consume_user_quota,
    clear_conversation_history,
    count_active_conversations,
    count_active_quota_users,
    ensure_ai_bot_identity,
    generate_ai_chat_reply,
    get_effective_ai_chat_access_mode,
    get_effective_ai_chat_provider,
    get_effective_ai_chat_user_limit,
    is_ai_chat_enabled_cached,
    is_message_addressing_bot,
    refund_user_quota,
    reset_all_user_quotas,
    set_ai_chat_access_mode_state,
    set_ai_chat_provider_state,
    set_ai_chat_state,
    set_ai_chat_user_limit_state,
    should_send_quota_notice,
    strip_bot_mention,
)
from bot.services.ai_media_reaction_service import (
    classify_emoji_sentiment,
    classify_zero_token_reaction,
    detect_explicit_media_request,
    extract_ai_chat_media,
    get_ai_chat_media_counts_cached,
    infer_mood_from_emoji,
    invalidate_ai_media_counts_cache,
    is_ai_chat_media_autocollect_cached,
    maybe_capture_ai_chat_media,
    mood_to_reaction,
    parse_ai_chat_directives,
    send_ai_chat_media_reply,
    send_ai_reaction,
    set_ai_chat_media_autocollect_state,
    should_reply_with_media_to_media,
)
from bot.services.ai_memory_service import (
    bind_memory_source_topic,
    ensure_memory_source_for_group,
    is_allowed_ai_chat_topic,
    learn_from_topic_messages,
    resolve_memory_group,
)
from bot.services.llm_client import LLMError
from bot.services.publisher import strip_html_tags
from bot.texts.ai_chat_texts import DEFAULT_AI_CHAT_SYSTEM_PROMPT
from bot.utils.general_topic import resolve_thread_id, to_send_thread_id
from bot.utils.html_utils import escape_html
from bot.utils.text_split import TELEGRAM_MESSAGE_MAX_CHARS, smart_split

logger = logging.getLogger(__name__)

router = Router(name="ai_chat_router")

FSM_PROMPT_TTL_SECONDS = 180.0
MEDIA_FSM_TTL_SECONDS = 300.0
AI_MEMORY_PAGE_SIZE = 5
_DIGIT_TRANS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


class AiChatStates(StatesGroup):
    """FSM states for entering custom system prompt, quota limit, or chat media in /manage_ai."""

    waiting_custom_prompt = State()
    waiting_quota_limit = State()
    waiting_chat_media = State()


def _parse_quota_value(raw: str | None) -> int | None:
    """Parse regular-user quota limit ('0'/'off'->0, '-1'/'unlimited'->-1, '1..10000'->int)."""
    if not raw:
        return None
    cleaned = raw.translate(_DIGIT_TRANS).strip().lower()
    if not cleaned:
        return None
    if cleaned in ("0", "off", "close", "closed", "disable", "lock", "locked", "بسته", "بستن", "قفل", "خاموش"):
        return 0
    if cleaned in ("-1", "unlimited", "unlim", "inf", "open", "نامحدود", "باز", "بدون محدودیت"):
        return -1
    if cleaned.isdigit():
        val = int(cleaned)
        if 1 <= val <= repository.AI_CHAT_USER_LIMIT_MAX:
            return val
    return None


def _format_quota_summary(limit_count: int, window_hours: int) -> str:
    """Format human-readable Persian summary of the active regular-user quota."""
    if limit_count < 0:
        return "♾️ نامحدود"
    if limit_count == 0:
        return "🔒 سهمیه بسته (۰ پیام)"
    return f"{limit_count} پیام در {window_hours} ساعت"


def _clip_text(value: str | None, max_len: int) -> str:
    """Normalize whitespace and trim text to max_len characters."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _parse_manage_arg(raw_text: str | None) -> tuple[bool | None, str | None, str | None, str]:
    """Parse optional shorthand arguments from '/manage_ai ...'."""
    if not raw_text:
        return None, None, None, ""
    parts = raw_text.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, None, None, ""
    arg_full = parts[1].strip()
    tokens = arg_full.split(maxsplit=1)
    cmd = tokens[0].strip().lower()
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    if cmd in ("on", "enable", "enabled", "1", "true", "روشن", "فعال") and not rest:
        return True, None, None, ""
    if cmd in ("off", "disable", "disabled", "0", "false", "خاموش", "غیرفعال") and not rest:
        return False, None, None, ""
    if cmd in ("9router", "ninerouter", "9-router"):
        return None, "9router", None, ""
    if cmd in ("openai", "openrouter", "gateway"):
        return None, "openai", None, ""
    if cmd in ("gemini", "google", "aistudio"):
        return None, "gemini", None, ""
    if cmd in ("bind", "اتصال"):
        return None, None, "bind", ""
    if cmd in ("learn", "یادگیری"):
        return None, None, "learn", ""
    if cmd in ("memory", "mem", "حافظه"):
        page_idx = 0
        if rest:
            norm_page = rest.translate(_DIGIT_TRANS).strip()
            if norm_page.isdigit():
                page_idx = max(0, int(norm_page) - 1)
        return None, None, "memory", str(page_idx)
    if cmd in ("quota", "limit", "سهمیه"):
        if not rest:
            return None, None, "quota", ""
        if rest.lower() in ("reset", "clear", "ریست", "پاک"):
            return None, None, "quota", "reset"
        return None, None, "quota", rest
    if cmd in ("media", "رسانه", "استیکر", "گیف"):
        return None, None, "media", ""
    return None, None, None, ""


async def _send_in_context(
    message: Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    bot: Bot | None = None,
) -> Message | None:
    """Send an HTML reply in private chat or the exact supergroup topic."""
    if not message.chat:
        return None
    if message.chat.type == "private":
        return await message.answer(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    active_bot = bot or message.bot
    if not active_bot:
        return None
    target_thread = to_send_thread_id(resolve_thread_id(message))
    return await active_bot.send_message(
        chat_id=message.chat.id,
        message_thread_id=target_thread,
        text=text,
        reply_markup=reply_markup,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


_SECRET_REDACT_RE = re.compile(r"(?:AQ\.[A-Za-z0-9._-]+|AIza[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+)")


def _pick_preferred_gateway_provider(settings: object) -> str:
    """Return '9router' when 9Router is configured or active, otherwise 'openai'."""
    if (
        str(getattr(settings, "AI_CHAT_PROVIDER", "") or "").strip().lower() == "9router"
        or str(getattr(settings, "AI_PROVIDER", "") or "").strip().lower() == "9router"
        or getattr(settings, "NINEROUTER_API_KEY", None)
        or getattr(settings, "AI_CHAT_NINEROUTER_API_KEY", None)
    ):
        return "9router"
    base_url, _, _, _ = resolve_openai_compat_config(settings, for_chat=True, provider="openai")
    if is_9router_base_url(base_url):
        return "9router"
    return "openai"


def _format_key_status(settings: object, provider: str) -> str:
    """Format masked API key source label (last 4 chars only, never exposing secret)."""
    if is_openai_compatible_provider(provider):
        base_url, api_key, _model, key_src = resolve_openai_compat_config(
            settings, for_chat=True, provider=provider
        )
        if base_url and api_key:
            suffix = api_key[-4:] if len(api_key) >= 4 else "****"
            return f"✅ متصل (<code>{key_src}: ...{escape_html(suffix)}</code>)"
        label = "9Router" if provider == "9router" else "OpenAI"
        return f"❌ تنظیمات {label} کامل نیست"

    chat_key = (getattr(settings, "AI_CHAT_API_KEY", None) or "").strip()
    if chat_key:
        suffix = chat_key[-4:] if len(chat_key) >= 4 else "****"
        return f"✅ متصل (<code>AI_CHAT_API_KEY: ...{escape_html(suffix)}</code>)"

    gem_key = (getattr(settings, "GEMINI_API_KEY", None) or "").strip()
    if gem_key:
        suffix = gem_key[-4:] if len(gem_key) >= 4 else "****"
        return f"⚠️ از کلید پیش‌فرض (<code>GEMINI_API_KEY: ...{escape_html(suffix)}</code>)"

    return "❌ کلید API تنظیم نشده"


def _format_safe_llm_error(exc: LLMError) -> str:
    """Extract concise, secret-redacted diagnostic summary for the bot owner."""
    detail = ""
    for candidate in (getattr(exc, "body_excerpt", None), getattr(exc, "message", None)):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        raw = candidate.strip()
        json_start = raw.find("{")
        if json_start != -1:
            try:
                parsed = json.loads(raw[json_start:])
                if isinstance(parsed, dict):
                    err = parsed.get("error")
                    if isinstance(err, dict) and isinstance(err.get("message"), str):
                        detail = err["message"].strip()
                        break
            except Exception:
                pass
        if not detail:
            detail = raw

    detail = _SECRET_REDACT_RE.sub("***", detail)
    detail = re.sub(r"\s+", " ", detail).strip()
    prefix = f"[HTTP {exc.status_code}] " if exc.status_code is not None else "[Network] "
    combined = f"{prefix}{detail}" if detail else f"{prefix}Unknown error"
    return combined[:240]


async def _render_memory_overview(
    owner_id: int,
    chat_id: int | None = None,
    chat_type: str = "private",
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render paginated learned group culture and full member personalities overview."""
    mem_chat_id, group_title = await resolve_memory_group(chat_id, chat_type)
    if mem_chat_id is None:
        return (
            "👥 <b>حافظه بلندمدت اعضا</b>\n\nهنوز گروه فعالی برای یادگیری پیدا نشد.",
            build_ai_memory_view_kb(owner_id=owner_id, has_memory=False),
        )

    source = await ensure_memory_source_for_group(mem_chat_id)
    group_mem = await repository.get_ai_group_memory(mem_chat_id)
    total_profiles = await repository.count_ai_member_profiles(mem_chat_id)
    total_pages = max(1, (total_profiles + AI_MEMORY_PAGE_SIZE - 1) // AI_MEMORY_PAGE_SIZE)
    safe_page = max(0, min(int(page), total_pages - 1))
    profiles = await repository.list_ai_member_profiles_page(
        mem_chat_id,
        limit=AI_MEMORY_PAGE_SIZE,
        offset=safe_page * AI_MEMORY_PAGE_SIZE,
    )

    def _build_text(
        summary_cap: int,
        style_cap: int,
        interests_cap: int,
        facts_cap: int,
    ) -> str:
        group_label = escape_html(group_title or str(mem_chat_id))
        if safe_page == 0:
            lines = [f"👥 <b>حافظه بلندمدت گروه {group_label}</b>\n"]
            if source:
                src_name = source.get("topic_name") or f"تاپیک {source['thread_id']}"
                lines.append(f"📚 تاپیک یادگیری: <b>{escape_html(src_name)}</b>")
            else:
                lines.append("📚 تاپیک یادگیری: <i>تنظیم نشده (/manage_ai bind داخل گفتگوی آزاد)</i>")

            if group_mem and (
                group_mem.get("culture_summary")
                or group_mem.get("inside_jokes")
                or group_mem.get("ongoing_topics")
            ):
                lines.append("\n🌐 <b>شناخت از فضای گروه:</b>")
                if group_mem.get("culture_summary"):
                    lines.append(f"• حال‌وهوا: {escape_html(_clip_text(group_mem['culture_summary'], 220))}")
                if group_mem.get("inside_jokes"):
                    lines.append(f"• شوخی‌ها و اصطلاحات: {escape_html(_clip_text(group_mem['inside_jokes'], 220))}")
                if group_mem.get("ongoing_topics"):
                    lines.append(f"• بحث‌های فعلی: {escape_html(_clip_text(group_mem['ongoing_topics'], 220))}")
        else:
            lines = [f"👥 <b>حافظه بلندمدت گروه {group_label}</b>"]

        if profiles:
            start_idx = safe_page * AI_MEMORY_PAGE_SIZE + 1
            end_idx = start_idx + len(profiles) - 1
            lines.append(
                f"\n🧑 <b>اعضای شناخته‌شده ({total_profiles} نفر — نمایش {start_idx} تا {end_idx} | صفحه {safe_page + 1}/{total_pages}):</b>"
            )
            for idx_offset, prof in enumerate(profiles):
                num = start_idx + idx_offset
                name = escape_html(_clip_text(prof.get("display_name") or "عضو", 50))
                msg_count = int(prof.get("message_count") or 0)
                lines.append(f"\n{num}. <b>{name}</b> ({msg_count} پیام)")

                summary = _clip_text(prof.get("personality_summary"), summary_cap)
                style = _clip_text(prof.get("speaking_style"), style_cap)
                interests = _clip_text(prof.get("interests_and_topics"), interests_cap)
                facts = _clip_text(prof.get("notable_facts"), facts_cap)

                if summary:
                    lines.append(f"   • شخصیت: {escape_html(summary)}")
                if style:
                    lines.append(f"   • لحن: {escape_html(style)}")
                if interests:
                    lines.append(f"   • علایق: {escape_html(interests)}")
                if facts:
                    lines.append(f"   • نکات: {escape_html(facts)}")
        else:
            lines.append("\n<i>هنوز پروفایلی یادگیری نشده؛ دکمه «🧠 یادگیری الان» رو بزن.</i>")

        return "\n".join(lines)

    rendered = _build_text(260, 120, 140, 140)
    if len(rendered) > 3950:
        rendered = _build_text(140, 70, 80, 80)
    if len(rendered) > 4000:
        rendered = rendered[:3990].rstrip() + "…"

    has_memory = bool(total_profiles > 0 or group_mem)
    return rendered, build_ai_memory_view_kb(
        owner_id=owner_id,
        has_memory=has_memory,
        page=safe_page,
        total_pages=total_pages,
    )


async def _render_quota_panel(owner_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render regular-user AI chat quota management view and inline controls."""
    settings = get_settings()
    access_mode = get_effective_ai_chat_access_mode()
    limit_count = get_effective_ai_chat_user_limit()
    window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
    active_users = count_active_quota_users()

    access_status = (
        "👥 <b>باز برای همه اعضای گفتگوی آزاد</b>"
        if access_mode == "public"
        else "🔒 <b>محدود به مدیر ربات</b> (فعلاً کاربران معمولی دسترسی ندارند)"
    )
    quota_status = _format_quota_summary(limit_count, window_hours)

    text = (
        "📊 <b>مدیریت سهمیه چت کاربران معمولی</b>\n\n"
        f"• سطح دسترسی گروه: {access_status}\n"
        f"• سهمیه فعلی کاربران معمولی: <b>{quota_status}</b>\n"
        f"• کاربران دارای مصرف فعال در پنجره اخیر: <b>{active_users}</b> نفر\n"
        "• سهمیه مدیران ربات: ♾️ <b>همیشه نامحدود</b>\n\n"
        "می‌تونی عدد سهمیه رو روی ۱۰/۲۰/۵۰/۱۰۰ یا هر عدد دلخواه بذاری، سهمیه همه رو الان ریست کنی، "
        "سهمیه رو ببندی (۰)، یا نامحدودش کنی.\n"
        "• میانبر سریع: <code>/manage_ai quota 50</code> | <code>off</code> | <code>unlimited</code> | <code>reset</code>"
    )
    return text, build_ai_quota_kb(owner_id=owner_id, current_limit=limit_count)


async def _render_media_panel(owner_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render AI chat media bank panel (stickers, GIFs, video notes, and auto-collect)."""
    counts = await repository.count_ai_chat_media_by_type()
    sources = await repository.count_ai_chat_media_by_source()
    autocollect = is_ai_chat_media_autocollect_cached()
    auto_badge = "🟢 <b>روشن</b>" if autocollect else "🔴 <b>خاموش</b>"

    text = (
        "🎭 <b>مدیریت رسانه و واکنش چت هوش مصنوعی</b>\n\n"
        f"• استیکر: <b>{counts.get('sticker', 0)}</b> عدد\n"
        f"• گیف (GIF): <b>{counts.get('animation', 0)}</b> عدد\n"
        f"• ویدیومسیج (ویدیو گرد): <b>{counts.get('video_note', 0)}</b> عدد\n"
        f"• منبع ذخیره: 👤 دستی <b>{sources.get('manual', 0)}</b> | 🤖 خودکار <b>{sources.get('auto', 0)}</b>\n"
        f"• جمع‌آوری خودکار از گفتگوی آزاد: {auto_badge}\n\n"
        "<b>نحوه عملکرد:</b>\n"
        "• پیام‌های ایموجی یا کوتاهِ بی‌نیاز به جواب (مثل «آفرین»، «دقیقاً»، «مرسی»، «اوکی») "
        "بدون مصرف هیچ توکنی فقط با ریکشن (👍 / 👎 / 😁) جواب می‌گیرن.\n"
        "• ربات توی جواب‌هاش به‌تناسب فضا از استیکر، گیف و ویدیومسیج‌های این بانک هم استفاده می‌کنه.\n"
        "• <i>ویدیومسیج فقط به‌صورت دستی از همینجا اضافه می‌شه (برای حفظ حریم خصوصی اعضا).</i>"
    )
    return text, build_ai_media_panel_kb(owner_id=owner_id, autocollect=autocollect, counts=counts)


async def _run_manual_memory_learn(
    chat_id: int | None,
    chat_type: str,
) -> tuple[bool, str]:
    """Trigger immediate learning from the resolved group's free-chat topic."""
    mem_chat_id, _ = await resolve_memory_group(chat_id, chat_type)
    if mem_chat_id is None:
        return False, "گروه فعالی برای یادگیری پیدا نشد."

    source = await ensure_memory_source_for_group(mem_chat_id)
    if source is None:
        return False, "تاپیک گفتگوی آزاد پیدا نشد؛ داخل همون تاپیک دستور /manage_ai bind رو بزن."

    result = await learn_from_topic_messages(
        mem_chat_id,
        int(source["thread_id"]),
        topic_name=source.get("topic_name"),
    )
    processed = int(result.get("messages_processed") or 0)
    members_cnt = int(result.get("members_updated") or 0)
    topic_label = result.get("topic_name") or "گفتگوی آزاد"
    if processed == 0:
        return True, f"پیام جدیدی توی «{topic_label}» برای یادگیری نبود (حافظه به‌روزه)."
    return True, f"🧠 {processed} پیام از «{topic_label}» بررسی شد و شناخت {members_cnt} عضو به‌روزرسانی شد."


async def render_manage_ai_panel(
    owner_id: int,
    bot_username: str | None = None,
    chat_id: int | None = None,
    chat_type: str = "private",
) -> tuple[str, InlineKeyboardMarkup]:
    """Render status summary text and inline keyboard for /manage_ai."""
    settings = get_settings()
    enabled = is_ai_chat_enabled_cached()
    provider = get_effective_ai_chat_provider()
    access_mode = get_effective_ai_chat_access_mode()
    custom_prompt = await repository.get_ai_chat_custom_prompt()
    conv_count = count_active_conversations()

    mem_chat_id, _ = await resolve_memory_group(chat_id, chat_type)
    source_label = "پیدا نشد (داخل گفتگوی آزاد <code>/manage_ai bind</code> بزن)"
    learned_count = 0
    if mem_chat_id is not None:
        source = await ensure_memory_source_for_group(mem_chat_id)
        if source:
            src_name = source.get("topic_name") or f"تاپیک {source['thread_id']}"
            source_label = f"<b>{escape_html(src_name)}</b>"
        learned_count = await repository.count_ai_member_profiles(mem_chat_id)

    limit_count = get_effective_ai_chat_user_limit()
    window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
    status_badge = "🟢 <b>روشن</b>" if enabled else "🔴 <b>خاموش</b>"
    provider_label = format_ai_provider_label(settings, for_chat=True, provider=provider)
    quota_summary = _format_quota_summary(limit_count, window_hours)
    access_badge = (
        f"👥 <b>همه اعضای گفتگوی آزاد</b> ({quota_summary} | مدیر نامحدود)"
        if access_mode == "public"
        else "🔒 <b>فقط مدیر ربات</b>"
    )
    if is_openai_compatible_provider(provider):
        _, _, resolved_model, _ = resolve_openai_compat_config(settings, for_chat=True, provider=provider)
        active_model = resolved_model or "تنظیم‌نشده"
    else:
        active_model = settings.AI_CHAT_MODEL
    key_badge = _format_key_status(settings, provider)
    prompt_badge = "✏️ سفارشی" if custom_prompt else "🇮🇷 پیش‌فرض (لحن طبیعی ایرانی)"
    try:
        media_counts = await get_ai_chat_media_counts_cached()
    except Exception:
        media_counts = {"sticker": 0, "animation": 0, "video_note": 0}
    mention_hint = f"<code>@{escape_html(bot_username)}</code>" if bot_username else "منشن ربات"
    preferred_gateway = _pick_preferred_gateway_provider(settings)
    gateway_label = "9Router" if preferred_gateway == "9router" else "OpenAI"

    text = (
        "🧠 <b>مدیریت چت هوش مصنوعی</b>\n\n"
        f"• وضعیت فعلی: {status_badge}\n"
        f"• سرویس فعال: 🔌 <b>{escape_html(provider_label)}</b>\n"
        f"• سطح دسترسی: {access_badge}\n"
        "• محدوده گروه: 📍 <b>فقط تاپیک گفتگوی آزاد</b>\n"
        f"• مدل گفتگو: <code>{escape_html(active_model)}</code>\n"
        f"• کلید API: {key_badge}\n"
        f"• پرامپت فعال: {prompt_badge}\n"
        f"• تاپیک یادگیری: 📚 {source_label}\n"
        f"• بانک رسانه: 🎭 استیکر <b>{media_counts.get('sticker', 0)}</b> | گیف <b>{media_counts.get('animation', 0)}</b> | ویدیومسیج <b>{media_counts.get('video_note', 0)}</b>\n"
        f"• حافظه بلندمدت: 👥 <b>{learned_count}</b> عضو شناخته‌شده | مکالمه‌های موقت: <b>{conv_count}</b>\n\n"
        "<b>نحوه استفاده هنگام روشن بودن:</b>\n"
        f"• فقط در گفتگوی آزاد: روی پیام ربات <b>ریپلای</b> بزن یا با {mention_hint} صداش کن\n"
        "• در پیوی: فقط مدیر ربات\n"
        "• میانبر سریع: <code>/manage_ai on</code> | <code>off</code> | <code>9router</code> | <code>gemini</code> | <code>quota</code> | <code>media</code> | <code>bind</code> | <code>learn</code> | <code>memory</code>"
    )
    kb = build_manage_ai_kb(
        enabled=enabled,
        has_custom_prompt=bool(custom_prompt),
        owner_id=owner_id,
        provider=provider,
        access_mode=access_mode,
        gateway_label=gateway_label,
        quota_limit=limit_count,
        window_hours=window_hours,
    )
    return text, kb


@router.message(Command("manage_ai"))
async def cmd_manage_ai(message: Message, bot: Bot, state: FSMContext) -> None:
    """Owner-only command to open the AI chat control panel or run /manage_ai subcommands."""
    user = message.from_user
    if user is None or not is_global_admin(user.id):
        if message.chat and message.chat.type != "private":
            try:
                await message.delete()
            except Exception:
                pass
        else:
            await message.answer("⛔ این بخش فقط مخصوص مدیر ربات است.")
        return

    if message.chat and message.chat.type != "private":
        try:
            await message.delete()
        except Exception:
            pass

    await state.clear()

    chat_id = message.chat.id if message.chat else None
    chat_type = message.chat.type if message.chat else "private"
    desired_toggle, desired_provider, sub_action, sub_arg = _parse_manage_arg(message.text)

    if desired_toggle is not None:
        await set_ai_chat_state(desired_toggle)
    if desired_provider is not None:
        await set_ai_chat_provider_state(desired_provider)

    notice = ""
    if sub_action == "bind":
        if chat_type == "private" or chat_id is None:
            notice = "⚠️ دستور <code>/manage_ai bind</code> رو باید داخل خودِ تاپیک موردنظر (مثلاً گفتگوی آزاد) بزنی.\n\n"
        else:
            tid = resolve_thread_id(message)
            src = await bind_memory_source_topic(chat_id, tid)
            notice = f"✅ تاپیک <b>«{escape_html(src.get('topic_name') or str(tid))}»</b> برای یادگیری شخصیت اعضا متصل شد.\n\n"
    elif sub_action == "learn":
        try:
            ok_learn, learn_msg = await _run_manual_memory_learn(chat_id, chat_type)
            prefix = "✅" if ok_learn else "⚠️"
            notice = f"{prefix} {escape_html(learn_msg)}\n\n"
        except LLMError as exc:
            notice = f"⚠️ خطا در یادگیری از پیام‌ها: <code>{escape_html(_format_safe_llm_error(exc))}</code>\n\n"
    elif sub_action == "memory":
        page_idx = int(sub_arg) if sub_arg.isdigit() else 0
        mem_text, mem_kb = await _render_memory_overview(
            owner_id=user.id, chat_id=chat_id, chat_type=chat_type, page=page_idx
        )
        await _send_in_context(message, mem_text, reply_markup=mem_kb, bot=bot)
        return
    elif sub_action == "quota":
        q_notice = ""
        if sub_arg == "reset":
            cleared = reset_all_user_quotas()
            q_notice = (
                f"♻️ سهمیه همه کاربران ریست شد ({cleared} کاربر فعال).\n\n"
                if cleared > 0
                else "♻️ سهمیه همه کاربران ریست شد.\n\n"
            )
        elif sub_arg:
            parsed_limit = _parse_quota_value(sub_arg)
            if parsed_limit is None:
                q_notice = "⚠️ مقدار سهمیه نامعتبره. از <code>0</code> (بستن)، <code>unlimited</code> (نامحدود) یا عدد ۱ تا ۱۰۰۰۰ استفاده کن.\n\n"
            else:
                effective = await set_ai_chat_user_limit_state(parsed_limit)
                window_hours = max(1, int(getattr(get_settings(), "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
                q_notice = f"✅ سهمیه کاربران معمولی روی <b>{_format_quota_summary(effective, window_hours)}</b> تنظیم شد.\n\n"
        q_text, q_kb = await _render_quota_panel(owner_id=user.id)
        await _send_in_context(message, f"{q_notice}{q_text}", reply_markup=q_kb, bot=bot)
        return
    elif sub_action == "media":
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await _send_in_context(message, m_text, reply_markup=m_kb, bot=bot)
        return

    _, bot_username = await ensure_ai_bot_identity(bot)
    text, kb = await render_manage_ai_panel(
        owner_id=user.id,
        bot_username=bot_username,
        chat_id=chat_id,
        chat_type=chat_type,
    )
    await _send_in_context(message, f"{notice}{text}", reply_markup=kb, bot=bot)


@router.callback_query(AiChatCB.filter())
async def on_manage_ai_callback(
    callback: CallbackQuery,
    callback_data: AiChatCB,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Handle inline button callbacks for /manage_ai (strictly owner-only)."""
    user = callback.from_user
    if user is None or not is_global_admin(user.id):
        await safe_answer_callback(callback, "⛔ این پنل فقط مخصوص مدیر ربات است.", show_alert=True)
        return

    action = callback_data.action
    _, bot_username = await ensure_ai_bot_identity(bot)
    cb_chat = getattr(callback.message, "chat", None) if callback.message else None
    cb_chat_id = getattr(cb_chat, "id", None)
    cb_chat_type = getattr(cb_chat, "type", None) or (
        "supergroup" if isinstance(cb_chat_id, int) and cb_chat_id < 0 else "private"
    )

    if action == "toggle":
        new_state = await set_ai_chat_state(not is_ai_chat_enabled_cached())
        toast = "🟢 چت هوش مصنوعی روشن شد!" if new_state else "🔴 چت هوش مصنوعی خاموش شد."
        await safe_answer_callback(callback, toast)
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "prov":
        current_provider = get_effective_ai_chat_provider()
        settings = get_settings()
        next_provider = (
            "gemini"
            if is_openai_compatible_provider(current_provider)
            else _pick_preferred_gateway_provider(settings)
        )
        await set_ai_chat_provider_state(next_provider)
        next_label = format_ai_provider_label(settings, for_chat=True, provider=next_provider)
        toast = f"🔌 سرویس گفتگو به {next_label} تغییر کرد."
        await safe_answer_callback(callback, toast)
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "acc":
        current_mode = get_effective_ai_chat_access_mode()
        next_mode = "owner_only" if current_mode == "public" else "public"
        await set_ai_chat_access_mode_state(next_mode)
        settings = get_settings()
        window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
        quota_summary = _format_quota_summary(get_effective_ai_chat_user_limit(), window_hours)
        toast = (
            f"👥 چت برای همه اعضای گفتگوی آزاد باز شد ({quota_summary})."
            if next_mode == "public"
            else "🔒 دسترسی چت به فقط مدیر ربات محدود شد."
        )
        await safe_answer_callback(callback, toast)
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "quota":
        await state.clear()
        await safe_answer_callback(callback)
        q_text, q_kb = await _render_quota_panel(owner_id=user.id)
        await safe_edit_message(callback, q_text, q_kb)
        return

    if action == "qset":
        parsed_limit = _parse_quota_value(callback_data.arg)
        if parsed_limit is None:
            await safe_answer_callback(callback, "⚠️ مقدار سهمیه نامعتبر است.", show_alert=True)
            return
        effective = await set_ai_chat_user_limit_state(parsed_limit)
        window_hours = max(1, int(getattr(get_settings(), "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
        await safe_answer_callback(
            callback,
            f"✅ سهمیه کاربران: {_format_quota_summary(effective, window_hours)}",
        )
        q_text, q_kb = await _render_quota_panel(owner_id=user.id)
        await safe_edit_message(callback, q_text, q_kb)
        return

    if action == "qreset":
        cleared = reset_all_user_quotas()
        toast = (
            f"♻️ سهمیه همه کاربران ریست شد ({cleared} کاربر فعال)."
            if cleared > 0
            else "♻️ سهمیه همه کاربران ریست شد."
        )
        await safe_answer_callback(callback, toast)
        q_text, q_kb = await _render_quota_panel(owner_id=user.id)
        await safe_edit_message(callback, q_text, q_kb)
        return

    if action == "qcustom":
        await safe_answer_callback(callback)
        await state.set_state(AiChatStates.waiting_quota_limit)
        await state.update_data(
            fsm_started_at=time.monotonic(),
            panel_chat_id=callback.message.chat.id if callback.message else None,
            panel_message_id=callback.message.message_id if callback.message else None,
        )
        prompt_text = (
            "✏️ <b>تنظیم سهمیه دلخواه کاربران معمولی</b>\n\n"
            "عدد مورد نظرت رو بفرست (بین ۱ تا ۱۰۰۰۰):\n"
            "• برای بستن سهمیه کاربران معمولی: <code>0</code>\n"
            "• برای نامحدود کردن: <code>-1</code> یا <code>نامحدود</code>\n"
            "<i>برای لغو دکمه زیر یا دستور /cancel رو بزن.</i>"
        )
        await safe_edit_message(callback, prompt_text, build_ai_quota_cancel_kb(owner_id=user.id))
        return

    if action == "qcancel":
        await state.clear()
        await safe_answer_callback(callback, "لغو شد.")
        q_text, q_kb = await _render_quota_panel(owner_id=user.id)
        await safe_edit_message(callback, q_text, q_kb)
        return

    if action == "clear":
        cleared = clear_conversation_history()
        await safe_answer_callback(callback, f"🧹 حافظه {cleared} مکالمه پاک شد.")
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "learn":
        try:
            ok_learn, learn_msg = await _run_manual_memory_learn(cb_chat_id, cb_chat_type)
            await safe_answer_callback(callback, learn_msg[:180], show_alert=not ok_learn)
        except LLMError as exc:
            await safe_answer_callback(
                callback,
                f"⚠️ خطای یادگیری: {_format_safe_llm_error(exc)[:150]}",
                show_alert=True,
            )
            return
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "mem":
        await safe_answer_callback(callback)
        mem_text, mem_kb = await _render_memory_overview(
            owner_id=user.id,
            chat_id=cb_chat_id,
            chat_type=cb_chat_type,
            page=callback_data.page,
        )
        await safe_edit_message(callback, mem_text, mem_kb)
        return

    if action == "noop":
        await safe_answer_callback(callback)
        return

    if action == "clrmem":
        mem_chat_id, _ = await resolve_memory_group(cb_chat_id, cb_chat_type)
        if mem_chat_id is not None:
            deleted_members, _ = await repository.clear_ai_chat_long_term_memory(mem_chat_id)
            await safe_answer_callback(callback, f"🗑 حافظه بلندمدت ({deleted_members} عضو) پاک شد.")
        else:
            await safe_answer_callback(callback, "حافظه‌ای برای پاک کردن پیدا نشد.")
        mem_text, mem_kb = await _render_memory_overview(
            owner_id=user.id, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, mem_text, mem_kb)
        return

    if action == "refresh":
        await state.clear()
        await safe_answer_callback(callback)
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "viewp":
        await safe_answer_callback(callback)
        custom_prompt = await repository.get_ai_chat_custom_prompt()
        active_prompt = custom_prompt or DEFAULT_AI_CHAT_SYSTEM_PROMPT
        kind_label = "سفارشی" if custom_prompt else "پیش‌فرض (ایرانی طبیعی)"
        preview = active_prompt if len(active_prompt) <= 3200 else active_prompt[:3200] + "…"
        text = (
            f"📝 <b>پرامپت فعال فعلی ({kind_label}):</b>\n\n"
            f"<blockquote>{escape_html(preview)}</blockquote>"
        )
        await safe_edit_message(callback, text, build_ai_prompt_view_kb(owner_id=user.id))
        return

    if action == "setp":
        await safe_answer_callback(callback)
        await state.set_state(AiChatStates.waiting_custom_prompt)
        await state.update_data(
            fsm_started_at=time.monotonic(),
            panel_chat_id=callback.message.chat.id if callback.message else None,
            panel_message_id=callback.message.message_id if callback.message else None,
        )
        prompt_text = (
            "✏️ <b>تنظیم پرامپت سفارشی هوش مصنوعی</b>\n\n"
            "متن پرامپت جدیدت رو همینجا بفرست (حداکثر ۴۰۰۰ کاراکتر).\n"
            "<i>برای لغو دکمه زیر یا دستور /cancel رو بزن.</i>"
        )
        await safe_edit_message(callback, prompt_text, build_ai_prompt_cancel_kb(owner_id=user.id))
        return

    if action == "resetp":
        await repository.set_ai_chat_custom_prompt(None)
        clear_conversation_history()
        await safe_answer_callback(callback, "♻️ پرامپت به حالت پیش‌فرض ایرانی برگشت.")
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "media":
        await state.clear()
        await safe_answer_callback(callback)
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await safe_edit_message(callback, m_text, m_kb)
        return

    if action == "madd":
        arg = (callback_data.arg or "").strip().lower()
        if arg in ("", "pick"):
            await state.clear()
            await safe_answer_callback(callback)
            mood_text = (
                "🎭 <b>افزودن رسانه به بانک هوش مصنوعی</b>\n\n"
                "اول حس‌وحال (Mood) رسانه‌هایی که می‌خوای بفرستی رو انتخاب کن:"
            )
            await safe_edit_message(callback, mood_text, build_ai_media_mood_kb(owner_id=user.id))
            return

        selected_mood = arg if arg in ("auto", *repository.AI_CHAT_MEDIA_MOODS) else "auto"
        await safe_answer_callback(callback)
        await state.set_state(AiChatStates.waiting_chat_media)
        await state.update_data(
            fsm_started_at=time.monotonic(),
            media_mood=selected_mood,
        )
        mood_labels = {
            "auto": "✨ خودکار / عمومی",
            "laugh": "😁 خنده",
            "agree": "👍 تایید",
            "disagree": "👎 مخالفت",
            "hype": "🔥 هیجان",
            "general": "✨ عمومی",
        }
        upload_text = (
            f"📥 <b>ارسال رسانه ({mood_labels.get(selected_mood, selected_mood)})</b>\n\n"
            "الان هر چند تا <b>استیکر</b>، <b>گیف (GIF)</b> یا <b>ویدیومسیج (ویدیو گرد)</b> می‌خوای همینجا بفرست تا ذخیره بشن.\n"
            "<i>وقتی کارت تموم شد، دکمه «اتمام و بازگشت» یا /done رو بزن.</i>"
        )
        await safe_edit_message(callback, upload_text, build_ai_media_done_kb(owner_id=user.id))
        return

    if action == "mdone":
        await state.clear()
        await safe_answer_callback(callback, "✅ ثبت رسانه‌ها تمام شد.")
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await safe_edit_message(callback, m_text, m_kb)
        return

    if action == "mauto":
        new_auto = await set_ai_chat_media_autocollect_state(not is_ai_chat_media_autocollect_cached())
        toast = "🤖 جمع‌آوری خودکار استیکر و گیف روشن شد." if new_auto else "⏸ جمع‌آوری خودکار استیکر و گیف خاموش شد."
        await safe_answer_callback(callback, toast)
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await safe_edit_message(callback, m_text, m_kb)
        return

    if action == "mtest":
        target_type = (callback_data.arg or "").strip()
        if target_type not in repository.AI_CHAT_MEDIA_TYPES or not callback.message or not callback.message.chat:
            await safe_answer_callback(callback, "⚠️ نوع رسانه نامعتبر است.", show_alert=True)
            return
        test_chat = callback.message.chat
        test_tid = resolve_thread_id(callback.message) if test_chat.type != "private" else 0
        test_send_tid = to_send_thread_id(test_tid) if test_chat.type != "private" else None
        sent = await send_ai_chat_media_reply(
            bot,
            chat_id=test_chat.id,
            thread_id=test_tid,
            send_thread_id=test_send_tid,
            media_type=target_type,
            mood="general",
        )
        await safe_answer_callback(
            callback,
            "✅ ارسال شد!" if sent else "⚠️ رسانه‌ای از این نوع در بانک موجود نیست.",
            show_alert=not sent,
        )
        return

    if action == "mclr":
        clr_mode = (callback_data.arg or "all").strip().lower()
        if clr_mode == "auto":
            deleted = await repository.clear_ai_chat_media(source="auto")
            toast = f"🗑 {deleted} رسانه خودکار پاک شد."
        else:
            deleted = await repository.clear_ai_chat_media()
            toast = f"🗑 کل بانک رسانه ({deleted} مورد) پاک شد."
        invalidate_ai_media_counts_cache()
        await safe_answer_callback(callback, toast)
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await safe_edit_message(callback, m_text, m_kb)
        return

    if action == "cancel":
        await state.clear()
        await safe_answer_callback(callback, "لغو شد.")
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
        return

    if action == "close":
        await state.clear()
        await safe_answer_callback(callback)
        if callback.message:
            try:
                await callback.message.delete()
                return
            except Exception:
                await safe_edit_message(callback, "✖️ پنل مدیریت هوش مصنوعی بسته شد.")
        return

    await safe_answer_callback(callback)


@router.message(StateFilter(AiChatStates.waiting_custom_prompt))
async def on_custom_prompt_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Receive custom system prompt text from the bot owner."""
    user = message.from_user
    if user is None or not is_global_admin(user.id):
        return

    data = await state.get_data()
    started_at = data.get("fsm_started_at")
    if isinstance(started_at, (int, float)) and (time.monotonic() - started_at) > FSM_PROMPT_TTL_SECONDS:
        await state.clear()
        await _send_in_context(message, "⏰ زمان وارد کردن پرامپت تموم شد. دوباره <code>/manage_ai</code> رو بزن.", bot=bot)
        return

    raw_text = (message.text or "").strip()
    if raw_text.lower().startswith("/cancel"):
        await state.clear()
        await _send_in_context(message, "❌ ویرایش پرامپت لغو شد.", bot=bot)
        return

    if not raw_text:
        await _send_in_context(message, "⚠️ لطفاً متن پرامپت رو به‌صورت پیام متنی بفرست (یا /cancel بزن).", bot=bot)
        return

    if len(raw_text) > 4000:
        await _send_in_context(message, "⚠️ طول پرامپت خیلی زیاده (حداکثر ۴۰۰۰ کاراکتر مجازه).", bot=bot)
        return

    await repository.set_ai_chat_custom_prompt(raw_text)
    clear_conversation_history()
    await state.clear()

    _, bot_username = await ensure_ai_bot_identity(bot)
    panel_text, kb = await render_manage_ai_panel(owner_id=user.id, bot_username=bot_username)
    await _send_in_context(
        message,
        f"✅ پرامپت جدید ذخیره شد و حافظه مکالمه تازه‌سازی شد.\n\n{panel_text}",
        reply_markup=kb,
        bot=bot,
    )


@router.message(StateFilter(AiChatStates.waiting_quota_limit))
async def on_custom_quota_input(message: Message, bot: Bot, state: FSMContext) -> None:
    """Receive custom regular-user AI chat quota limit from the bot owner."""
    user = message.from_user
    if user is None or not is_global_admin(user.id):
        return

    data = await state.get_data()
    started_at = data.get("fsm_started_at")
    if isinstance(started_at, (int, float)) and (time.monotonic() - started_at) > FSM_PROMPT_TTL_SECONDS:
        await state.clear()
        await _send_in_context(
            message,
            "⏰ زمان وارد کردن سهمیه تموم شد. دوباره <code>/manage_ai quota</code> رو بزن.",
            bot=bot,
        )
        return

    raw_text = (message.text or "").strip()
    if raw_text.lower().startswith("/cancel"):
        await state.clear()
        await _send_in_context(message, "❌ تنظیم سهمیه لغو شد.", bot=bot)
        return

    parsed = _parse_quota_value(raw_text)
    if parsed is None:
        await _send_in_context(
            message,
            "⚠️ لطفاً یک عدد بین ۱ تا ۱۰۰۰۰، یا <code>0</code> (بستن)، یا <code>-1</code> / <code>نامحدود</code> بفرست (یا /cancel بزن).",
            bot=bot,
        )
        return

    effective = await set_ai_chat_user_limit_state(parsed)
    await state.clear()

    window_hours = max(1, int(getattr(get_settings(), "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
    q_text, q_kb = await _render_quota_panel(owner_id=user.id)
    await _send_in_context(
        message,
        f"✅ سهمیه کاربران معمولی روی <b>{_format_quota_summary(effective, window_hours)}</b> تنظیم شد.\n\n{q_text}",
        reply_markup=q_kb,
        bot=bot,
    )


@router.message(StateFilter(AiChatStates.waiting_chat_media))
async def on_chat_media_upload(message: Message, bot: Bot, state: FSMContext) -> None:
    """Receive stickers, GIFs (animations), or circular video notes from the bot owner."""
    user = message.from_user
    if user is None or not is_global_admin(user.id):
        return

    data = await state.get_data()
    started_at = data.get("fsm_started_at")
    if isinstance(started_at, (int, float)) and (time.monotonic() - started_at) > MEDIA_FSM_TTL_SECONDS:
        await state.clear()
        await _send_in_context(
            message,
            "⏰ زمان افزودن رسانه تموم شد. دوباره <code>/manage_ai media</code> رو بزن.",
            bot=bot,
        )
        return

    raw_text = (message.text or "").strip()
    if raw_text.lower().startswith(("/cancel", "/done")) or raw_text in ("تموم", "پایان", "بس", "انصراف"):
        await state.clear()
        m_text, m_kb = await _render_media_panel(owner_id=user.id)
        await _send_in_context(message, f"✅ ثبت رسانه‌ها تمام شد.\n\n{m_text}", reply_markup=m_kb, bot=bot)
        return

    media = extract_ai_chat_media(message)
    if media is None:
        await _send_in_context(
            message,
            "⚠️ لطفاً فقط <b>استیکر</b>، <b>گیف (GIF)</b> یا <b>ویدیومسیج (ویدیو گرد)</b> بفرست (برای پایان /done بزن).",
            reply_markup=build_ai_media_done_kb(owner_id=user.id),
            bot=bot,
        )
        return

    selected_mood = str(data.get("media_mood") or "auto").strip().lower()
    effective_mood = (
        infer_mood_from_emoji(media.get("emoji"))
        if selected_mood == "auto"
        else (selected_mood if selected_mood in repository.AI_CHAT_MEDIA_MOODS else "general")
    )

    _, is_new = await repository.upsert_ai_chat_media(
        media_type=media["media_type"],
        file_id=media["file_id"],
        file_unique_id=media["file_unique_id"],
        mood=effective_mood,
        emoji=media.get("emoji"),
        source="manual",
        added_by=user.id,
        chat_id=message.chat.id if message.chat else None,
    )
    invalidate_ai_media_counts_cache()
    await state.update_data(fsm_started_at=time.monotonic())

    counts = await repository.count_ai_chat_media_by_type()
    type_label = {"sticker": "استیکر", "animation": "گیف", "video_note": "ویدیومسیج"}.get(
        media["media_type"], media["media_type"]
    )
    status_word = "ذخیره شد" if is_new else "به‌روزرسانی شد (از قبل بود)"
    await _send_in_context(
        message,
        (
            f"✅ {type_label} با حس <code>{escape_html(effective_mood)}</code> {status_word}!\n"
            f"📦 موجودی فعلی: استیکر <b>{counts.get('sticker', 0)}</b> | "
            f"گیف <b>{counts.get('animation', 0)}</b> | "
            f"ویدیومسیج <b>{counts.get('video_note', 0)}</b>\n"
            "<i>می‌تونی بازم بفرستی یا دکمه زیر رو بزنی.</i>"
        ),
        reply_markup=build_ai_media_done_kb(owner_id=user.id),
        bot=bot,
    )


class AiChatTriggerFilter(Filter):
    """Allow interactive AI chat in PV (owner-only) or in the free-chat group topic when addressed."""

    async def __call__(self, message: Message, bot: Bot) -> bool:
        if not is_ai_chat_enabled_cached():
            return False

        if not message.chat or not message.from_user or message.from_user.is_bot:
            return False

        if message.sender_chat:
            return False

        raw_text = (message.text or message.caption or "").strip()
        has_media = extract_ai_chat_media(message) is not None
        if raw_text.startswith("/"):
            return False
        if not raw_text and not has_media:
            return False

        is_owner = is_global_admin(message.from_user.id)

        if message.chat.type == "private":
            return is_owner

        if message.chat.type == "supergroup":
            settings = get_settings()
            if settings.ALLOWED_CHAT_IDS and message.chat.id not in settings.ALLOWED_CHAT_IDS:
                return False

            if get_effective_ai_chat_access_mode() == "owner_only" and not is_owner:
                return False

            thread_id = resolve_thread_id(message)
            if not await is_allowed_ai_chat_topic(message.chat.id, thread_id):
                return False

            bot_id, bot_username = await ensure_ai_bot_identity(bot)
            return is_message_addressing_bot(message, bot_id, bot_username)

        return False


@router.message(StateFilter(default_state), AiChatTriggerFilter())
async def handle_ai_chat_message(message: Message, bot: Bot) -> None:
    """Respond naturally in Iranian Persian with zero-token reactions and sticker/GIF/video_note replies."""
    chat = message.chat
    user = message.from_user
    if chat is None or user is None:
        return

    is_group = chat.type != "private"
    thread_id = resolve_thread_id(message) if is_group else 0
    send_thread_id = to_send_thread_id(thread_id) if is_group else None

    _, bot_username = await ensure_ai_bot_identity(bot)
    raw_text = (message.text or message.caption or "").strip()
    stripped_text = strip_bot_mention(raw_text, bot_username)
    incoming_media = extract_ai_chat_media(message)

    # Best-effort passive capture if group user addressed the bot with a sticker/GIF
    if is_group and incoming_media is not None:
        try:
            await maybe_capture_ai_chat_media(message)
        except Exception:
            pass

    # 1. Zero-token fast-path for media-only messages (sticker / GIF / video_note)
    if not stripped_text and incoming_media is not None:
        if incoming_media["media_type"] == "sticker" and incoming_media.get("emoji"):
            react_emoji = classify_emoji_sentiment(str(incoming_media["emoji"]))
        elif incoming_media["media_type"] == "animation":
            react_emoji = "😁"
        else:
            react_emoji = "👍"
        await send_ai_reaction(bot, chat.id, message.message_id, react_emoji)

        try:
            media_counts = await get_ai_chat_media_counts_cached()
        except Exception:
            media_counts = {}
        if media_counts.get(incoming_media["media_type"], 0) > 0 and should_reply_with_media_to_media(
            chat.id, thread_id
        ):
            mood = infer_mood_from_emoji(incoming_media.get("emoji"))
            await send_ai_chat_media_reply(
                bot,
                chat_id=chat.id,
                thread_id=thread_id,
                send_thread_id=send_thread_id,
                media_type=incoming_media["media_type"],
                mood=mood,
                reply_to_message_id=message.message_id if is_group else None,
            )
        return

    # 2. Zero-token fast-path for emoji-only or short no-reply acknowledgments
    zero_token_react = classify_zero_token_reaction(stripped_text)
    if zero_token_react is not None:
        await send_ai_reaction(bot, chat.id, message.message_id, zero_token_react)
        return

    # 3. Zero-token fast-path for explicit sticker / GIF / video_note requests
    requested_media = detect_explicit_media_request(stripped_text)
    if requested_media is not None:
        sent = await send_ai_chat_media_reply(
            bot,
            chat_id=chat.id,
            thread_id=thread_id,
            send_thread_id=send_thread_id,
            media_type=requested_media,
            mood="general",
            reply_to_message_id=message.message_id if is_group else None,
        )
        if not sent:
            media_label = {"sticker": "استیکر", "animation": "گیف", "video_note": "ویدیومسیج"}.get(
                requested_media, "رسانه"
            )
            empty_msg = f"هنوز {media_label}ی توی بانکم ذخیره نشده 😅"
            if is_global_admin(user.id):
                empty_msg += "\n💡 از <code>/manage_ai media</code> می‌تونی اضافه کنی."
            await _send_in_context(message, empty_msg, bot=bot)
        return

    # 4. Standard LLM conversational path (quota-guarded)
    allowed, _remaining, wait_mins = check_and_consume_user_quota(user.id)
    if not allowed:
        if should_send_quota_notice(user.id):
            settings = get_settings()
            limit_count = get_effective_ai_chat_user_limit()
            window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
            if limit_count == 0 or wait_mins < 0:
                notice_text = "🔒 سهمیه چت کاربران معمولی فعلاً توسط مدیر بسته شده."
            else:
                notice_text = (
                    f"⏳ سهمیه چت تو ({limit_count} پیام در هر {window_hours} ساعت) فعلاً پر شده رفیق! "
                    f"حدود <b>{wait_mins} دقیقه</b> دیگه دوباره می‌تونی باهام حرف بزنی."
                )
            try:
                await bot.send_message(
                    chat_id=chat.id,
                    message_thread_id=send_thread_id,
                    text=notice_text,
                    parse_mode="HTML",
                    reply_to_message_id=message.message_id if is_group else None,
                    disable_web_page_preview=True,
                )
            except Exception:
                pass
        return

    user_text = stripped_text or "سلام"

    topic_name: str | None = None
    if is_group:
        try:
            topic_name = await repository.get_topic_name(chat.id, thread_id)
        except Exception:
            topic_name = None

    reply_context: str | None = None
    reply = message.reply_to_message
    if reply and not getattr(reply, "forum_topic_created", None):
        reply_raw = (reply.text or reply.caption or "").strip()
        if reply_raw:
            reply_context = reply_raw[:160]

    speaker_name = (user.first_name or user.full_name or "دوست عزیز").strip() or "دوست عزیز"

    try:
        await bot.send_chat_action(
            chat_id=chat.id,
            action="typing",
            message_thread_id=send_thread_id,
        )
    except Exception:
        pass

    try:
        reply_html = await generate_ai_chat_reply(
            chat_id=chat.id,
            thread_id=thread_id,
            chat_type=chat.type,
            speaker_name=speaker_name,
            user_text=user_text,
            speaker_user_id=user.id,
            topic_name=topic_name,
            reply_context=reply_context,
        )
    except ValueError as exc:
        refund_user_quota(user.id)
        logger.warning("AI chat configuration error: %s", exc)
        await _send_in_context(
            message,
            f"⚠️ تنظیمات کلید/مدل برای سرویس فعلی کامل نیست: <code>{escape_html(str(exc))}</code>",
            bot=bot,
        )
        return
    except LLMError as exc:
        refund_user_quota(user.id)
        logger.warning("AI chat LLMError for chat %s [status=%s]: %s", chat.id, exc.status_code, exc)
        err_text = "الان به سرویس هوش مصنوعی وصل نشدم 😅 یه چند لحظه دیگه دوباره پیام بده."
        if is_global_admin(user.id):
            safe_err = _format_safe_llm_error(exc)
            err_text += f"\n\n🔍 <b>علت خطا:</b> <code>{escape_html(safe_err)}</code>"
            if "location is not supported" in safe_err.lower():
                err_text += "\n💡 آی‌پی سرور در Google AI Studio محدود است؛ از <code>/manage_ai 9router</code> (یا <code>/manage_ai openai</code>) استفاده کن."
        await _send_in_context(
            message,
            err_text,
            bot=bot,
        )
        return
    except Exception as exc:
        refund_user_quota(user.id)
        logger.error("Unexpected error in AI chat handler: %s", exc, exc_info=True)
        await _send_in_context(
            message,
            "یه مشکل غیرمنتظره موقع جواب دادن پیش اومد 😅 دوباره امتحان کن.",
            bot=bot,
        )
        return

    parsed = parse_ai_chat_directives(reply_html)
    if parsed.reaction and not parsed.text and not parsed.media_type:
        await send_ai_reaction(bot, chat.id, message.message_id, parsed.reaction)
        return

    if parsed.reaction:
        await send_ai_reaction(bot, chat.id, message.message_id, parsed.reaction)

    if parsed.text:
        parts = smart_split(parsed.text, limit=TELEGRAM_MESSAGE_MAX_CHARS, count_rendered=True)
        for idx, part in enumerate(parts):
            reply_to = message.message_id if (is_group and idx == 0) else None
            try:
                await bot.send_message(
                    chat_id=chat.id,
                    message_thread_id=send_thread_id,
                    text=part,
                    parse_mode="HTML",
                    reply_to_message_id=reply_to,
                    disable_web_page_preview=True,
                )
            except TelegramBadRequest as exc:
                err_lower = str(exc).lower()
                if "can't parse entities" in err_lower or "parsing" in err_lower:
                    await bot.send_message(
                        chat_id=chat.id,
                        message_thread_id=send_thread_id,
                        text=strip_html_tags(part),
                        parse_mode=None,
                        reply_to_message_id=reply_to,
                        disable_web_page_preview=True,
                    )
                elif "message to be replied not found" in err_lower or "reply" in err_lower:
                    await bot.send_message(
                        chat_id=chat.id,
                        message_thread_id=send_thread_id,
                        text=part,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                else:
                    logger.error("Failed sending AI chat reply to %s: %s", chat.id, exc)
                    break

    if parsed.media_type:
        sent_media = await send_ai_chat_media_reply(
            bot,
            chat_id=chat.id,
            thread_id=thread_id,
            send_thread_id=send_thread_id,
            media_type=parsed.media_type,
            mood=parsed.media_mood,
            reply_to_message_id=message.message_id if (is_group and not parsed.text) else None,
        )
        if not sent_media and not parsed.text:
            await send_ai_reaction(bot, chat.id, message.message_id, mood_to_reaction(parsed.media_mood))
