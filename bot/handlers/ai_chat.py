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
    build_ai_memory_view_kb,
    build_ai_prompt_cancel_kb,
    build_ai_prompt_view_kb,
    build_manage_ai_kb,
)
from bot.services.access import is_global_admin
from bot.services.ai_chat_service import (
    check_and_consume_user_quota,
    clear_conversation_history,
    count_active_conversations,
    ensure_ai_bot_identity,
    generate_ai_chat_reply,
    get_effective_ai_chat_access_mode,
    get_effective_ai_chat_provider,
    is_ai_chat_enabled_cached,
    is_message_addressing_bot,
    refund_user_quota,
    set_ai_chat_access_mode_state,
    set_ai_chat_provider_state,
    set_ai_chat_state,
    should_send_quota_notice,
    strip_bot_mention,
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


class AiChatStates(StatesGroup):
    """FSM state for entering a custom system prompt in /manage_ai."""

    waiting_custom_prompt = State()


def _parse_manage_arg(raw_text: str | None) -> tuple[bool | None, str | None, str | None]:
    """Parse optional shorthand argument from '/manage_ai on|off|9router|openai|gemini|bind|learn|memory'."""
    if not raw_text:
        return None, None, None
    parts = raw_text.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, None, None
    arg = parts[1].strip().lower()
    if arg in ("on", "enable", "enabled", "1", "true", "روشن", "فعال"):
        return True, None, None
    if arg in ("off", "disable", "disabled", "0", "false", "خاموش", "غیرفعال"):
        return False, None, None
    if arg in ("9router", "ninerouter", "9-router"):
        return None, "9router", None
    if arg in ("openai", "openrouter", "gateway"):
        return None, "openai", None
    if arg in ("gemini", "google", "aistudio"):
        return None, "gemini", None
    if arg in ("bind", "اتصال"):
        return None, None, "bind"
    if arg in ("learn", "یادگیری"):
        return None, None, "learn"
    if arg in ("memory", "mem", "حافظه"):
        return None, None, "memory"
    return None, None, None


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
) -> tuple[str, InlineKeyboardMarkup]:
    """Render learned group culture and member personalities overview."""
    mem_chat_id, group_title = await resolve_memory_group(chat_id, chat_type)
    if mem_chat_id is None:
        return (
            "👥 <b>حافظه بلندمدت اعضا</b>\n\nهنوز گروه فعالی برای یادگیری پیدا نشد.",
            build_ai_memory_view_kb(owner_id=owner_id, has_memory=False),
        )

    source = await ensure_memory_source_for_group(mem_chat_id)
    group_mem = await repository.get_ai_group_memory(mem_chat_id)
    profiles = await repository.list_ai_member_profiles(mem_chat_id, limit=12)
    total_profiles = await repository.count_ai_member_profiles(mem_chat_id)

    lines = [f"👥 <b>حافظه بلندمدت گروه {escape_html(group_title or str(mem_chat_id))}</b>\n"]
    if source:
        src_name = source.get("topic_name") or f"تاپیک {source['thread_id']}"
        lines.append(f"📚 تاپیک یادگیری: <b>{escape_html(src_name)}</b>")
    else:
        lines.append("📚 تاپیک یادگیری: <i>تنظیم نشده (/manage_ai bind داخل گفتگوی آزاد)</i>")

    if group_mem and (
        group_mem.get("culture_summary") or group_mem.get("inside_jokes") or group_mem.get("ongoing_topics")
    ):
        lines.append("\n🌐 <b>شناخت از فضای گروه:</b>")
        if group_mem.get("culture_summary"):
            lines.append(f"• حال‌وهوا: {escape_html(group_mem['culture_summary'])}")
        if group_mem.get("inside_jokes"):
            lines.append(f"• شوخی‌ها و اصطلاحات: {escape_html(group_mem['inside_jokes'])}")
        if group_mem.get("ongoing_topics"):
            lines.append(f"• بحث‌های فعلی: {escape_html(group_mem['ongoing_topics'])}")

    if profiles:
        lines.append(f"\n🧑 <b>اعضای شناخته‌شده ({total_profiles} نفر):</b>")
        for prof in profiles:
            name = escape_html(str(prof.get("display_name") or "عضو"))
            summary = escape_html(str(prof.get("personality_summary") or ""))
            entry = f"• <b>{name}</b>: {summary}"
            if len("\n".join(lines)) + len(entry) + 2 > 3600:
                lines.append("<i>… و اعضای بیشتر در دیتابیس</i>")
                break
            lines.append(entry)
    else:
        lines.append("\n<i>هنوز پروفایلی یادگیری نشده؛ دکمه «🧠 یادگیری الان» رو بزن.</i>")

    has_memory = bool(profiles or group_mem)
    return "\n".join(lines), build_ai_memory_view_kb(owner_id=owner_id, has_memory=has_memory)


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

    limit_count = int(getattr(settings, "AI_CHAT_USER_LIMIT_COUNT", 20))
    window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
    status_badge = "🟢 <b>روشن</b>" if enabled else "🔴 <b>خاموش</b>"
    provider_label = format_ai_provider_label(settings, for_chat=True, provider=provider)
    access_badge = (
        f"👥 <b>همه اعضای گفتگوی آزاد</b> ({limit_count} پیام در {window_hours} ساعت | مدیر نامحدود)"
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
        f"• حافظه بلندمدت: 👥 <b>{learned_count}</b> عضو شناخته‌شده | مکالمه‌های موقت: <b>{conv_count}</b>\n\n"
        "<b>نحوه استفاده هنگام روشن بودن:</b>\n"
        f"• فقط در گفتگوی آزاد: روی پیام ربات <b>ریپلای</b> بزن یا با {mention_hint} صداش کن\n"
        "• در پیوی: فقط مدیر ربات\n"
        "• میانبر سریع: <code>/manage_ai on</code> | <code>off</code> | <code>9router</code> | <code>gemini</code> | <code>bind</code> | <code>learn</code> | <code>memory</code>"
    )
    kb = build_manage_ai_kb(
        enabled=enabled,
        has_custom_prompt=bool(custom_prompt),
        owner_id=owner_id,
        provider=provider,
        access_mode=access_mode,
        gateway_label=gateway_label,
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
    desired_toggle, desired_provider, memory_action = _parse_manage_arg(message.text)

    if desired_toggle is not None:
        await set_ai_chat_state(desired_toggle)
    if desired_provider is not None:
        await set_ai_chat_provider_state(desired_provider)

    notice = ""
    if memory_action == "bind":
        if chat_type == "private" or chat_id is None:
            notice = "⚠️ دستور <code>/manage_ai bind</code> رو باید داخل خودِ تاپیک موردنظر (مثلاً گفتگوی آزاد) بزنی.\n\n"
        else:
            tid = resolve_thread_id(message)
            src = await bind_memory_source_topic(chat_id, tid)
            notice = f"✅ تاپیک <b>«{escape_html(src.get('topic_name') or str(tid))}»</b> برای یادگیری شخصیت اعضا متصل شد.\n\n"
    elif memory_action == "learn":
        try:
            ok_learn, learn_msg = await _run_manual_memory_learn(chat_id, chat_type)
            prefix = "✅" if ok_learn else "⚠️"
            notice = f"{prefix} {escape_html(learn_msg)}\n\n"
        except LLMError as exc:
            notice = f"⚠️ خطا در یادگیری از پیام‌ها: <code>{escape_html(_format_safe_llm_error(exc))}</code>\n\n"
    elif memory_action == "memory":
        mem_text, mem_kb = await _render_memory_overview(owner_id=user.id, chat_id=chat_id, chat_type=chat_type)
        await _send_in_context(message, mem_text, reply_markup=mem_kb, bot=bot)
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
        toast = (
            "👥 چت برای همه اعضای گفتگوی آزاد باز شد (۲۰ پیام در ۲ ساعت)."
            if next_mode == "public"
            else "🔒 دسترسی چت به فقط مدیر ربات محدود شد."
        )
        await safe_answer_callback(callback, toast)
        text, kb = await render_manage_ai_panel(
            owner_id=user.id, bot_username=bot_username, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, text, kb)
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
            owner_id=user.id, chat_id=cb_chat_id, chat_type=cb_chat_type
        )
        await safe_edit_message(callback, mem_text, mem_kb)
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
        text, kb = await render_manage_ai_panel(owner_id=user.id, bot_username=bot_username)
        await safe_edit_message(callback, text, kb)
        return

    if action == "cancel":
        await state.clear()
        await safe_answer_callback(callback, "لغو شد.")
        text, kb = await render_manage_ai_panel(owner_id=user.id, bot_username=bot_username)
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
        if not raw_text or raw_text.startswith("/"):
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
    """Respond naturally in Iranian Persian in PV (owner) or free-chat topic (with 20/2h user quota)."""
    chat = message.chat
    user = message.from_user
    if chat is None or user is None:
        return

    is_group = chat.type != "private"
    thread_id = resolve_thread_id(message) if is_group else 0
    send_thread_id = to_send_thread_id(thread_id) if is_group else None

    allowed, _remaining, wait_mins = check_and_consume_user_quota(user.id)
    if not allowed:
        if should_send_quota_notice(user.id):
            settings = get_settings()
            limit_count = int(getattr(settings, "AI_CHAT_USER_LIMIT_COUNT", 20))
            window_hours = max(1, int(getattr(settings, "AI_CHAT_USER_LIMIT_WINDOW_SECONDS", 7200)) // 3600)
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

    _, bot_username = await ensure_ai_bot_identity(bot)
    raw_text = (message.text or message.caption or "").strip()
    user_text = strip_bot_mention(raw_text, bot_username) or "سلام"

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

    parts = smart_split(reply_html, limit=TELEGRAM_MESSAGE_MAX_CHARS, count_rendered=True)
    if not parts:
        return

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
