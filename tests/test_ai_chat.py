"""Offline unit tests for owner-only Iranian Persian AI chat and /manage_ai panel."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, Message, User
from bot.database import repository
from bot.database.db import close_db, init_db
from bot.handlers.ai_chat import (
    AiChatStates,
    AiChatTriggerFilter,
    _format_safe_llm_error,
    cmd_manage_ai,
    handle_ai_chat_message,
    on_custom_prompt_input,
    on_custom_quota_input,
    on_manage_ai_callback,
    render_manage_ai_panel,
)
from bot.keyboards.ai_chat import AiChatCB
from bot.services.ai_chat_service import (
    check_and_consume_user_quota,
    clear_conversation_history,
    count_active_conversations,
    count_active_quota_users,
    generate_ai_chat_reply,
    get_effective_ai_chat_access_mode,
    get_effective_ai_chat_provider,
    get_effective_ai_chat_user_limit,
    init_ai_chat_state,
    is_ai_chat_enabled_cached,
    refund_user_quota,
    reset_ai_chat_state,
    reset_all_user_quotas,
    set_ai_bot_identity,
    set_ai_chat_access_mode_state,
    set_ai_chat_state,
    set_ai_chat_user_limit_state,
    should_send_quota_notice,
)
from bot.services.llm_client import GeminiClient, LLMError, OpenAIClient


def _fake_settings(admin_ids: list[int] | None = None, allowed_chats: list[int] | None = None):
    return SimpleNamespace(
        ADMIN_IDS=admin_ids if admin_ids is not None else [11111],
        ALLOWED_CHAT_IDS=allowed_chats if allowed_chats is not None else [],
        AI_CHAT_PROVIDER="gemini",
        AI_CHAT_API_KEY="AQ.TestKeyForUnitTests",
        GEMINI_API_KEY=None,
        AI_CHAT_MODEL="gemini-3.5-flash-lite",
        AI_CHAT_FALLBACK_MODELS=["gemini-flash-lite-latest", "gemini-3.8-flash"],
        AI_CHAT_OPENAI_BASE_URL=None,
        AI_CHAT_OPENAI_API_KEY=None,
        AI_CHAT_OPENAI_MODEL=None,
        OPENAI_BASE_URL="https://openrouter.ai/api/v1",
        OPENAI_API_KEY="sk-or-test9999",
        OPENAI_MODEL="google/gemini-2.5-flash",
        AI_REASONING_EFFORT=None,
        AI_CHAT_MAX_OUTPUT_TOKENS=600,
        AI_CHAT_TEMPERATURE=0.85,
        AI_CHAT_HISTORY_LIMIT=8,
        AI_CHAT_USER_LIMIT_COUNT=20,
        AI_CHAT_USER_LIMIT_WINDOW_SECONDS=7200,
        TIMEZONE="Asia/Tehran",
        PROXY_URL=None,
    )


def _make_fsm() -> FSMContext:
    state = MagicMock(spec=FSMContext)
    state.get_state = AsyncMock(return_value=None)
    state.clear = AsyncMock()
    state.set_state = AsyncMock()
    state.get_data = AsyncMock(return_value={})
    state.update_data = AsyncMock()
    return state


def _make_message(
    user_id: int,
    text: str,
    *,
    chat_id: int | None = None,
    chat_type: str = "private",
    thread_id: int | None = None,
    message_id: int = 500,
    reply_from_user_id: int | None = None,
    reply_text: str | None = None,
    first_name: str = "آرش",
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.text = text
    msg.caption = None
    msg.sender_chat = None
    msg.message_thread_id = thread_id
    msg.is_topic_message = thread_id is not None

    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id if chat_id is not None else user_id
    msg.chat.type = chat_type

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.first_name = first_name
    user.full_name = f"{first_name} تست"
    user.username = "arash_owner"
    msg.from_user = user

    if reply_from_user_id is not None:
        reply = MagicMock(spec=Message)
        reply.forum_topic_created = None
        reply.text = reply_text
        reply.caption = None
        reply_user = MagicMock(spec=User)
        reply_user.id = reply_from_user_id
        reply.from_user = reply_user
        msg.reply_to_message = reply
    else:
        msg.reply_to_message = None

    msg.answer = AsyncMock()
    msg.delete = AsyncMock()
    return msg


def _make_bot(bot_id: int = 999999, bot_username: str = "PrimeTestBot") -> Bot:
    bot = AsyncMock(spec=Bot)
    bot.id = bot_id
    bot.get_me = AsyncMock(return_value=SimpleNamespace(id=bot_id, username=bot_username))
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_gemini_client_generate_chat_multiturn_payload_and_thought_filter() -> None:
    """Verify GeminiClient.generate_chat sends multi-turn contents, temperature, and handles MAX_TOKENS cleanly."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {
                    "parts": [
                        {"text": "internal thought", "thought": True},
                        {"text": "آره دقیقاً، همون مسیر خیلی بهتره!"},
                    ]
                },
            }
        ]
    }

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False
    mock_http.post = AsyncMock(return_value=mock_resp)

    client = GeminiClient(
        api_key="AQ.test",
        model="gemini-3.5-flash-lite",
        max_output_tokens=512,
        http_client=mock_http,
    )
    reply = await client.generate_chat(
        system_prompt="پرامپت تستی",
        messages=[
            {"role": "user", "text": "سلام چطوری؟"},
            {"role": "model", "text": "قربونت، تو چطوری؟"},
            {"role": "user", "text": "می‌خوام تست بگیرم"},
        ],
        temperature=0.85,
        max_output_tokens=1024,
        model="gemini-flash-lite-latest",
    )

    assert reply == "آره دقیقاً، همون مسیر خیلی بهتره!"
    assert "models/gemini-flash-lite-latest:generateContent" in mock_http.post.call_args.args[0]
    call_kwargs = mock_http.post.call_args.kwargs
    assert call_kwargs["headers"]["x-goog-api-key"] == "AQ.test"
    payload = call_kwargs["json"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "پرامپت تستی"
    assert len(payload["contents"]) == 3
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"
    assert payload["generationConfig"]["temperature"] == 0.85
    assert payload["generationConfig"]["maxOutputTokens"] == 1024
    assert "thinkingConfig" not in payload["generationConfig"]


@pytest.mark.asyncio
async def test_ai_chat_db_persistence_and_cache_rehydration(tmp_path: Path) -> None:
    """Verify ai_chat_enabled and ai_chat_custom_prompt persist in SQLite meta and rehydrate cache."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_chat_meta.db"))
    try:
        assert await repository.get_ai_chat_enabled() is False
        assert await init_ai_chat_state() is False
        assert is_ai_chat_enabled_cached() is False

        await set_ai_chat_state(True)
        assert is_ai_chat_enabled_cached() is True
        assert await repository.get_ai_chat_enabled() is True

        # Simulate restart by clearing memory cache and reloading from SQLite
        reset_ai_chat_state()
        assert is_ai_chat_enabled_cached() is False
        assert await init_ai_chat_state() is True
        assert is_ai_chat_enabled_cached() is True

        assert await repository.get_ai_chat_custom_prompt() is None
        await repository.set_ai_chat_custom_prompt("  فقط خیلی خودمونی جواب بده  ")
        assert await repository.get_ai_chat_custom_prompt() == "فقط خیلی خودمونی جواب بده"

        await repository.set_ai_chat_custom_prompt(None)
        assert await repository.get_ai_chat_custom_prompt() is None
    finally:
        await close_db()
        reset_ai_chat_state()


def test_ai_chat_callback_data_under_64_bytes() -> None:
    """Ensure all AiChatCB packed strings fit Telegram's 64-byte callback_data limit."""
    for action in (
        "toggle",
        "prov",
        "acc",
        "quota",
        "qset",
        "qreset",
        "qcustom",
        "qcancel",
        "clear",
        "refresh",
        "viewp",
        "setp",
        "resetp",
        "learn",
        "mem",
        "clrmem",
        "noop",
        "cancel",
        "close",
    ):
        packed = AiChatCB(action=action, owner_id=999999999999, arg="10000", page=999).pack()
        assert len(packed.encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_manage_ai_command_and_callbacks_strictly_owner_only(tmp_path: Path) -> None:
    """Verify /manage_ai on|off and inline panel callbacks only work for ADMIN_IDS owners."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "manage_ai.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        bot = _make_bot()
        fsm = _make_fsm()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg):
            # 1. Non-owner denied in PV and ignored after delete in group
            stranger_pv = _make_message(user_id=22222, text="/manage_ai", chat_type="private")
            await cmd_manage_ai(stranger_pv, bot, fsm)
            stranger_reply = stranger_pv.answer.call_args.kwargs.get("text") or stranger_pv.answer.call_args.args[0]
            assert "فقط مخصوص مدیر ربات" in stranger_reply
            assert is_ai_chat_enabled_cached() is False

            stranger_grp = _make_message(
                user_id=22222,
                text="/manage_ai on",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            await cmd_manage_ai(stranger_grp, bot, fsm)
            stranger_grp.delete.assert_awaited_once()
            bot.send_message.assert_not_called()
            assert is_ai_chat_enabled_cached() is False

            # 2. Owner shorthand '/manage_ai on' turns it ON and sends panel in group topic
            owner_grp = _make_message(
                user_id=11111,
                text="/manage_ai on",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            await cmd_manage_ai(owner_grp, bot, fsm)
            assert is_ai_chat_enabled_cached() is True
            assert bot.send_message.call_count == 1
            sent_text = bot.send_message.call_args.kwargs["text"]
            assert "روشن" in sent_text
            assert bot.send_message.call_args.kwargs["message_thread_id"] == 3

            # 3. Non-owner callback rejected with alert
            cb_denied = MagicMock(spec=CallbackQuery)
            cb_denied.from_user = SimpleNamespace(id=22222)
            cb_denied.answer = AsyncMock()
            cb_denied.message = MagicMock(spec=Message)
            await on_manage_ai_callback(cb_denied, AiChatCB(action="toggle", owner_id=11111), bot, fsm)
            assert cb_denied.answer.call_args.kwargs["show_alert"] is True
            assert is_ai_chat_enabled_cached() is True

            # 4. Owner callback toggles OFF
            cb_owner = MagicMock(spec=CallbackQuery)
            cb_owner.from_user = SimpleNamespace(id=11111)
            cb_owner.answer = AsyncMock()
            cb_owner.message = MagicMock(spec=Message)
            cb_owner.message.chat = SimpleNamespace(id=-100555)
            cb_owner.message.message_id = 777
            cb_owner.message.edit_text = AsyncMock()
            await on_manage_ai_callback(cb_owner, AiChatCB(action="toggle", owner_id=11111), bot, fsm)
            assert is_ai_chat_enabled_cached() is False
            assert "خاموش" in cb_owner.message.edit_text.call_args.kwargs["text"]

            # 5. Owner '/manage_ai off' keeps it OFF
            owner_pv = _make_message(user_id=11111, text="/manage_ai off", chat_type="private")
            await cmd_manage_ai(owner_pv, bot, fsm)
            assert is_ai_chat_enabled_cached() is False

            # 6. Owner sets and resets custom prompt via FSM
            await on_manage_ai_callback(cb_owner, AiChatCB(action="setp", owner_id=11111), bot, fsm)
            fsm.set_state.assert_awaited_with(AiChatStates.waiting_custom_prompt)
            prompt_msg = _make_message(user_id=11111, text="لحن پرانرژی و کوتاه", chat_type="private")
            await on_custom_prompt_input(prompt_msg, bot, fsm)
            assert await repository.get_ai_chat_custom_prompt() == "لحن پرانرژی و کوتاه"

            await on_manage_ai_callback(cb_owner, AiChatCB(action="resetp", owner_id=11111), bot, fsm)
            assert await repository.get_ai_chat_custom_prompt() is None
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_ai_chat_trigger_filter_rules(tmp_path: Path) -> None:
    """Verify AiChatTriggerFilter enforces toggle, strict free-chat topic gating, access mode, and addressing."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_trigger.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        chat_id = -100555
        hub_tid = 3
        wins_tid = 9
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await repository.upsert_topic(chat_id, wins_tid, "Wins | بُردها")

        bot = _make_bot(bot_id=999999, bot_username="PrimeTestBot")
        set_ai_bot_identity(999999, "PrimeTestBot")
        flt = AiChatTriggerFilter()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ):
            # 1. When OFF: even owner in PV is blocked
            owner_pv = _make_message(user_id=11111, text="سلام رفیق", chat_type="private")
            assert await flt(owner_pv, bot) is False

            # Turn ON (default access_mode == 'public')
            await set_ai_chat_state(True)
            assert get_effective_ai_chat_access_mode() == "public"

            # 2. PV is ALWAYS owner-only: owner -> allowed, non-owner -> blocked, commands -> blocked
            assert await flt(owner_pv, bot) is True
            owner_cmd = _make_message(user_id=11111, text="/panel", chat_type="private")
            assert await flt(owner_cmd, bot) is False
            other_pv = _make_message(user_id=22222, text="سلام", chat_type="private")
            assert await flt(other_pv, bot) is False

            # 3. Non-owner in گفتگوی آزاد addressing bot -> ALLOWED in public mode!
            other_hub_mention = _make_message(
                user_id=22222,
                text="@PrimeTestBot چطوری؟",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
            )
            assert await flt(other_hub_mention, bot) is True
            other_hub_reply = _make_message(
                user_id=22222,
                text="راست میگی",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                reply_from_user_id=999999,
                reply_text="پیام قبلی ربات",
            )
            assert await flt(other_hub_reply, bot) is True

            # 4. Plain message without reply/mention in گفتگوی آزاد -> blocked
            owner_hub_plain = _make_message(
                user_id=11111,
                text="بچه‌ها جلسه ساعت ۹ شروع می‌شه",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
            )
            assert await flt(owner_hub_plain, bot) is False

            # 5. Any OTHER topic (e.g., Wins) -> strictly BLOCKED even for owner mentioning @bot
            owner_wins_mention = _make_message(
                user_id=11111,
                text="@PrimeTestBot چطوری؟",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=wins_tid,
            )
            assert await flt(owner_wins_mention, bot) is False
            other_wins_mention = _make_message(
                user_id=22222,
                text="@PrimeTestBot سلام",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=wins_tid,
            )
            assert await flt(other_wins_mention, bot) is False

            # 6. Switch to 'owner_only' mode -> non-owner in گفتگوی آزاد blocked, owner allowed
            await set_ai_chat_access_mode_state("owner_only")
            assert await flt(other_hub_mention, bot) is False
            owner_hub_mention = _make_message(
                user_id=11111,
                text="سلام @primetestbot نظرت چیه؟",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
            )
            assert await flt(owner_hub_mention, bot) is True
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_generate_ai_chat_reply_history_prompt_override_and_handler(tmp_path: Path) -> None:
    """Verify multi-turn conversation memory, custom prompt injection, and end-to-end group/PV handler replies."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_chat_reply.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        recorded_calls: list[tuple[str, list[dict[str, str]]]] = []

        async def fake_generate_chat(system_prompt, messages, temperature=0.85, max_output_tokens=1024, model=None):
            recorded_calls.append((system_prompt, list(messages)))
            return f"باشه آرش، حله! (تعداد پیام‌ها: {len(messages)})"

        mock_client = MagicMock(spec=GeminiClient)
        mock_client.generate_chat = AsyncMock(side_effect=fake_generate_chat)

        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_client
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg), patch(
            "bot.services.access.get_settings", return_value=cfg
        ):
            # Turn 1 in group topic 3
            r1 = await generate_ai_chat_reply(
                chat_id=-100555,
                thread_id=3,
                chat_type="supergroup",
                speaker_name="آرش",
                user_text="چطوری رفیق؟",
                topic_name="The Hub | گفتگوی آزاد",
            )
            assert "باشه آرش، حله!" in r1
            assert "The Hub | گفتگوی آزاد" in recorded_calls[0][0]
            assert len(recorded_calls[0][1]) == 1

            # Turn 2 in same topic retains Turn 1 user+model turns (total 3 items sent)
            r2 = await generate_ai_chat_reply(
                chat_id=-100555,
                thread_id=3,
                chat_type="supergroup",
                speaker_name="آرش",
                user_text="یکم ایده تولید محتوا بده",
                topic_name="The Hub | گفتگوی آزاد",
            )
            assert "تعداد پیام‌ها: 3" in r2
            assert count_active_conversations() == 1

            # Custom prompt override takes effect immediately
            await repository.set_ai_chat_custom_prompt("شخصیت تست سفارشی هیلز")
            await generate_ai_chat_reply(
                chat_id=-100555,
                thread_id=3,
                chat_type="supergroup",
                speaker_name="آرش",
                user_text="تست پرامپت",
            )
            assert "شخصیت تست سفارشی هیلز" in recorded_calls[-1][0]

            # Handler end-to-end in supergroup replies directly to message in thread
            bot = _make_bot(bot_id=999999, bot_username="PrimeTestBot")
            set_ai_bot_identity(999999, "PrimeTestBot")
            grp_msg = _make_message(
                user_id=11111,
                text="@PrimeTestBot یه سر بزن اینجا",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
                message_id=888,
            )
            await handle_ai_chat_message(grp_msg, bot)
            bot.send_chat_action.assert_awaited_once()
            bot.send_message.assert_awaited_once()
            send_kwargs = bot.send_message.call_args.kwargs
            assert send_kwargs["chat_id"] == -100555
            assert send_kwargs["message_thread_id"] == 3
            assert send_kwargs["reply_to_message_id"] == 888
            # Mention stripped before reaching model
            assert "@PrimeTestBot" not in recorded_calls[-1][1][-1]["text"]
            assert "یه سر بزن اینجا" in recorded_calls[-1][1][-1]["text"]

            # LLMError handled gracefully without crashing, shows safe owner diagnostics, and keeps count clean
            clear_conversation_history()
            mock_client.generate_chat = AsyncMock(
                side_effect=LLMError(
                    'Gemini API returned status 400: {"error":{"message":"API key AQ.Secret123456 not valid"}}',
                    status_code=400,
                    body_excerpt='{"error":{"message":"API key AQ.Secret123456 not valid"}}',
                    is_retryable=False,
                )
            )
            pv_msg = _make_message(user_id=11111, text="سلام", chat_type="private")
            await handle_ai_chat_message(pv_msg, bot)
            err_reply = pv_msg.answer.call_args.kwargs["text"]
            assert "وصل نشدم" in err_reply
            assert "[HTTP 400]" in err_reply
            assert "AQ.Secret123456" not in err_reply
            assert "***" in err_reply
            assert count_active_conversations() == 0

            # Panel shows masked last-4 key suffix only
            panel_text, _ = await render_manage_ai_panel(owner_id=11111, bot_username="PrimeTestBot")
            assert "AI_CHAT_API_KEY: ...ests" in panel_text
            assert "AQ.TestKeyForUnitTests" not in panel_text

            # Helper handles network errors cleanly
            net_msg = _format_safe_llm_error(LLMError("Network error connecting to Gemini API: timed out"))
            assert net_msg.startswith("[Network] ")
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_ai_chat_automatic_model_fallback_on_503_and_404(tmp_path: Path) -> None:
    """Verify generate_ai_chat_reply automatically falls back on 404/503 without failing the user."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_chat_fallback.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        tried_models: list[str] = []

        async def fallback_generate_chat(system_prompt, messages, temperature=0.85, max_output_tokens=1024, model=None):
            tried_models.append(model)
            if model == "gemini-3.5-flash-lite":
                raise LLMError("model deprecated", status_code=404, is_retryable=False)
            if model == "gemini-flash-lite-latest":
                raise LLMError("high demand", status_code=503, is_retryable=True)
            return "الان روبه‌راه شدم رفیق!"

        mock_client = MagicMock(spec=GeminiClient)
        mock_client.generate_chat = AsyncMock(side_effect=fallback_generate_chat)

        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_client
        ):
            reply = await generate_ai_chat_reply(
                chat_id=-100555,
                thread_id=1,
                chat_type="supergroup",
                speaker_name="آرش",
                user_text="تست فال‌بک",
            )

        assert reply == "الان روبه‌راه شدم رفیق!"
        assert tried_models == ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_openai_client_generate_chat_multiturn_roles_and_override() -> None:
    """Verify OpenAIClient.generate_chat sends system/user/assistant roles, temperature, and model override."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "قربونت رفیق، بگو چیکار کنیم!"},
            }
        ]
    }

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False
    mock_http.post = AsyncMock(return_value=mock_resp)

    client = OpenAIClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test9999",
        model="google/gemini-2.5-flash",
        max_output_tokens=1024,
        http_client=mock_http,
    )
    reply = await client.generate_chat(
        system_prompt="پرامپت فارسی",
        messages=[
            {"role": "user", "text": "سلام!"},
            {"role": "model", "text": "سلام رفیق!"},
            {"role": "user", "text": "یه ایده بده"},
        ],
        temperature=0.85,
        max_output_tokens=768,
        model="antigravity/gemini-2.5-flash",
    )

    assert reply == "قربونت رفیق، بگو چیکار کنیم!"
    assert mock_http.post.call_args.args[0] == "https://openrouter.ai/api/v1/chat/completions"
    payload = mock_http.post.call_args.kwargs["json"]
    assert payload["model"] == "antigravity/gemini-2.5-flash"
    assert payload["temperature"] == 0.85
    assert payload["max_tokens"] == 768
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_ai_chat_provider_switch_via_command_callback_and_reply(tmp_path: Path) -> None:
    """Verify /manage_ai openai|gemini, AiChatCB('prov'), OpenAI routing, and geo-block hint."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_provider_switch.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        assert len(AiChatCB(action="prov", owner_id=999999999999).pack().encode("utf-8")) <= 64
        bot = _make_bot()
        fsm = _make_fsm()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg):
            # 1. Switch to OpenAI via '/manage_ai openai'
            cmd_msg = _make_message(user_id=11111, text="/manage_ai openai", chat_type="private")
            await cmd_manage_ai(cmd_msg, bot, fsm)
            assert get_effective_ai_chat_provider() == "openai"
            assert await repository.get_ai_chat_provider() == "openai"
            panel_text = cmd_msg.answer.call_args.kwargs["text"]
            assert "OpenAI / Gateway" in panel_text
            assert "OPENAI_API_KEY: ...9999" in panel_text
            assert "google/gemini-2.5-flash" in panel_text

            # 2. Callback 'prov' toggles back to Gemini and then to OpenAI again
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=11111)
            cb.message.message_id = 101
            cb.message.edit_text = AsyncMock()

            await on_manage_ai_callback(cb, AiChatCB(action="prov", owner_id=11111), bot, fsm)
            assert get_effective_ai_chat_provider() == "gemini"
            await on_manage_ai_callback(cb, AiChatCB(action="prov", owner_id=11111), bot, fsm)
            assert get_effective_ai_chat_provider() == "openai"

            # 3. generate_ai_chat_reply routes through OpenAIClient when provider == 'openai'
            mock_openai = MagicMock(spec=OpenAIClient)
            mock_openai.generate_chat = AsyncMock(return_value="حله، الان از درگاه OpenAI جواب میدم!")
            with patch("bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_openai) as mock_factory:
                reply = await generate_ai_chat_reply(
                    chat_id=11111,
                    thread_id=0,
                    chat_type="private",
                    speaker_name="آرش",
                    user_text="تست سرویس OpenAI",
                )
            mock_factory.assert_called_once_with(cfg, provider="openai")
            assert reply == "حله، الان از درگاه OpenAI جواب میدم!"

            # 4. Geo-block error on Gemini includes /manage_ai openai hint
            await on_manage_ai_callback(cb, AiChatCB(action="prov", owner_id=11111), bot, fsm)
            assert get_effective_ai_chat_provider() == "gemini"
            with patch(
                "bot.handlers.ai_chat.generate_ai_chat_reply",
                side_effect=LLMError(
                    'Gemini API returned status 400: {"error":{"message":"User location is not supported for the API use."}}',
                    status_code=400,
                    body_excerpt='{"error":{"message":"User location is not supported for the API use."}}',
                    is_retryable=False,
                ),
            ):
                pv_msg = _make_message(user_id=11111, text="سلام", chat_type="private")
                await handle_ai_chat_message(pv_msg, bot)
            err_reply = pv_msg.answer.call_args.kwargs["text"]
            assert "/manage_ai openai" in err_reply
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_ai_chat_non_admin_rate_limit_20_per_2_hours_and_refund(tmp_path: Path) -> None:
    """Verify non-manager 20-msgs/2h sliding window, notice anti-spam, LLMError refund, and manager bypass."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_rate_limit.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_settings", return_value=cfg
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg):
            t0 = 10000.0
            # 1. Non-admin sends 20 messages -> all 20 allowed
            for i in range(20):
                ok, rem, wait = check_and_consume_user_quota(22222, now=t0 + i * 10)
                assert ok is True
                assert rem == 19 - i
                assert wait == 0

            # 2. 21st message at t0 + 300 -> blocked (~115 mins wait)
            ok21, rem21, wait21 = check_and_consume_user_quota(22222, now=t0 + 300)
            assert ok21 is False
            assert rem21 == 0
            assert 110 <= wait21 <= 120

            # Anti-spam notice cooldown (600s)
            assert should_send_quota_notice(22222, now=t0 + 300) is True
            assert should_send_quota_notice(22222, now=t0 + 350) is False
            assert should_send_quota_notice(22222, now=t0 + 901) is True

            # 3. After 2 hours (7201s from first message), oldest slot expires and user can chat again
            ok_after, rem_after, _ = check_and_consume_user_quota(22222, now=t0 + 7201)
            assert ok_after is True
            assert rem_after == 0  # 19 remaining inside window + 1 new = 20

            # 4. Refund restores slot cleanly
            refund_user_quota(22222)
            ok_refunded, _, _ = check_and_consume_user_quota(22222, now=t0 + 7202)
            assert ok_refunded is True

            # 5. Bot manager (ADMIN_IDS) has unlimited quota
            for i in range(30):
                ok_mgr, rem_mgr, _ = check_and_consume_user_quota(11111, now=t0 + i)
                assert ok_mgr is True
                assert rem_mgr == -1

            # 6. End-to-end handler blocks 21st message for user 33333 without calling LLM,
            #    and refunds on LLMError for user 44444
            bot = _make_bot()
            set_ai_bot_identity(999999, "PrimeTestBot")
            for i in range(20):
                assert check_and_consume_user_quota(33333)[0] is True

            blocked_msg = _make_message(
                user_id=33333,
                text="@PrimeTestBot سلام",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            with patch("bot.handlers.ai_chat.generate_ai_chat_reply") as mock_gen:
                await handle_ai_chat_message(blocked_msg, bot)
                mock_gen.assert_not_called()
            assert "سهمیه چت تو" in bot.send_message.call_args.kwargs["text"]

            # LLMError refunds user 44444's slot
            failing_msg = _make_message(
                user_id=44444,
                text="@PrimeTestBot سلام",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            with patch(
                "bot.handlers.ai_chat.generate_ai_chat_reply",
                side_effect=LLMError("temp", status_code=503, is_retryable=True),
            ):
                await handle_ai_chat_message(failing_msg, bot)
            # Slot was refunded, so remaining is still 20 (next call has remaining == 19)
            ok44, rem44, _ = check_and_consume_user_quota(44444)
            assert ok44 is True and rem44 == 19
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_ai_chat_dynamic_quota_set_reset_close_unlimited_and_fsm(tmp_path: Path) -> None:
    """Verify dynamic quota limit persistence, presets, custom FSM input, reset all, close (0), and unlimited (-1)."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_dynamic_quota.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        bot = _make_bot(bot_id=999999, bot_username="PrimeTestBot")
        set_ai_bot_identity(999999, "PrimeTestBot")
        fsm = _make_fsm()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_settings", return_value=cfg
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg):
            # 1. Default limit from settings (20), then DB persistence & cache rehydration for 50
            assert get_effective_ai_chat_user_limit() == 20
            await set_ai_chat_user_limit_state(50)
            assert get_effective_ai_chat_user_limit() == 50
            assert await repository.get_ai_chat_user_limit() == 50

            reset_ai_chat_state()
            assert get_effective_ai_chat_user_limit() == 20
            await init_ai_chat_state()
            assert get_effective_ai_chat_user_limit() == 50

            # 2. Regular user can consume 50 slots; 51st is blocked
            for _ in range(50):
                assert check_and_consume_user_quota(22222)[0] is True
            assert check_and_consume_user_quota(22222)[0] is False
            assert count_active_quota_users() == 1

            # 3. Reset all quotas via callback 'qreset' -> unblocks user immediately
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=11111, type="private")
            cb.message.message_id = 301
            cb.message.edit_text = AsyncMock()

            await on_manage_ai_callback(cb, AiChatCB(action="qreset", owner_id=11111), bot, fsm)
            reset_toast = cb.answer.call_args.kwargs.get("text") or cb.answer.call_args.args[0]
            assert "ریست شد" in reset_toast
            assert count_active_quota_users() == 0
            assert check_and_consume_user_quota(22222)[0] is True

            # 4. Callback 'qset' arg='0' closes regular-user quota, while manager remains unlimited
            await on_manage_ai_callback(cb, AiChatCB(action="qset", owner_id=11111, arg="0"), bot, fsm)
            assert get_effective_ai_chat_user_limit() == 0
            ok_closed, rem_closed, wait_closed = check_and_consume_user_quota(33333)
            assert (ok_closed, rem_closed, wait_closed) == (False, 0, -1)
            assert check_and_consume_user_quota(11111) == (True, -1, 0)

            closed_msg = _make_message(
                user_id=33333,
                text="@PrimeTestBot سلام",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            with patch("bot.handlers.ai_chat.generate_ai_chat_reply") as mock_gen:
                await handle_ai_chat_message(closed_msg, bot)
                mock_gen.assert_not_called()
            assert "بسته شده" in bot.send_message.call_args.kwargs["text"]

            # 5. Shorthand '/manage_ai quota ۵۰' and '/manage_ai quota reset'
            cmd_set = _make_message(user_id=11111, text="/manage_ai quota ۵۰", chat_type="private")
            await cmd_manage_ai(cmd_set, bot, fsm)
            assert get_effective_ai_chat_user_limit() == 50
            assert "50 پیام" in cmd_set.answer.call_args.kwargs["text"]

            check_and_consume_user_quota(44444)
            cmd_reset = _make_message(user_id=11111, text="/manage_ai quota reset", chat_type="private")
            await cmd_manage_ai(cmd_reset, bot, fsm)
            assert count_active_quota_users() == 0

            # 6. Custom FSM input: invalid -> warning, valid Persian digits -> sets limit, '/cancel' -> clears state
            await on_manage_ai_callback(cb, AiChatCB(action="qcustom", owner_id=11111), bot, fsm)
            fsm.set_state.assert_awaited_with(AiChatStates.waiting_quota_limit)

            bad_msg = _make_message(user_id=11111, text="abc", chat_type="private")
            await on_custom_quota_input(bad_msg, bot, fsm)
            assert "بین ۱ تا ۱۰۰۰۰" in bad_msg.answer.call_args.kwargs["text"]
            assert get_effective_ai_chat_user_limit() == 50

            good_msg = _make_message(user_id=11111, text="۷۵", chat_type="private")
            await on_custom_quota_input(good_msg, bot, fsm)
            assert get_effective_ai_chat_user_limit() == 75
            fsm.clear.assert_awaited()

            cancel_msg = _make_message(user_id=11111, text="/cancel", chat_type="private")
            await on_custom_quota_input(cancel_msg, bot, fsm)
            assert "لغو شد" in cancel_msg.answer.call_args.kwargs["text"]

            # 7. Unlimited mode (-1) via callback allows unlimited messages
            await on_manage_ai_callback(cb, AiChatCB(action="qset", owner_id=11111, arg="-1"), bot, fsm)
            assert get_effective_ai_chat_user_limit() == -1
            for _ in range(120):
                assert check_and_consume_user_quota(55555) == (True, -1, 0)
            assert reset_all_user_quotas() == 0
    finally:
        await close_db()
        reset_ai_chat_state()
