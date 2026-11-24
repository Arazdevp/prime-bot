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
    cmd_manage_ai,
    handle_ai_chat_message,
    on_custom_prompt_input,
    on_manage_ai_callback,
)
from bot.keyboards.ai_chat import AiChatCB
from bot.services.ai_chat_service import (
    clear_conversation_history,
    count_active_conversations,
    generate_ai_chat_reply,
    init_ai_chat_state,
    is_ai_chat_enabled_cached,
    reset_ai_chat_state,
    set_ai_bot_identity,
    set_ai_chat_state,
)
from bot.services.llm_client import GeminiClient, LLMError


def _fake_settings(admin_ids: list[int] | None = None, allowed_chats: list[int] | None = None):
    return SimpleNamespace(
        ADMIN_IDS=admin_ids if admin_ids is not None else [11111],
        ALLOWED_CHAT_IDS=allowed_chats if allowed_chats is not None else [],
        AI_CHAT_API_KEY="AQ.TestKeyForUnitTests",
        GEMINI_API_KEY=None,
        AI_CHAT_MODEL="gemini-3.5-flash-lite",
        AI_CHAT_FALLBACK_MODELS=["gemini-flash-lite-latest", "gemini-3.8-flash"],
        AI_CHAT_MAX_OUTPUT_TOKENS=1024,
        AI_CHAT_TEMPERATURE=0.85,
        AI_CHAT_HISTORY_LIMIT=6,
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
    for action in ("toggle", "clear", "refresh", "viewp", "setp", "resetp", "cancel", "close"):
        packed = AiChatCB(action=action, owner_id=999999999999).pack()
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
    """Verify AiChatTriggerFilter enforces toggle, owner-only access, and group reply/@mention gating."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_trigger.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        bot = _make_bot(bot_id=999999, bot_username="PrimeTestBot")
        set_ai_bot_identity(999999, "PrimeTestBot")
        flt = AiChatTriggerFilter()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ):
            # 1. When OFF: even owner in PV is blocked
            owner_pv = _make_message(user_id=11111, text="سلام رفیق", chat_type="private")
            assert await flt(owner_pv, bot) is False

            # Turn ON
            await set_ai_chat_state(True)

            # 2. Owner in PV -> allowed; commands -> blocked
            assert await flt(owner_pv, bot) is True
            owner_cmd = _make_message(user_id=11111, text="/panel", chat_type="private")
            assert await flt(owner_cmd, bot) is False

            # 3. Non-owner in PV and group -> always blocked
            other_pv = _make_message(user_id=22222, text="سلام", chat_type="private")
            assert await flt(other_pv, bot) is False
            other_grp_mention = _make_message(
                user_id=22222,
                text="@PrimeTestBot چطوری؟",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            assert await flt(other_grp_mention, bot) is False

            # 4. Owner in group without reply/mention -> blocked (doesn't interrupt normal chat)
            owner_grp_plain = _make_message(
                user_id=11111,
                text="بچه‌ها جلسه ساعت ۹ شروع می‌شه",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            assert await flt(owner_grp_plain, bot) is False

            # 5. Owner in group mentioning @PrimeTestBot (case-insensitive) -> allowed
            owner_grp_mention = _make_message(
                user_id=11111,
                text="سلام @primetestbot نظرت چیه؟",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
            )
            assert await flt(owner_grp_mention, bot) is True

            # 6. Owner in group replying to bot's own message -> allowed
            owner_grp_reply_bot = _make_message(
                user_id=11111,
                text="دقیق‌تر بگو",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
                reply_from_user_id=999999,
                reply_text="پیام قبلی ربات",
            )
            assert await flt(owner_grp_reply_bot, bot) is True
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
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg):
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

            # LLMError handled gracefully without crashing
            mock_client.generate_chat = AsyncMock(side_effect=LLMError("API down", status_code=503, is_retryable=False))
            pv_msg = _make_message(user_id=11111, text="سلام", chat_type="private")
            await handle_ai_chat_message(pv_msg, bot)
            assert "وصل نشدم" in pv_msg.answer.call_args.kwargs["text"]

            assert clear_conversation_history() >= 1
            assert count_active_conversations() == 0
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
