"""Offline unit tests for zero-token AI chat reactions and sticker/GIF/video_note media replies."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, Message, ReactionTypeEmoji, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.handlers.ai_chat import (
    AiChatStates,
    AiChatTriggerFilter,
    cmd_manage_ai,
    handle_ai_chat_message,
    on_chat_media_upload,
    on_manage_ai_callback,
)
from bot.handlers.listener import capture_message, clear_known_group_cache
from bot.keyboards.ai_chat import AiChatCB
from bot.services.ai_chat_service import (
    check_and_consume_user_quota,
    generate_ai_chat_reply,
    init_ai_chat_state,
    reset_ai_chat_state,
    set_ai_bot_identity,
    set_ai_chat_state,
)
from bot.services.ai_media_reaction_service import (
    ALLOWED_AI_REACTIONS,
    clamp_ai_reaction,
    classify_zero_token_reaction,
    detect_explicit_media_request,
    is_ai_chat_media_autocollect_cached,
    parse_ai_chat_directives,
    send_ai_reaction,
)
from bot.services.llm_client import GeminiClient
from bot.texts.ai_chat_texts import build_runtime_system_prompt


def _fake_settings(admin_ids: list[int] | None = None):
    return SimpleNamespace(
        ADMIN_IDS=admin_ids if admin_ids is not None else [11111],
        ALLOWED_CHAT_IDS=[],
        AI_CHAT_PROVIDER="gemini",
        AI_CHAT_API_KEY="AQ.TestKeyForUnitTests",
        GEMINI_API_KEY=None,
        AI_CHAT_MODEL="gemini-3.5-flash-lite",
        AI_CHAT_FALLBACK_MODELS=["gemini-flash-lite-latest"],
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
        MIN_MESSAGE_LENGTH=3,
        TIMEZONE="Asia/Tehran",
        PROXY_URL=None,
    )


def _make_fsm(initial_data: dict | None = None) -> FSMContext:
    state = MagicMock(spec=FSMContext)
    store = dict(initial_data or {})
    state.get_state = AsyncMock(return_value=None)
    state.clear = AsyncMock(side_effect=lambda: store.clear())
    state.set_state = AsyncMock()
    state.get_data = AsyncMock(side_effect=lambda: dict(store))
    state.update_data = AsyncMock(side_effect=lambda **kw: store.update(kw))
    return state


def _make_message(
    user_id: int,
    text: str | None = None,
    *,
    caption: str | None = None,
    chat_id: int | None = None,
    chat_type: str = "private",
    thread_id: int | None = None,
    message_id: int = 500,
    reply_from_user_id: int | None = None,
    reply_text: str | None = None,
    first_name: str = "آرش",
    sticker: object | None = None,
    animation: object | None = None,
    video_note: object | None = None,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.text = text
    msg.caption = caption
    msg.sender_chat = None
    msg.message_thread_id = thread_id
    msg.is_topic_message = thread_id is not None
    msg.sticker = sticker
    msg.animation = animation
    msg.video_note = video_note
    msg.photo = None
    msg.video = None
    msg.document = None
    msg.voice = None
    msg.audio = None
    msg.forum_topic_created = None

    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id if chat_id is not None else user_id
    msg.chat.type = chat_type
    msg.chat.title = "Hills Prime"
    msg.chat.is_forum = chat_type == "supergroup"

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.first_name = first_name
    user.full_name = f"{first_name} تست"
    user.username = "user_test"
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
    bot.set_message_reaction = AsyncMock(return_value=True)
    bot.send_sticker = AsyncMock()
    bot.send_animation = AsyncMock()
    bot.send_video_note = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_migration_v9_to_v10_and_media_repository_crud(tmp_path: Path) -> None:
    """Upgrading from user_version=9 creates ai_chat_media while preserving existing data and CRUD works."""
    db_file = tmp_path / "upgrade_v9_to_v10.db"
    conn = await aiosqlite.connect(str(db_file))
    await conn.executescript(
        """
        CREATE TABLE topics (
            chat_id INTEGER NOT NULL,
            thread_id INTEGER NOT NULL,
            name TEXT,
            is_active INTEGER,
            updated_at TEXT,
            PRIMARY KEY(chat_id, thread_id)
        );
        INSERT INTO topics (chat_id, thread_id, name, is_active, updated_at)
        VALUES (-100111, 3, 'The Hub | گفتگوی آزاد', 1, '2026-01-01T00:00:00+00:00');
        PRAGMA user_version = 9;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        ver = (await cur.fetchone())[0]
        assert ver >= 10

        # Existing topic row preserved
        assert await repository.get_topic_name(-100111, 3) == "The Hub | گفتگوی آزاد"

        # Default auto-collect is True
        assert await repository.get_ai_chat_media_autocollect() is True
        await repository.set_ai_chat_media_autocollect(False)
        assert await repository.get_ai_chat_media_autocollect() is False

        # Upsert auto item, then upgrade same unique_id via manual
        row_id1, is_new1 = await repository.upsert_ai_chat_media(
            media_type="sticker",
            file_id="fid_auto_1",
            file_unique_id="uid_1",
            mood="general",
            emoji="😂",
            source="auto",
            chat_id=-100111,
        )
        assert is_new1 is True

        row_id2, is_new2 = await repository.upsert_ai_chat_media(
            media_type="sticker",
            file_id="fid_manual_1",
            file_unique_id="uid_1",
            mood="laugh",
            emoji="😂",
            source="manual",
            added_by=11111,
        )
        assert is_new2 is False
        assert row_id2 == row_id1

        cands = await repository.list_ai_chat_media_candidates("sticker", mood="laugh")
        assert len(cands) == 1
        assert cands[0]["source"] == "manual"
        assert cands[0]["file_id"] == "fid_manual_1"

        # Mark used increments use_count
        await repository.mark_ai_chat_media_used(row_id1)
        cands_after = await repository.list_ai_chat_media_candidates("sticker", mood="laugh")
        assert cands_after[0]["use_count"] == 1

        # Auto-collect pruning never deletes manual items when exceeding cap
        for idx in range(repository.AI_CHAT_AUTO_MEDIA_MAX_PER_TYPE + 5):
            await repository.upsert_ai_chat_media(
                media_type="sticker",
                file_id=f"fid_a_{idx}",
                file_unique_id=f"uid_a_{idx}",
                mood="general",
                source="auto",
            )
        src_counts = await repository.count_ai_chat_media_by_source()
        assert src_counts["manual"] == 1
        assert src_counts["auto"] == repository.AI_CHAT_AUTO_MEDIA_MAX_PER_TYPE

        # Clear auto keeps manual intact
        deleted_auto = await repository.clear_ai_chat_media(source="auto")
        assert deleted_auto == repository.AI_CHAT_AUTO_MEDIA_MAX_PER_TYPE
        assert (await repository.count_ai_chat_media_by_type())["sticker"] == 1
    finally:
        await close_db()
        reset_ai_chat_state()


def test_zero_token_reaction_classifier_accuracy_and_guardrails() -> None:
    """Verify emoji-only & no-reply messages map strictly to 👍/👎/😁 while questions/greetings pass to LLM."""
    # Emoji-only
    assert classify_zero_token_reaction("😂😂") == "😁"
    assert classify_zero_token_reaction("🤣🔥") == "😁"
    assert classify_zero_token_reaction("👍") == "👍"
    assert classify_zero_token_reaction("❤️🔥") == "👍"
    assert classify_zero_token_reaction("👎") == "👎"
    assert classify_zero_token_reaction("😡😒") == "👎"

    # Persian & English no-reply acknowledgments and stretched variants
    assert classify_zero_token_reaction("آفرین") == "👍"
    assert classify_zero_token_reaction("افرینننن") == "👍"
    assert classify_zero_token_reaction("دقیقاً") == "👍"
    assert classify_zero_token_reaction("مرسیییی") == "👍"
    assert classify_zero_token_reaction("مرسی رفیق") == "👍"
    assert classify_zero_token_reaction("دمت گرم داش") == "👍"
    assert classify_zero_token_reaction("اوکی حله") == "👍"
    assert classify_zero_token_reaction("آفرین دقیقا") == "👍"
    assert classify_zero_token_reaction("دقیقا 😂") == "😁"
    assert classify_zero_token_reaction("خخخخ") == "😁"
    assert classify_zero_token_reaction("ههههه") == "😁"
    assert classify_zero_token_reaction("lol") == "😁"
    assert classify_zero_token_reaction("ok") == "👍"
    assert classify_zero_token_reaction("thanks") == "👍"

    # Negative no-reply acknowledgments
    assert classify_zero_token_reaction("نه") == "👎"
    assert classify_zero_token_reaction("اصلا") == "👎"
    assert classify_zero_token_reaction("موافق نیستم") == "👎"
    assert classify_zero_token_reaction("بد بود") == "👎"

    # Guardrails: NEVER swallow greetings, questions, bare vocatives, or substantive text
    assert classify_zero_token_reaction("سلام") is None
    assert classify_zero_token_reaction("خوبی") is None
    assert classify_zero_token_reaction("چطوری") is None
    assert classify_zero_token_reaction("دقیقا؟") is None
    assert classify_zero_token_reaction("مرسی، چیکار کنم؟") is None
    assert classify_zero_token_reaction("👍؟") is None
    assert classify_zero_token_reaction("داش") is None
    assert classify_zero_token_reaction("") is None
    assert classify_zero_token_reaction("بنظرت یوتیوب رو از کجا شروع کنم؟") is None


@pytest.mark.asyncio
async def test_send_ai_reaction_strictly_clamped_and_resilient() -> None:
    """Reactions are clamped to ('👍', '👎', '😁') and Telegram errors never raise."""
    assert clamp_ai_reaction("❤️") == "👍"
    assert clamp_ai_reaction("😡") == "👎"
    assert clamp_ai_reaction("😂") == "😁"
    for r in ALLOWED_AI_REACTIONS:
        assert clamp_ai_reaction(r) == r

    bot = _make_bot()
    ok = await send_ai_reaction(bot, -100555, 42, "🔥")
    assert ok is True
    sent_reactions = bot.set_message_reaction.call_args.kwargs["reaction"]
    assert isinstance(sent_reactions[0], ReactionTypeEmoji)
    assert sent_reactions[0].emoji == "👍"

    bot.set_message_reaction = AsyncMock(
        side_effect=TelegramBadRequest(method=MagicMock(), message="REACTION_INVALID")
    )
    ok_fail = await send_ai_reaction(bot, -100555, 42, "😁")
    assert ok_fail is False


@pytest.mark.asyncio
async def test_zero_token_handler_fast_paths_emoji_ack_and_explicit_media(tmp_path: Path) -> None:
    """Verify emoji, no-reply ack, media-only, and explicit media requests consume 0 LLM tokens & 0 quota."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_zero_token.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        chat_id = -100555
        hub_tid = 3
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await set_ai_chat_state(True)
        set_ai_bot_identity(999999, "PrimeTestBot")
        bot = _make_bot()

        await repository.upsert_ai_chat_media(
            media_type="sticker",
            file_id="stk_hype_1",
            file_unique_id="ustk_1",
            mood="general",
            emoji="🔥",
            source="manual",
        )
        await repository.upsert_ai_chat_media(
            media_type="animation",
            file_id="gif_1",
            file_unique_id="ugif_1",
            mood="laugh",
            source="manual",
        )
        await repository.upsert_ai_chat_media(
            media_type="video_note",
            file_id="vn_1",
            file_unique_id="uvn_1",
            mood="general",
            source="manual",
        )

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.generate_ai_chat_reply"
        ) as mock_llm:
            # 1. Emoji-only message -> reaction 😁, 0 LLM calls, 0 quota
            m_emoji = _make_message(
                user_id=22222,
                text="😂🔥",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=101,
                reply_from_user_id=999999,
            )
            await handle_ai_chat_message(m_emoji, bot)
            mock_llm.assert_not_called()
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "😁"
            bot.send_message.assert_not_called()

            # 2. No-reply Persian word ('آفرین') -> reaction 👍, 0 LLM calls
            m_ack = _make_message(
                user_id=22222,
                text="@PrimeTestBot آفرین دقیقا",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=102,
            )
            await handle_ai_chat_message(m_ack, bot)
            mock_llm.assert_not_called()
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "👍"

            # 3. Incoming sticker replying to bot -> passes trigger filter, reacts + sends sticker (0 LLM calls)
            flt = AiChatTriggerFilter()
            stk_obj = SimpleNamespace(file_id="in_stk_1", file_unique_id="in_ustk_1", emoji="😂")
            m_stk = _make_message(
                user_id=22222,
                text=None,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=103,
                reply_from_user_id=999999,
                sticker=stk_obj,
            )
            assert await flt(m_stk, bot) is True
            await handle_ai_chat_message(m_stk, bot)
            mock_llm.assert_not_called()
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "😁"
            bot.send_sticker.assert_awaited_once()

            # 4. Explicit media requests -> 0 LLM calls
            assert detect_explicit_media_request("یه استیکر بده") == "sticker"
            assert detect_explicit_media_request("یه گیف بفرست") == "animation"
            assert detect_explicit_media_request("یه ویدیو مسیج بده") == "video_note"

            m_req_gif = _make_message(
                user_id=22222,
                text="@PrimeTestBot یه گیف بفرست",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=104,
            )
            await handle_ai_chat_message(m_req_gif, bot)
            bot.send_animation.assert_awaited_once()
            mock_llm.assert_not_called()

            m_req_vn = _make_message(
                user_id=22222,
                text="@PrimeTestBot ویدیو مسیج بده",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=105,
            )
            await handle_ai_chat_message(m_req_vn, bot)
            bot.send_video_note.assert_awaited_once()
            mock_llm.assert_not_called()

            # Non-admin user quota remained completely untouched (full 20 remaining)
            ok, rem, _ = check_and_consume_user_quota(22222)
            assert ok is True and rem == 19
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_llm_directives_prompt_gating_and_handler_dispatch(tmp_path: Path) -> None:
    """Verify prompt includes only available media types and handler executes [REACT] / [STICKER] / [GIF] / [VIDEONOTE]."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_directives.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        # Prompt gating
        prompt_none = build_runtime_system_prompt(
            "پرامپت پایه",
            speaker_name="آرش",
            chat_type="private",
            available_media_types=[],
        )
        assert "[REACT:👍]" in prompt_none
        assert "[STICKER:" not in prompt_none
        assert "[GIF:" not in prompt_none
        assert "[VIDEONOTE:" not in prompt_none

        prompt_some = build_runtime_system_prompt(
            "پرامپت پایه",
            speaker_name="آرش",
            chat_type="private",
            available_media_types=["sticker", "video_note"],
        )
        assert "[STICKER:mood]" in prompt_some
        assert "[VIDEONOTE:mood]" in prompt_some
        assert "[GIF:mood]" not in prompt_some

        # Parser tests
        p1 = parse_ai_chat_directives("ایول داش!\n[STICKER:hype]")
        assert p1.text == "ایول داش!"
        assert p1.media_type == "sticker"
        assert p1.media_mood == "hype"

        p2 = parse_ai_chat_directives("[REACT:😂]")
        assert p2.text == ""
        assert p2.reaction == "😁"

        # Populate media bank
        await repository.upsert_ai_chat_media(
            media_type="sticker", file_id="stk_1", file_unique_id="u_stk_1", mood="hype", source="manual"
        )
        await repository.upsert_ai_chat_media(
            media_type="animation", file_id="gif_1", file_unique_id="u_gif_1", mood="laugh", source="manual"
        )
        await repository.upsert_ai_chat_media(
            media_type="video_note", file_id="vn_1", file_unique_id="u_vn_1", mood="general", source="manual"
        )
        await init_ai_chat_state()

        bot = _make_bot()
        set_ai_bot_identity(999999, "PrimeTestBot")

        mock_client = MagicMock(spec=GeminiClient)
        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_client
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg), patch(
            "bot.services.access.get_settings", return_value=cfg
        ):
            # Case A: LLM returns [REACT:😁] only -> set_message_reaction, no send_message
            mock_client.generate_chat = AsyncMock(return_value="[REACT:😁]")
            msg_a = _make_message(user_id=11111, text="یه چیزی گفتم", chat_type="private", message_id=201)
            await handle_ai_chat_message(msg_a, bot)
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "😁"
            bot.send_message.assert_not_called()

            # Case B: LLM returns text + [STICKER:hype] -> sends text AND sticker
            mock_client.generate_chat = AsyncMock(return_value="بنازم رفیق!\n[STICKER:hype]")
            msg_b = _make_message(
                user_id=11111,
                text="@PrimeTestBot چطور بود؟",
                chat_id=-100555,
                chat_type="supergroup",
                thread_id=3,
                message_id=202,
            )
            await handle_ai_chat_message(msg_b, bot)
            assert bot.send_message.call_args.kwargs["text"] == "بنازم رفیق!"
            assert bot.send_sticker.call_args.kwargs["sticker"] == "stk_1"

            # Case C: LLM returns [GIF:laugh] only -> sends animation only
            mock_client.generate_chat = AsyncMock(return_value="[GIF:laugh]")
            msg_c = _make_message(user_id=11111, text="یه جوک گفتم", chat_type="private", message_id=203)
            await handle_ai_chat_message(msg_c, bot)
            assert bot.send_animation.call_args.kwargs["animation"] == "gif_1"

            # Case D: LLM returns [VIDEONOTE:general] only -> sends video_note only
            mock_client.generate_chat = AsyncMock(return_value="[VIDEONOTE:general]")
            msg_d = _make_message(user_id=11111, text="بیا بالا", chat_type="private", message_id=204)
            await handle_ai_chat_message(msg_d, bot)
            assert bot.send_video_note.call_args.kwargs["video_note"] == "vn_1"
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_listener_passive_autocollect_and_manage_ai_media_ui(tmp_path: Path) -> None:
    """Verify listener auto-collects stickers/GIFs from free-chat topic only, respects toggle, and FSM upload works."""
    reset_ai_chat_state()
    clear_known_group_cache()
    await init_db(str(tmp_path / "ai_media_ui.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        chat_id = -100555
        hub_tid = 3
        wins_tid = 9
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await repository.upsert_topic(chat_id, wins_tid, "Wins | بُردها")
        await init_ai_chat_state()
        assert is_ai_chat_media_autocollect_cached() is True

        with patch("bot.services.ai_media_reaction_service.get_settings", return_value=cfg), patch(
            "bot.services.access.get_settings", return_value=cfg
        ), patch("bot.handlers.ai_chat.get_settings", return_value=cfg):
            # 1. Sticker in گفتگوی آزاد -> auto-collected
            msg_hub_stk = _make_message(
                user_id=22222,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=301,
                sticker=SimpleNamespace(file_id="hub_stk_1", file_unique_id="uhub_stk_1", emoji="🔥"),
            )
            await capture_message(msg_hub_stk)
            counts = await repository.count_ai_chat_media_by_type()
            assert counts["sticker"] == 1

            # 2. Sticker in Wins topic -> NOT collected
            msg_wins_stk = _make_message(
                user_id=22222,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=wins_tid,
                message_id=302,
                sticker=SimpleNamespace(file_id="wins_stk_1", file_unique_id="uwins_stk_1", emoji="👍"),
            )
            await capture_message(msg_wins_stk)
            assert (await repository.count_ai_chat_media_by_type())["sticker"] == 1

            # 3. Video note in گفتگوی آزاد -> NEVER auto-collected (privacy)
            msg_hub_vn = _make_message(
                user_id=22222,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=303,
                video_note=SimpleNamespace(file_id="hub_vn_1", file_unique_id="uhub_vn_1"),
            )
            await capture_message(msg_hub_vn)
            assert (await repository.count_ai_chat_media_by_type())["video_note"] == 0

            # 4. Callback UI: open media panel, toggle auto-collect OFF, start upload FSM, upload video_note
            bot = _make_bot()
            fsm = _make_fsm()
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=11111, type="private")
            cb.message.message_id = 555
            cb.message.edit_text = AsyncMock()

            await on_manage_ai_callback(cb, AiChatCB(action="media", owner_id=11111), bot, fsm)
            assert "مدیریت رسانه" in cb.message.edit_text.call_args.kwargs["text"]

            await on_manage_ai_callback(cb, AiChatCB(action="mauto", owner_id=11111), bot, fsm)
            assert is_ai_chat_media_autocollect_cached() is False

            # GIF sent while auto-collect is OFF -> NOT collected
            msg_hub_gif = _make_message(
                user_id=22222,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=304,
                animation=SimpleNamespace(file_id="hub_gif_1", file_unique_id="uhub_gif_1"),
            )
            await capture_message(msg_hub_gif)
            assert (await repository.count_ai_chat_media_by_type())["animation"] == 0

            # Owner picks mood & uploads a video_note via FSM
            await on_manage_ai_callback(cb, AiChatCB(action="madd", owner_id=11111, arg="hype"), bot, fsm)
            fsm.set_state.assert_awaited_with(AiChatStates.waiting_chat_media)

            upload_vn = _make_message(
                user_id=11111,
                chat_type="private",
                video_note=SimpleNamespace(file_id="owner_vn_1", file_unique_id="uowner_vn_1"),
            )
            await on_chat_media_upload(upload_vn, bot, fsm)
            assert (await repository.count_ai_chat_media_by_type())["video_note"] == 1

            # Test button sends video_note
            await on_manage_ai_callback(cb, AiChatCB(action="mtest", owner_id=11111, arg="video_note"), bot, fsm)
            bot.send_video_note.assert_awaited_once()

            # '/manage_ai media' shorthand opens media panel
            cmd_msg = _make_message(user_id=11111, text="/manage_ai media", chat_type="private")
            await cmd_manage_ai(cmd_msg, bot, fsm)
            assert "مدیریت رسانه" in cmd_msg.answer.call_args.kwargs["text"]

        # Verify all new CallbackData actions stay <= 64 bytes
        for action, arg in (
            ("media", ""),
            ("madd", "pick"),
            ("madd", "disagree"),
            ("mdone", ""),
            ("mauto", ""),
            ("mtest", "video_note"),
            ("mclr", "auto"),
            ("mclr", "all"),
        ):
            packed = AiChatCB(action=action, owner_id=999999999999, arg=arg, page=0).pack()
            assert len(packed.encode("utf-8")) <= 64
    finally:
        await close_db()
        reset_ai_chat_state()
        clear_known_group_cache()
