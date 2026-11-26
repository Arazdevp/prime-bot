"""Offline unit tests for AI chat live group awareness and long-term member personality learning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import aiosqlite
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, Message, User
from bot.database import repository
from bot.database.db import close_db, get_db, init_db
from bot.handlers.ai_chat import cmd_manage_ai, on_manage_ai_callback
from bot.keyboards.ai_chat import AiChatCB
from bot.scheduler.jobs import run_summary_for_topic
from bot.services.ai_chat_service import (
    generate_ai_chat_reply,
    reset_ai_chat_state,
    set_ai_chat_state,
)
from bot.services.ai_memory_service import (
    bind_memory_source_topic,
    ensure_memory_source_for_group,
    learn_from_topic_messages,
    maybe_learn_before_purge,
)
from bot.services.llm_client import GeminiClient


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
        AI_CHAT_MAX_OUTPUT_TOKENS=1024,
        AI_CHAT_TEMPERATURE=0.85,
        AI_CHAT_HISTORY_LIMIT=8,
        CHUNK_MAX_CHARS=24000,
        MIN_MESSAGES_PER_TOPIC=2,
        SEND_DELAY_SECONDS=0.0,
        AI_MAX_RETRIES=0,
        AI_CHUNK_DELAY_SECONDS=0.0,
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


def _make_group_msg(
    user_id: int,
    text: str,
    *,
    chat_id: int = -1002458921841,
    thread_id: int = 3,
    message_id: int = 700,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.text = text
    msg.caption = None
    msg.sender_chat = None
    msg.message_thread_id = thread_id
    msg.is_topic_message = True
    msg.reply_to_message = None

    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "supergroup"

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.first_name = "آرش"
    user.full_name = "آرش"
    user.username = "arash_owner"
    msg.from_user = user

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
async def test_migration_v7_to_v8_creates_ai_memory_tables_and_preserves_data(tmp_path: Path) -> None:
    """Upgrading from user_version=7 creates AI memory tables without losing existing rows."""
    db_file = tmp_path / "upgrade_v7_to_v8.db"
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
        CREATE TABLE group_settings (
            chat_id INTEGER PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'blacklist',
            destination_mode TEXT NOT NULL DEFAULT 'per_topic',
            destination_thread_id INTEGER,
            enabled INTEGER NOT NULL DEFAULT 1,
            min_messages INTEGER NOT NULL DEFAULT 5
        );
        INSERT INTO group_settings (chat_id, mode, min_messages) VALUES (-100888, 'whitelist', 9);
        PRAGMA user_version = 7;
        """
    )
    await conn.commit()
    await conn.close()

    try:
        await init_db(str(db_file))
        db = get_db()
        cur = await db.execute("PRAGMA user_version;")
        assert (await cur.fetchone())[0] >= 8

        gs = await repository.get_group_settings(-100888)
        assert gs is not None and gs["min_messages"] == 9

        # Verify new tables exist and work
        src = await repository.upsert_ai_memory_source(-100888, 3, "The Hub | گفتگوی آزاد")
        assert src["last_learned_message_id"] == 0
        await repository.upsert_ai_group_memory(-100888, "Hills Prime", "صمیمی و پرانرژی", "تایمر ۳ صبح", "یوتیوب")
        gm = await repository.get_ai_group_memory(-100888)
        assert gm is not None and gm["culture_summary"] == "صمیمی و پرانرژی"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_auto_detect_free_chat_topic_and_incremental_learning(tmp_path: Path) -> None:
    """Verify auto-detection of گفتگوی آزاد, JSON learning, hallucinated ID filtering, and incremental merge."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_mem_learn.db"))
    cfg = _fake_settings()
    try:
        chat_id = -1002458921841
        hub_tid = 3
        await repository.upsert_known_group(chat_id, "Hills Prime network")
        await repository.upsert_topic(chat_id, 2, "Wins | بُردها")
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")

        src = await ensure_memory_source_for_group(chat_id)
        assert src is not None
        assert src["thread_id"] == hub_tid

        # Insert 3 real user messages into گفتگوی آزاد
        msgs_batch1 = [
            (101, 501, "پیت", "بچه‌ها من امشب تا ۲ بیدارم دارم ادیت تامنیل می‌زنم 🎨"),
            (102, 502, "ستایش", "دمت گرم پیت! راستی من جزوه‌های دانشگاه رو هم مرتب کردم."),
            (103, 501, "پیت", "ایول ستایش، آخر شب نتیجه رو بفرست."),
        ]
        for mid, uid, name, txt in msgs_batch1:
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=hub_tid,
                message_id=mid,
                user_id=uid,
                sender_name=name,
                text=txt,
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at=f"2026-11-24T18:0{mid - 100}:00+00:00",
            )

        # Mock LLM returns fenced JSON including one valid member (501) and one hallucinated ID (999999)
        learn_json_1 = """```json
        {
          "group_culture": "جو صمیمی، پرکار و حمایتی بین اعضا.",
          "inside_jokes": "شوخی درباره بیدار ماندن تا ۲ شب برای ادیت.",
          "ongoing_topics": "طراحی تامنیل و درس‌های دانشگاه.",
          "members": [
            {
              "user_id": 501,
              "display_name": "پیت",
              "personality_summary": "پرانرژی، شب‌زنده‌دار و پیگیر کارهای گرافیکی.",
              "speaking_style": "خودمانی و پرانگیزه",
              "interests_and_topics": "طراحی تامنیل و ادیت",
              "notable_facts": "معمولاً شب‌ها تا دیروقت کار می‌کند."
            },
            {
              "user_id": 999999,
              "display_name": "شخص خیالی",
              "personality_summary": "نباید ذخیره شود"
            }
          ]
        }
        ```"""
        mock_llm = MagicMock(spec=GeminiClient)
        mock_llm.generate_chat = AsyncMock(return_value=learn_json_1)

        with patch("bot.services.ai_memory_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_settings", return_value=cfg
        ):
            res1 = await learn_from_topic_messages(chat_id, hub_tid, client=mock_llm)
            assert res1["messages_processed"] == 3
            assert res1["members_updated"] == 1
            assert res1["group_updated"] is True

            # Hallucinated user 999999 must be ignored; user 501 stored with message_count=2
            profiles = await repository.list_ai_member_profiles(chat_id)
            assert [p["user_id"] for p in profiles] == [501]
            assert profiles[0]["message_count"] == 2
            assert "شب‌زنده‌دار" in profiles[0]["personality_summary"]

            # Watermark updated to 103 -> calling again with no new messages does 0 LLM calls
            mock_llm.generate_chat.reset_mock()
            res_empty = await learn_from_topic_messages(chat_id, hub_tid, client=mock_llm)
            assert res_empty["messages_processed"] == 0
            mock_llm.generate_chat.assert_not_called()

            # Add new message 104 from user 502 and verify incremental prompt includes existing memory
            await repository.insert_message(
                chat_id=chat_id,
                thread_id=hub_tid,
                message_id=104,
                user_id=502,
                sender_name="ستایش",
                text="بچه‌ها کسی منابع معماری یا نوت‌برداری خوب سراغ داره؟",
                media_type=None,
                reply_to_message_id=None,
                reply_to_name=None,
                reply_to_snippet=None,
                created_at="2026-11-24T19:00:00+00:00",
            )
            mock_llm.generate_chat = AsyncMock(
                return_value='{"group_culture":"جو صمیمی و علمی","inside_jokes":"","ongoing_topics":"منابع درسی و ادیت","members":[{"user_id":502,"display_name":"ستایش","personality_summary":"منظم و پیگیر مباحث درسی و همکاری تیمی.","speaking_style":"محترمانه و صمیمی","interests_and_topics":"معماری و نوت‌برداری","notable_facts":""}]}'
            )
            res2 = await learn_from_topic_messages(chat_id, hub_tid, client=mock_llm)
            assert res2["messages_processed"] == 1
            assert res2["members_updated"] == 1
            prompt_sent = mock_llm.generate_chat.call_args.kwargs["messages"][0]["text"]
            assert "جو صمیمی، پرکار و حمایتی" in prompt_sent
            assert "[uid:502]" in prompt_sent
            assert "[uid:501]" not in prompt_sent
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_group_chat_reply_injects_group_presence_live_messages_and_member_profiles(tmp_path: Path) -> None:
    """Verify generate_ai_chat_reply injects group title, live topic msgs, group culture, and member profiles."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_group_aware.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        chat_id = -1002458921841
        hub_tid = 3
        await repository.upsert_known_group(chat_id, "Hills Prime network")
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await repository.upsert_ai_memory_source(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await repository.upsert_ai_group_memory(
            chat_id=chat_id,
            group_title="Hills Prime network",
            culture_summary="گروه فعال تولید محتوا با لحن رفاقتی.",
            inside_jokes="چالش دیپ‌ورک نیمه‌شب",
            ongoing_topics="آماده‌سازی ایونت حضوری",
        )
        await repository.upsert_ai_member_profile(
            chat_id=chat_id,
            user_id=501,
            display_name="پیت",
            personality_summary="خوش‌اخلاق، خلاق و پیگیر دیزاین.",
            speaking_style="راحت و خودمونی",
            interests_and_topics="تولید محتوا و تامنیل",
            notable_facts="هم‌بنیان‌گذار مجموعه است.",
            add_message_count=20,
        )
        await repository.insert_message(
            chat_id=chat_id,
            thread_id=hub_tid,
            message_id=201,
            user_id=501,
            sender_name="پیت",
            text="الان دارم روی کاور ویدیوی جدید کار می‌کنم",
            media_type=None,
            reply_to_message_id=None,
            reply_to_name=None,
            reply_to_snippet=None,
            created_at="2026-11-24T19:10:00+00:00",
        )

        captured_sys_prompts: list[str] = []

        async def capture_chat(system_prompt, messages, temperature=0.85, max_output_tokens=1024, model=None):
            captured_sys_prompts.append(system_prompt)
            return "آره پیت راست میگه، ایونت حضوری هم داره جذاب می‌شه!"

        mock_client = MagicMock(spec=GeminiClient)
        mock_client.generate_chat = AsyncMock(side_effect=capture_chat)

        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_memory_service.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_client):
            reply = await generate_ai_chat_reply(
                chat_id=chat_id,
                thread_id=hub_tid,
                chat_type="supergroup",
                speaker_name="آرش",
                speaker_user_id=11111,
                topic_name="The Hub | گفتگوی آزاد",
                user_text="بچه‌ها درباره چیا حرف می‌زنن امروز؟",
            )

        assert "ایونت حضوری" in reply
        sys_prompt = captured_sys_prompts[0]
        assert "Hills Prime network" in sys_prompt
        assert "تو الان داخل گروه تلگرامی" in sys_prompt
        assert "چالش دیپ‌ورک نیمه‌شب" in sys_prompt
        assert "خوش‌اخلاق، خلاق و پیگیر دیزاین" in sys_prompt
        assert "الان دارم روی کاور ویدیوی جدید کار می‌کنم" in sys_prompt
        # Verify speaker-tagged user turn in multi-member group mode
        assert mock_client.generate_chat.call_args.kwargs["messages"][-1]["text"].startswith("آرش:")
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_maybe_learn_before_purge_and_manage_ai_memory_commands(tmp_path: Path) -> None:
    """Verify pre-purge learning runs before DB deletion and /manage_ai bind|learn|memory + callbacks work."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_mem_cmd.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        for act in ("learn", "mem", "clrmem"):
            assert len(AiChatCB(action=act, owner_id=999999999999).pack().encode("utf-8")) <= 64

        chat_id = -100777
        hub_tid = 3
        await repository.upsert_known_group(chat_id, "Prime Test Group")
        await repository.ensure_group_settings(chat_id, default_min_messages=2)
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد", is_active=1)

        # When AI chat is OFF and topic is NOT bound, maybe_learn_before_purge stays idle
        sample_msgs = [
            {
                "id": i,
                "chat_id": chat_id,
                "thread_id": hub_tid,
                "message_id": 300 + i,
                "user_id": 600 + (i % 2),
                "sender_name": f"عضو {i % 2}",
                "text": f"پیام تستی خیلی مهم درباره تولید محتوا شماره {i}",
                "created_at": "2026-11-24T10:00:00+00:00",
            }
            for i in range(1, 7)
        ]
        assert await maybe_learn_before_purge(chat_id, hub_tid, "The Hub | گفتگوی آزاد", sample_msgs) is False

        # Enable AI chat and test /manage_ai bind inside topic
        bot = _make_bot()
        fsm = _make_fsm()
        await set_ai_chat_state(True)

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_memory_service.get_settings", return_value=cfg
        ):
            bind_msg = _make_group_msg(11111, "/manage_ai bind", chat_id=chat_id, thread_id=hub_tid)
            await cmd_manage_ai(bind_msg, bot, fsm)
            src = await repository.get_ai_memory_source(chat_id, hub_tid)
            assert src is not None
            assert "تاپیک یادگیری" in bot.send_message.call_args.kwargs["text"]

            # Insert those 6 messages into DB, then trigger pre-purge during run_summary_for_topic
            for m in sample_msgs:
                await repository.insert_message(
                    chat_id=chat_id,
                    thread_id=hub_tid,
                    message_id=m["message_id"],
                    user_id=m["user_id"],
                    sender_name=m["sender_name"],
                    text=m["text"],
                    media_type=None,
                    reply_to_message_id=None,
                    reply_to_name=None,
                    reply_to_snippet=None,
                    created_at=m["created_at"],
                )

            mock_chat_llm = MagicMock(spec=GeminiClient)
            mock_chat_llm.generate_chat = AsyncMock(
                return_value='{"group_culture":"جمع پرانرژی","inside_jokes":"ندی","ongoing_topics":"تولید محتوا","members":[{"user_id":600,"display_name":"عضو 0","personality_summary":"پرانرژی و اهل بحث تولید محتوا","speaking_style":"صمیمی","interests_and_topics":"محتوا","notable_facts":""}]}'
            )
            mock_sum_llm = AsyncMock()
            mock_sum_llm.generate = AsyncMock(return_value="📌 <b>خلاصه تاپیک</b>\n\n• بحث درباره تولید محتوا.")

            with patch("bot.services.ai_memory_service.get_ai_chat_client", return_value=mock_chat_llm), patch(
                "bot.scheduler.jobs.get_settings", return_value=cfg
            ), patch("bot.services.summarizer.get_settings", return_value=cfg), patch(
                "bot.services.publisher.get_settings", return_value=cfg
            ):
                ok, _ = await run_summary_for_topic(
                    bot=bot,
                    chat_id=chat_id,
                    thread_id=hub_tid,
                    mark_summarized=True,
                    is_nightly=False,
                    llm_client=mock_sum_llm,
                )
            assert ok is True
            # Messages were deleted after summary, BUT learned profile survived in ai_member_profiles!
            assert await repository.fetch_unsummarized_messages(chat_id, hub_tid) == []
            assert await repository.count_ai_member_profiles(chat_id) == 1

            # Test 'mem' and 'clrmem' inline callbacks
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=chat_id, type="supergroup")
            cb.message.message_id = 555
            cb.message.edit_text = AsyncMock()

            await on_manage_ai_callback(cb, AiChatCB(action="mem", owner_id=11111), bot, fsm)
            mem_view_text = cb.message.edit_text.call_args.kwargs["text"]
            assert "جمع پرانرژی" in mem_view_text
            assert "عضو 0" in mem_view_text

            await on_manage_ai_callback(cb, AiChatCB(action="clrmem", owner_id=11111), bot, fsm)
            assert await repository.count_ai_member_profiles(chat_id) == 0
    finally:
        await close_db()
        reset_ai_chat_state()


@pytest.mark.asyncio
async def test_ai_memory_paginated_member_profiles_view(tmp_path: Path) -> None:
    """Verify /manage_ai memory paginates ALL learned member profiles with full traits and safe message size."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_mem_paginated.db"))
    cfg = _fake_settings(admin_ids=[11111])
    try:
        chat_id = -100777
        hub_tid = 3
        await repository.upsert_known_group(chat_id, "Prime Paginated Group")
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد", is_active=1)
        await repository.upsert_ai_memory_source(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await repository.upsert_ai_group_memory(
            chat_id=chat_id,
            group_title="Prime Paginated Group",
            culture_summary="فرهنگ حمایتی و شوخ‌طبع.",
            inside_jokes="شوخی تایمر سحر",
            ongoing_topics="تولید محتوای یوتیوب",
        )

        # Insert 13 members (more than old 12-member cap) with all personality fields
        for i in range(1, 14):
            await repository.upsert_ai_member_profile(
                chat_id=chat_id,
                user_id=1000 + i,
                display_name=f"عضو شماره {i}",
                personality_summary=f"خلاصه شخصیت عضو {i} با روحیه پیگیر و خلاق " + ("توضیح اضافه " * 12),
                speaking_style=f"لحن صمیمی و باحال {i}",
                interests_and_topics=f"تدوین، یوتیوب و طراحی {i}",
                notable_facts=f"روی پروژه {i} کار می‌کند",
                add_message_count=100 - i,
            )

        bot = _make_bot()
        fsm = _make_fsm()
        cb = MagicMock(spec=CallbackQuery)
        cb.from_user = SimpleNamespace(id=11111)
        cb.answer = AsyncMock()
        cb.message = MagicMock(spec=Message)
        cb.message.chat = SimpleNamespace(id=chat_id, type="supergroup")
        cb.message.message_id = 555
        cb.message.edit_text = AsyncMock()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_memory_service.get_settings", return_value=cfg
        ):
            # Page 0: group culture + members 1..5, Next button, no Prev button
            await on_manage_ai_callback(cb, AiChatCB(action="mem", owner_id=11111, page=0), bot, fsm)
            p0_kwargs = cb.message.edit_text.call_args.kwargs
            p0_text = p0_kwargs["text"]
            p0_buttons = [btn for row in p0_kwargs["reply_markup"].inline_keyboard for btn in row]
            assert len(p0_text) <= 4096
            assert "13 نفر" in p0_text
            assert "صفحه 1/3" in p0_text
            assert "فرهنگ حمایتی" in p0_text
            assert "عضو شماره 1" in p0_text
            assert "عضو شماره 5" in p0_text
            assert "عضو شماره 6" not in p0_text
            assert "لحن صمیمی و باحال 1" in p0_text
            assert "تدوین، یوتیوب و طراحی 1" in p0_text
            assert "روی پروژه 1 کار می‌کند" in p0_text
            assert any(b.text == "بعدی ➡️" for b in p0_buttons)
            assert not any("قبلی" in b.text for b in p0_buttons)

            # Page 1: members 6..10, both Prev and Next buttons
            await on_manage_ai_callback(cb, AiChatCB(action="mem", owner_id=11111, page=1), bot, fsm)
            p1_kwargs = cb.message.edit_text.call_args.kwargs
            p1_text = p1_kwargs["text"]
            p1_buttons = [btn for row in p1_kwargs["reply_markup"].inline_keyboard for btn in row]
            assert len(p1_text) <= 4096
            assert "صفحه 2/3" in p1_text
            assert "عضو شماره 6" in p1_text
            assert "عضو شماره 10" in p1_text
            assert "عضو شماره 5" not in p1_text
            assert "عضو شماره 11" not in p1_text
            assert any("قبلی" in b.text for b in p1_buttons)
            assert any("بعدی" in b.text for b in p1_buttons)

            # Page 2: members 11..13 (beyond old 12-member cap!), Prev button, no Next button
            await on_manage_ai_callback(cb, AiChatCB(action="mem", owner_id=11111, page=2), bot, fsm)
            p2_kwargs = cb.message.edit_text.call_args.kwargs
            p2_text = p2_kwargs["text"]
            p2_buttons = [btn for row in p2_kwargs["reply_markup"].inline_keyboard for btn in row]
            assert len(p2_text) <= 4096
            assert "صفحه 3/3" in p2_text
            assert "عضو شماره 11" in p2_text
            assert "عضو شماره 13" in p2_text
            assert any("قبلی" in b.text for b in p2_buttons)
            assert not any("بعدی" in b.text for b in p2_buttons)

            # Out-of-range page clamps safely to last page (page 3/3)
            await on_manage_ai_callback(cb, AiChatCB(action="mem", owner_id=11111, page=99), bot, fsm)
            assert "صفحه 3/3" in cb.message.edit_text.call_args.kwargs["text"]
            assert "عضو شماره 13" in cb.message.edit_text.call_args.kwargs["text"]

            # Shorthand '/manage_ai memory 2' opens 1-based page 2 (0-based index 1)
            cmd_mem2 = _make_group_msg(11111, "/manage_ai memory 2", chat_id=chat_id, thread_id=hub_tid)
            await cmd_manage_ai(cmd_mem2, bot, fsm)
            assert "صفحه 2/3" in bot.send_message.call_args.kwargs["text"]
            assert "عضو شماره 6" in bot.send_message.call_args.kwargs["text"]
    finally:
        await close_db()
        reset_ai_chat_state()
