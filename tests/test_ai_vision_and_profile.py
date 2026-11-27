"""Offline unit tests for AI chat Vision (GIF/photo/sticker) and on-demand profile/bio inspection."""

import base64
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
    AiChatTriggerFilter,
    handle_ai_chat_message,
    on_manage_ai_callback,
)
from bot.keyboards.ai_chat import AiChatCB
from bot.services.ai_chat_service import (
    _conversations,
    generate_ai_chat_reply,
    init_ai_chat_state,
    reset_ai_chat_state,
    set_ai_bot_identity,
    set_ai_chat_state,
)
from bot.services.ai_vision_service import (
    detect_profile_inspection_intent,
    extract_visual_attachment_from_message,
    fetch_user_profile_context,
    is_ai_chat_vision_enabled_cached,
    set_ai_chat_vision_enabled_state,
)
from bot.services.llm_client import GeminiClient, OpenAIClient


_SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
_SAMPLE_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def _fake_settings(admin_ids: list[int] | None = None, provider: str = "gemini"):
    return SimpleNamespace(
        ADMIN_IDS=admin_ids if admin_ids is not None else [11111],
        ALLOWED_CHAT_IDS=[],
        AI_CHAT_PROVIDER=provider,
        AI_CHAT_API_KEY="AQ.TestKeyForUnitTests",
        GEMINI_API_KEY=None,
        AI_CHAT_MODEL="gemini-3.5-flash-lite",
        AI_CHAT_FALLBACK_MODELS=["gemini-flash-lite-latest"],
        AI_CHAT_NINEROUTER_BASE_URL=None,
        AI_CHAT_NINEROUTER_API_KEY=None,
        AI_CHAT_NINEROUTER_MODEL=None,
        NINEROUTER_BASE_URL=None,
        NINEROUTER_API_KEY=None,
        NINEROUTER_MODEL=None,
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
    text: str | None = None,
    *,
    caption: str | None = None,
    chat_id: int | None = None,
    chat_type: str = "private",
    thread_id: int | None = None,
    message_id: int = 600,
    reply_from_user_id: int | None = None,
    reply_from_is_bot: bool = False,
    reply_from_name: str = "سارا",
    reply_text: str | None = None,
    reply_animation: object | None = None,
    first_name: str = "آرش",
    photo: list[object] | None = None,
    animation: object | None = None,
    sticker: object | None = None,
) -> Message:
    msg = MagicMock(spec=Message)
    msg.message_id = message_id
    msg.text = text
    msg.caption = caption
    msg.sender_chat = None
    msg.message_thread_id = thread_id
    msg.is_topic_message = thread_id is not None
    msg.photo = photo
    msg.animation = animation
    msg.sticker = sticker
    msg.video = None
    msg.video_note = None
    msg.document = None
    msg.voice = None
    msg.audio = None
    msg.forum_topic_created = None

    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id if chat_id is not None else user_id
    msg.chat.type = chat_type

    user = MagicMock(spec=User)
    user.id = user_id
    user.is_bot = False
    user.first_name = first_name
    user.full_name = f"{first_name} تست"
    user.username = f"user_{user_id}"
    msg.from_user = user

    if reply_from_user_id is not None:
        reply = MagicMock(spec=Message)
        reply.forum_topic_created = None
        reply.text = reply_text
        reply.caption = None
        reply.photo = None
        reply.animation = reply_animation
        reply.sticker = None
        reply.video = None
        reply.video_note = None
        reply_user = MagicMock(spec=User)
        reply_user.id = reply_from_user_id
        reply_user.is_bot = reply_from_is_bot
        reply_user.first_name = reply_from_name
        reply_user.full_name = reply_from_name
        reply_user.username = f"user_{reply_from_user_id}"
        reply.from_user = reply_user
        msg.reply_to_message = reply
    else:
        msg.reply_to_message = None

    msg.answer = AsyncMock()
    msg.delete = AsyncMock()
    return msg


def _make_bot(
    *,
    bot_id: int = 999999,
    bot_username: str = "PrimeTestBot",
    file_bytes_map: dict[str, bytes] | None = None,
) -> Bot:
    bot = AsyncMock(spec=Bot)
    bot.id = bot_id
    bot.get_me = AsyncMock(return_value=SimpleNamespace(id=bot_id, username=bot_username))
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.set_message_reaction = AsyncMock(return_value=True)
    bot.send_sticker = AsyncMock()
    bot.send_animation = AsyncMock()
    bot.send_video_note = AsyncMock()

    fmap = file_bytes_map or {}

    async def _fake_download(file_id, destination=None, **kwargs):
        data = fmap.get(str(file_id), _SAMPLE_JPEG)
        if destination is not None:
            destination.write(data)
        return destination

    bot.download = AsyncMock(side_effect=_fake_download)
    bot.get_chat = AsyncMock(
        return_value=SimpleNamespace(
            bio="یوتیوبر و طراح تامبنیل | دنبال ساختن ویدیوهای خفن",
            username="arash_bio",
            first_name="آرش",
            last_name="راد",
        )
    )
    bot.get_user_profile_photos = AsyncMock(
        return_value=SimpleNamespace(
            photos=[[SimpleNamespace(file_id="pfp_fid_640", width=640, file_size=24000)]]
        )
    )
    return bot


@pytest.mark.asyncio
async def test_multimodal_gemini_and_openai_clients_and_history_pruning(tmp_path: Path) -> None:
    """Verify inlineData / image_url payloads, OpenAI 400 vision fallback, and history excluding attachments."""
    b64_img = base64.b64encode(_SAMPLE_JPEG).decode("ascii")

    # 1. GeminiClient encodes inlineData
    mock_gem_resp = MagicMock(spec=httpx.Response)
    mock_gem_resp.status_code = 200
    mock_gem_resp.json.return_value = {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "استایلت ۹ از ۱۰ شد!"}]}}]
    }
    mock_http_gem = AsyncMock(spec=httpx.AsyncClient)
    mock_http_gem.is_closed = False
    mock_http_gem.post = AsyncMock(return_value=mock_gem_resp)

    gem_client = GeminiClient(api_key="AQ.test", model="gemini-3.5-flash-lite", http_client=mock_http_gem)
    gem_reply = await gem_client.generate_chat(
        system_prompt="پرامپت",
        messages=[{"role": "user", "text": "چطوره؟", "attachments": [{"mime_type": "image/jpeg", "data_b64": b64_img}]}],
    )
    assert "۹ از ۱۰" in gem_reply
    parts = mock_http_gem.post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert parts[0] == {"text": "چطوره؟"}
    assert parts[1]["inlineData"] == {"mimeType": "image/jpeg", "data": b64_img}

    # 2. OpenAIClient encodes image_url and falls back to text-only on HTTP 400
    err_400 = MagicMock(spec=httpx.Response)
    err_400.status_code = 400
    err_400.text = '{"error":{"message":"image input not supported"}}'
    ok_200 = MagicMock(spec=httpx.Response)
    ok_200.status_code = 200
    ok_200.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "با متن جواب دادم!"}}]}

    mock_http_oai = AsyncMock(spec=httpx.AsyncClient)
    mock_http_oai.is_closed = False
    mock_http_oai.post = AsyncMock(side_effect=[err_400, ok_200])

    oai_client = OpenAIClient(
        base_url="http://localhost:20128/v1",
        api_key="sk-test",
        model="test-model",
        http_client=mock_http_oai,
    )
    oai_reply = await oai_client.generate_chat(
        system_prompt="پرامپت",
        messages=[{"role": "user", "text": "این چیه؟", "attachments": [{"mime_type": "image/jpeg", "data_b64": b64_img}]}],
    )
    assert oai_reply == "با متن جواب دادم!"
    first_payload = mock_http_oai.post.call_args_list[0].kwargs["json"]
    second_payload = mock_http_oai.post.call_args_list[1].kwargs["json"]
    assert first_payload["messages"][1]["content"][1]["image_url"]["url"] == f"data:image/jpeg;base64,{b64_img}"
    assert isinstance(second_payload["messages"][1]["content"], str)

    # 3. Conversation history in ai_chat_service stores text-only (attachments never re-sent on next turn)
    reset_ai_chat_state()
    await init_db(str(tmp_path / "vision_hist.db"))
    cfg = _fake_settings()
    try:
        mock_gem = MagicMock(spec=GeminiClient)
        mock_gem.generate_chat = AsyncMock(return_value="عجب عکسیه!")
        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_gem
        ):
            await generate_ai_chat_reply(
                chat_id=11111,
                thread_id=0,
                chat_type="private",
                speaker_name="آرش",
                user_text="عکسو ببین",
                attachments=[{"mime_type": "image/jpeg", "data_b64": b64_img}],
                visual_context_note="کاربر عکس فرستاده",
            )
            saved_msgs = _conversations[(11111, 0)].messages
            assert all("attachments" not in m for m in saved_msgs)
    finally:
        await close_db()
        reset_ai_chat_state()


def test_detect_profile_inspection_intent_rules() -> None:
    """Verify profile/bio/style intent detection for self, replied user, and negative guardrails."""
    assert detect_profile_inspection_intent("پروفایلم رو ببین به استایلم نمره بده") == (True, False)
    assert detect_profile_inspection_intent("بیوگرافیم چطوره؟") == (True, False)
    assert detect_profile_inspection_intent("به پروفایلش نمره بده", has_replied_human=True) == (True, True)
    assert detect_profile_inspection_intent("استایلشو ببین", has_replied_human=True) == (True, True)
    assert detect_profile_inspection_intent("پروفایلم رو ببین", has_replied_human=True) == (True, False)
    assert detect_profile_inspection_intent("چطوره؟", has_replied_human=True) == (False, False)
    assert detect_profile_inspection_intent("سلام رفیق خوبی؟") == (False, False)


@pytest.mark.asyncio
async def test_fetch_profile_and_extract_visual_attachments() -> None:
    """Verify profile bio+photo fetching (including privacy fallback) and GIF/photo extraction."""
    reset_ai_chat_state()
    bot = _make_bot(
        file_bytes_map={
            "pfp_fid_640": _SAMPLE_JPEG,
            "gif_mp4_1": _SAMPLE_MP4,
            "gif_thumb_1": _SAMPLE_JPEG,
            "photo_640": _SAMPLE_JPEG,
        }
    )

    # 1. Profile context with bio + photo + caching
    user_obj = SimpleNamespace(id=777, first_name="آرش", full_name="آرش راد", username="arash_bio")
    prof = await fetch_user_profile_context(bot, user_obj, now=100.0)
    assert prof["has_bio"] is True
    assert prof["has_photo"] is True
    assert "یوتیوبر" in prof["context_text"]
    assert prof["attachment"]["mime_type"] == "image/jpeg"

    # Cached within TTL -> does not re-hit Telegram API
    prof_cached = await fetch_user_profile_context(bot, user_obj, now=150.0)
    assert prof_cached["has_photo"] is True
    assert bot.get_chat.call_count == 1

    # 2. Privacy-hidden bio/photo handled gracefully
    bot_private = _make_bot()
    bot_private.get_chat = AsyncMock(return_value=SimpleNamespace(bio=None, username=None, first_name="علی", last_name=None))
    bot_private.get_user_profile_photos = AsyncMock(return_value=SimpleNamespace(photos=[]))
    prof_priv = await fetch_user_profile_context(bot_private, SimpleNamespace(id=888, first_name="علی"), now=100.0)
    assert prof_priv["has_bio"] is False
    assert prof_priv["has_photo"] is False
    assert "پنهان است" in prof_priv["context_text"]

    # 3. GIF extraction: Gemini uses video/mp4, 9Router/OpenAI uses thumbnail image/jpeg
    gif_obj = SimpleNamespace(
        file_id="gif_mp4_1",
        file_unique_id="ugif1",
        file_size=150_000,
        file_name="laugh.mp4",
        thumbnail=SimpleNamespace(file_id="gif_thumb_1", file_unique_id="uthumb1"),
    )
    msg_gif = _make_message(user_id=11111, animation=gif_obj)
    att_gem, note_gem = await extract_visual_attachment_from_message(bot, msg_gif, provider="gemini")
    assert att_gem is not None and att_gem["mime_type"] == "video/mp4"
    assert "گیف" in (note_gem or "")

    att_oai, _ = await extract_visual_attachment_from_message(bot, msg_gif, provider="9router")
    assert att_oai is not None and att_oai["mime_type"] == "image/jpeg"

    # 4. MagicMock(spec=Message) with unset attributes must NOT extract anything
    plain_msg = _make_message(user_id=11111, text="سلام")
    att_none, note_none = await extract_visual_attachment_from_message(bot, plain_msg, provider="gemini")
    assert att_none is None and note_none is None


@pytest.mark.asyncio
async def test_end_to_end_vision_gif_photo_profile_and_toggle(tmp_path: Path) -> None:
    """Verify end-to-end GIF/photo Vision, self/replied profile rating, toggle OFF fallback, and 0-token acks."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "ai_vision_e2e.db"))
    cfg = _fake_settings(admin_ids=[11111], provider="gemini")
    try:
        chat_id = -100555
        hub_tid = 3
        await repository.upsert_topic(chat_id, hub_tid, "The Hub | گفتگوی آزاد")
        await init_ai_chat_state()
        await set_ai_chat_state(True)
        set_ai_bot_identity(999999, "PrimeTestBot")

        bot = _make_bot(
            file_bytes_map={
                "pfp_fid_640": _SAMPLE_JPEG,
                "gif_mp4_1": _SAMPLE_MP4,
                "gif_thumb_1": _SAMPLE_JPEG,
                "photo_640": _SAMPLE_JPEG,
            }
        )
        recorded_calls: list[dict] = []

        async def fake_gen_chat(system_prompt, messages, temperature=0.85, max_output_tokens=600, model=None):
            recorded_calls.append({"system": system_prompt, "messages": messages})
            return "عجب تیپی زدی! ۹ از ۱۰ به استایلت 😎"

        mock_client = MagicMock(spec=GeminiClient)
        mock_client.generate_chat = AsyncMock(side_effect=fake_gen_chat)
        flt = AiChatTriggerFilter()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_client
        ):
            # 1. Bare GIF reply to bot -> Vision sees GIF and replies textually
            gif_obj = SimpleNamespace(
                file_id="gif_mp4_1",
                file_unique_id="ugif1",
                file_size=120_000,
                file_name="cat_vibe.mp4",
                thumbnail=SimpleNamespace(file_id="gif_thumb_1"),
            )
            msg_gif = _make_message(
                user_id=22222,
                text=None,
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=701,
                reply_from_user_id=999999,
                reply_from_is_bot=True,
                animation=gif_obj,
            )
            assert await flt(msg_gif, bot) is True
            await handle_ai_chat_message(msg_gif, bot)
            assert len(recorded_calls) == 1
            assert recorded_calls[-1]["messages"][-1]["attachments"][0]["mime_type"] == "video/mp4"
            assert "گیف" in recorded_calls[-1]["system"]
            bot.send_message.assert_awaited_once()

            # 2. Photo with @bot caption -> passes filter & sends photo to Vision
            photo_msg = _make_message(
                user_id=22222,
                caption="@PrimeTestBot این ست‌اپ چطوره؟",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=702,
                photo=[SimpleNamespace(file_id="photo_640", width=640, file_size=20_000)],
            )
            assert await flt(photo_msg, bot) is True
            await handle_ai_chat_message(photo_msg, bot)
            assert len(recorded_calls) == 2
            assert recorded_calls[-1]["messages"][-1]["attachments"][0]["mime_type"] == "image/jpeg"

            # 3. Profile & Bio style rating for self
            prof_msg = _make_message(
                user_id=22222,
                text="@PrimeTestBot پروفایلم رو ببین و به استایلم نمره بده",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=703,
            )
            await handle_ai_chat_message(prof_msg, bot)
            assert len(recorded_calls) == 3
            assert "یوتیوبر" in recorded_calls[-1]["system"]
            assert recorded_calls[-1]["messages"][-1]["attachments"][0]["mime_type"] == "image/jpeg"

            # 4. Replying to another member asking to rate their profile inspects that member
            bot.get_chat.reset_mock()
            reply_prof_msg = _make_message(
                user_id=22222,
                text="@PrimeTestBot به پروفایلش نمره بده",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=704,
                reply_from_user_id=33333,
                reply_from_name="سارا",
                reply_text="سلام بچه‌ها",
            )
            await handle_ai_chat_message(reply_prof_msg, bot)
            assert bot.get_chat.call_args.args[0] == 33333

            # 5. Zero-token ack still uses 0 LLM calls and 0 profile lookups
            calls_before_ack = len(recorded_calls)
            ack_msg = _make_message(
                user_id=22222,
                text="@PrimeTestBot مرسی رفیق",
                chat_id=chat_id,
                chat_type="supergroup",
                thread_id=hub_tid,
                message_id=705,
            )
            await handle_ai_chat_message(ack_msg, bot)
            assert len(recorded_calls) == calls_before_ack
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "👍"

            # 6. Toggle Vision OFF via callback 'mvis' -> bare GIF uses zero-token reaction instead of LLM
            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=11111, type="private")
            cb.message.message_id = 888
            cb.message.edit_text = AsyncMock()
            fsm = _make_fsm()

            await on_manage_ai_callback(cb, AiChatCB(action="mvis", owner_id=11111), bot, fsm)
            assert is_ai_chat_vision_enabled_cached() is False
            assert await repository.get_ai_chat_vision_enabled() is False

            calls_before_off = len(recorded_calls)
            await handle_ai_chat_message(msg_gif, bot)
            assert len(recorded_calls) == calls_before_off
            assert bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "😁"
            assert await set_ai_chat_vision_enabled_state(True) is True
    finally:
        await close_db()
        reset_ai_chat_state()
