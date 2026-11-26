"""Offline unit tests for 9Router integration across config, LLM client, AI chat, memory, and status UI."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Chat, Message, User

from bot.config import (
    DEFAULT_NINEROUTER_BASE_URL,
    Settings,
    is_local_base_url,
    resolve_openai_compat_config,
)
from bot.database import repository
from bot.database.db import close_db, init_db
from bot.handlers.admin import cmd_status
from bot.handlers.ai_chat import cmd_manage_ai, on_manage_ai_callback, render_manage_ai_panel
from bot.handlers.panel import _format_ai_provider
from bot.keyboards.ai_chat import AiChatCB
from bot.main import notify_admins_startup
from bot.services.ai_chat_service import (
    generate_ai_chat_reply,
    get_effective_ai_chat_provider,
    reset_ai_chat_state,
)
from bot.services.ai_memory_service import _call_memory_learner
from bot.services.llm_client import (
    LLMError,
    OpenAIClient,
    close_ai_chat_client,
    close_llm_client,
    get_ai_chat_client,
    get_llm_client,
)


def _ninerouter_settings(**overrides):
    data = {
        "ADMIN_IDS": [11111],
        "ALLOWED_CHAT_IDS": [],
        "PROXY_URL": "socks5://127.0.0.1:10808",
        "AI_PROVIDER": "9router",
        "NINEROUTER_BASE_URL": "http://localhost:20128/v1",
        "NINEROUTER_API_KEY": "sk-9r-test-1234",
        "NINEROUTER_MODEL": "antigravity/gemini-2.5-flash",
        "OPENAI_BASE_URL": None,
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": None,
        "GEMINI_API_KEY": "AQ.FallbackGemini9999",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "AI_MAX_OUTPUT_TOKENS": 8192,
        "AI_REASONING_EFFORT": None,
        "AI_CHAT_PROVIDER": "9router",
        "AI_CHAT_API_KEY": "AQ.ChatGemini8888",
        "AI_CHAT_MODEL": "gemini-3.5-flash-lite",
        "AI_CHAT_FALLBACK_MODELS": ["gemini-flash-lite-latest"],
        "AI_CHAT_NINEROUTER_BASE_URL": None,
        "AI_CHAT_NINEROUTER_API_KEY": None,
        "AI_CHAT_NINEROUTER_MODEL": None,
        "AI_CHAT_NINEROUTER_FALLBACK_MODELS": [],
        "AI_CHAT_OPENAI_BASE_URL": None,
        "AI_CHAT_OPENAI_API_KEY": None,
        "AI_CHAT_OPENAI_MODEL": None,
        "AI_CHAT_OPENAI_FALLBACK_MODELS": [],
        "AI_CHAT_MAX_OUTPUT_TOKENS": 600,
        "AI_CHAT_TEMPERATURE": 0.85,
        "AI_CHAT_HISTORY_LIMIT": 8,
        "AI_CHAT_USER_LIMIT_COUNT": 20,
        "AI_CHAT_USER_LIMIT_WINDOW_SECONDS": 7200,
        "SUMMARY_TIME_HOUR": 0,
        "SUMMARY_TIME_MINUTE": 1,
        "MIN_MESSAGES_PER_TOPIC": 5,
        "TIMEZONE": "Asia/Tehran",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_settings_9router_defaults_aliases_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings defaults to 9Router, normalizes aliases, falls back to OPENAI_*, and validates keys."""
    for key in (
        "AI_PROVIDER",
        "AI_CHAT_PROVIDER",
        "NINEROUTER_BASE_URL",
        "NINEROUTER_API_KEY",
        "NINEROUTER_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(
        _env_file=None,
        BOT_TOKEN="123456:ABCDEF",
        NINEROUTER_API_KEY="sk-9r-key",
        NINEROUTER_MODEL="antigravity/gemini-2.5-flash",
    )
    assert s.AI_PROVIDER == "9router"
    assert s.AI_CHAT_PROVIDER == "9router"
    assert resolve_openai_compat_config(s, for_chat=False) == (
        DEFAULT_NINEROUTER_BASE_URL,
        "sk-9r-key",
        "antigravity/gemini-2.5-flash",
        "NINEROUTER_API_KEY",
    )

    s_alias = Settings(
        _env_file=None,
        BOT_TOKEN="123456:ABCDEF",
        AI_PROVIDER="ninerouter",
        AI_CHAT_PROVIDER="9-router",
        OPENAI_API_KEY="sk-legacy-fallback",
        OPENAI_MODEL="kr/claude-sonnet-4.5",
    )
    assert s_alias.AI_PROVIDER == "9router"
    assert s_alias.AI_CHAT_PROVIDER == "9router"
    assert resolve_openai_compat_config(s_alias, for_chat=True) == (
        DEFAULT_NINEROUTER_BASE_URL,
        "sk-legacy-fallback",
        "kr/claude-sonnet-4.5",
        "OPENAI_API_KEY",
    )

    with pytest.raises(ValueError, match="NINEROUTER_API_KEY"):
        Settings(
            _env_file=None,
            BOT_TOKEN="123456:ABCDEF",
            AI_PROVIDER="9router",
            NINEROUTER_MODEL="antigravity/gemini-2.5-flash",
        )

    with pytest.raises(ValueError, match="NINEROUTER_MODEL"):
        Settings(
            _env_file=None,
            BOT_TOKEN="123456:ABCDEF",
            AI_PROVIDER="9router",
            NINEROUTER_API_KEY="sk-9r-key",
        )


@pytest.mark.asyncio
async def test_local_9router_bypasses_proxy_while_remote_and_gemini_keep_proxy() -> None:
    """Verify localhost 9Router disables proxy/trust_env while remote 9Router and Gemini honor PROXY_URL."""
    assert is_local_base_url("http://localhost:20128/v1") is True
    assert is_local_base_url("http://127.0.0.1:20128/v1") is True
    assert is_local_base_url("http://[::1]:20128/v1") is True
    assert is_local_base_url("https://9router.example.com/v1") is False
    assert is_local_base_url("https://openrouter.ai/api/v1") is False

    created_kwargs: list[dict] = []
    real_async_client = httpx.AsyncClient

    def _fake_async_client(**kwargs):
        created_kwargs.append(kwargs)
        mock_client = MagicMock(spec=real_async_client)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        return mock_client

    try:
        await close_llm_client()
        await close_ai_chat_client()

        with patch("bot.services.llm_client.httpx.AsyncClient", side_effect=_fake_async_client):
            local_cfg = _ninerouter_settings(NINEROUTER_BASE_URL="http://localhost:20128/v1")
            llm = get_llm_client(local_cfg)
            assert isinstance(llm, OpenAIClient)
            assert created_kwargs[-1].get("trust_env") is False
            assert "proxy" not in created_kwargs[-1]

            await close_llm_client()

            remote_cfg = _ninerouter_settings(NINEROUTER_BASE_URL="https://9router.example.com/v1")
            get_llm_client(remote_cfg)
            assert created_kwargs[-1].get("proxy") == "socks5://127.0.0.1:10808"

            # AI chat: local 9Router bypasses proxy, remote Gemini retains proxy
            get_ai_chat_client(local_cfg, provider="9router")
            assert created_kwargs[-1].get("trust_env") is False
            assert "proxy" not in created_kwargs[-1]

            get_ai_chat_client(local_cfg, provider="gemini")
            assert created_kwargs[-1].get("proxy") == "socks5://127.0.0.1:10808"
    finally:
        await close_llm_client()
        await close_ai_chat_client()


@pytest.mark.asyncio
async def test_9router_protocol_reasoning_sse_without_done_and_string_errors() -> None:
    """Verify flat reasoning_effort, SSE streams without [DONE], SSE errors, and string error codes."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False

    # 1. Flat reasoning_effort for 9Router + SSE without trailing [DONE] & list delta content
    sse_resp = MagicMock(spec=httpx.Response)
    sse_resp.status_code = 200
    sse_resp.headers = httpx.Headers({})
    sse_resp.json.side_effect = ValueError("Not JSON")
    sse_resp.text = (
        'data: {"choices":[{"delta":{"content":[{"type":"thinking","text":"..."},{"type":"text","text":"سلام "}]}}]}\n'
        'data: {"choices":[{"finish_reason":"stop","delta":{"content":"رفیق"}}]}\n'
    )
    mock_http.post = AsyncMock(return_value=sse_resp)

    client = OpenAIClient(
        base_url="http://localhost:20128/v1",
        api_key="sk-9r-test",
        model="antigravity/gemini-2.5-flash",
        http_client=mock_http,
        reasoning_effort="low",
    )
    out = await client.generate("sys", "user")
    assert out == "سلام رفیق"
    sent_payload = mock_http.post.call_args.kwargs["json"]
    assert sent_payload["reasoning_effort"] == "low"
    assert "reasoning" not in sent_payload

    # 2. SSE frame containing an upstream error maps string error codes properly
    sse_err = MagicMock(spec=httpx.Response)
    sse_err.status_code = 200
    sse_err.headers = httpx.Headers({})
    sse_err.json.side_effect = ValueError("Not JSON")
    sse_err.text = 'data: {"error":{"code":"rate_limit_exceeded","message":"Too many requests"}}\n'
    mock_http.post = AsyncMock(return_value=sse_err)

    with pytest.raises(LLMError) as exc_sse:
        await client.generate("sys", "user")
    assert exc_sse.value.status_code == 429
    assert exc_sse.value.is_retryable is True
    assert exc_sse.value.is_throttle is True

    # 3. HTTP 200 JSON error with string code 'invalid_api_key' -> non-retryable 401
    json_auth_err = MagicMock(spec=httpx.Response)
    json_auth_err.status_code = 200
    json_auth_err.headers = httpx.Headers({})
    json_auth_err.json.return_value = {
        "error": {"code": "invalid_api_key", "message": "Invalid 9Router API key"}
    }
    mock_http.post = AsyncMock(return_value=json_auth_err)

    with pytest.raises(LLMError) as exc_auth:
        await client.generate("sys", "user")
    assert exc_auth.value.status_code == 401
    assert exc_auth.value.is_retryable is False


@pytest.mark.asyncio
async def test_9router_ai_chat_memory_panel_and_status_labels(tmp_path: Path) -> None:
    """Verify 9Router end-to-end across AI chat fallback, memory learner, /manage_ai, /status, and startup."""
    reset_ai_chat_state()
    await init_db(str(tmp_path / "test_9router.db"))
    cfg = _ninerouter_settings(
        AI_CHAT_NINEROUTER_MODEL="kr/claude-sonnet-4.5",
        AI_CHAT_NINEROUTER_FALLBACK_MODELS=["antigravity/gemini-2.5-flash"],
    )

    try:
        mock_oa = MagicMock(spec=OpenAIClient)
        mock_oa.generate_chat = AsyncMock(
            side_effect=[
                LLMError("rate limited", status_code=429, is_retryable=True),
                "چطوری رفیق؟ همه‌چی روی 9Router ردیفه!",
            ]
        )

        with patch("bot.services.ai_chat_service.get_settings", return_value=cfg), patch(
            "bot.services.ai_chat_service.get_ai_chat_client", return_value=mock_oa
        ), patch("bot.services.ai_memory_service.get_settings", return_value=cfg):
            reply = await generate_ai_chat_reply(
                chat_id=11111,
                thread_id=0,
                chat_type="private",
                speaker_name="آرش",
                user_text="سلام",
            )
            assert reply == "چطوری رفیق؟ همه‌چی روی 9Router ردیفه!"
            assert mock_oa.generate_chat.call_count == 2
            assert mock_oa.generate_chat.call_args_list[0].kwargs["model"] == "kr/claude-sonnet-4.5"
            assert mock_oa.generate_chat.call_args_list[1].kwargs["model"] == "antigravity/gemini-2.5-flash"

            # Memory learner also routes through AI_CHAT_NINEROUTER_MODEL
            mock_oa.generate_chat = AsyncMock(return_value='{"group_culture":"خوب","members":[]}')
            mem_out = await _call_memory_learner(mock_oa, "پیام تستی")
            assert '"group_culture"' in mem_out
            assert mock_oa.generate_chat.call_args.kwargs["model"] == "kr/claude-sonnet-4.5"

        # /manage_ai 9router + callback toggle (9router <-> gemini) + panel display
        bot = MagicMock(spec=Bot)
        bot.id = 99999
        bot.get_me = AsyncMock(return_value=SimpleNamespace(id=99999, username="PrimeTestBot"))
        bot.send_message = AsyncMock()

        fsm = MagicMock(spec=FSMContext)
        fsm.clear = AsyncMock()

        cmd_msg = MagicMock(spec=Message)
        cmd_msg.text = "/manage_ai 9router"
        cmd_msg.chat = MagicMock(spec=Chat)
        cmd_msg.chat.id = 11111
        cmd_msg.chat.type = "private"
        cmd_msg.from_user = MagicMock(spec=User)
        cmd_msg.from_user.id = 11111
        cmd_msg.answer = AsyncMock()

        with patch("bot.services.access.get_settings", return_value=cfg), patch(
            "bot.handlers.ai_chat.get_settings", return_value=cfg
        ), patch("bot.services.ai_chat_service.get_settings", return_value=cfg):
            await cmd_manage_ai(cmd_msg, bot, fsm)
            assert get_effective_ai_chat_provider() == "9router"
            assert await repository.get_ai_chat_provider() == "9router"
            panel_text = cmd_msg.answer.call_args.kwargs["text"]
            assert "9Router" in panel_text
            assert "NINEROUTER_API_KEY: ...1234" in panel_text
            assert "kr/claude-sonnet-4.5" in panel_text

            cb = MagicMock(spec=CallbackQuery)
            cb.from_user = SimpleNamespace(id=11111)
            cb.answer = AsyncMock()
            cb.message = MagicMock(spec=Message)
            cb.message.chat = SimpleNamespace(id=11111, type="private")
            cb.message.message_id = 101
            cb.message.edit_text = AsyncMock()

            await on_manage_ai_callback(cb, AiChatCB(action="prov", owner_id=11111), bot, fsm)
            assert get_effective_ai_chat_provider() == "gemini"
            _, kb_gem = await render_manage_ai_panel(owner_id=11111, bot_username="PrimeTestBot")
            btn_texts = [btn.text for row in kb_gem.inline_keyboard for btn in row]
            assert any("9Router" in txt for txt in btn_texts)

            await on_manage_ai_callback(cb, AiChatCB(action="prov", owner_id=11111), bot, fsm)
            assert get_effective_ai_chat_provider() == "9router"

        # /status, panel _format_ai_provider, and startup message
        with patch("bot.handlers.panel.get_settings", return_value=cfg):
            assert _format_ai_provider() == "9Router (antigravity/gemini-2.5-flash)"

        with patch("bot.main.get_settings", return_value=cfg):
            await notify_admins_startup(bot)
            startup_text = bot.send_message.call_args.kwargs["text"]
            assert "9Router (antigravity/gemini-2.5-flash)" in startup_text

        status_msg = MagicMock(spec=Message)
        status_msg.chat = MagicMock(spec=Chat)
        status_msg.chat.id = -100555
        status_msg.chat.type = "supergroup"
        status_msg.message_thread_id = 1
        status_msg.is_topic_message = False
        status_msg.bot = bot
        status_msg.delete = AsyncMock()
        await repository.ensure_group_settings(-100555, default_min_messages=5)

        with patch("bot.handlers.admin.get_settings", return_value=cfg), patch(
            "bot.handlers.admin.send_topic_reply", new=AsyncMock()
        ) as mock_reply:
            await cmd_status(status_msg)
            status_text = mock_reply.call_args.args[1]
            assert "9Router (antigravity/gemini-2.5-flash)" in status_text
    finally:
        await close_db()
        reset_ai_chat_state()
