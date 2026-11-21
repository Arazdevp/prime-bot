"""Main application entry point: initialization, router wiring, and graceful shutdown."""

import asyncio
import logging
import signal
import sys
from typing import Any
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramForbiddenError
from bot.config import get_settings
from bot.database.db import close_db, init_db
from bot.handlers import admin, listener
from bot.scheduler.jobs import build_scheduler
from bot.services.llm_client import close_llm_client
from bot.utils.dates import format_persian_date, get_tehran_now
from bot.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def create_bot_session(proxy_url: str | None) -> AiohttpSession:
    """Create an AiohttpSession configured with optional SOCKS5 or HTTP proxy."""
    if not proxy_url:
        return AiohttpSession()

    proxy_lower = proxy_url.lower()
    if proxy_lower.startswith("socks5://") or proxy_lower.startswith("socks5h://"):
        try:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(proxy_url)
            logger.info("Configured SOCKS5 proxy connector for Telegram bot session.")
            return AiohttpSession(connector=connector)
        except ImportError:
            logger.error("aiohttp-socks is required for SOCKS5 proxies. Defaulting to standard session.")
            return AiohttpSession()
    else:
        # Standard HTTP/HTTPS proxy
        logger.info("Configured HTTP proxy for Telegram bot session.")
        return AiohttpSession(proxy=proxy_url)


async def notify_admins_startup(bot: Bot) -> None:
    """Send startup greeting and next run time notification to configured admins."""
    settings = get_settings()
    if not settings.ADMIN_IDS:
        return

    tehran_now = get_tehran_now()
    persian_now = format_persian_date(tehran_now)
    time_str = tehran_now.strftime("%H:%M")

    text = (
        "🟢 <b>ربات خلاصه‌ساز روزانه تلگرام روشن شد.</b>\n\n"
        f"📅 تاریخ: {persian_now}\n"
        f"⏰ ساعت شروع به کار: {time_str} (تهران)\n"
        f"🌙 ساعت اجرای شبانه: {settings.SUMMARY_TIME_HOUR:02d}:{settings.SUMMARY_TIME_MINUTE:02d}\n"
        f"🧠 مدل فعال: <code>{settings.AI_PROVIDER} ({settings.GEMINI_MODEL if settings.AI_PROVIDER == 'gemini' else settings.OPENAI_MODEL})</code>\n"
        f"📊 وضعیت نظارت: آماده دریافت و پردازش پیام‌ها."
    )

    for admin_id in settings.ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except (TelegramForbiddenError, Exception):
            continue


async def main() -> None:
    """Main application lifecycle entrypoint."""
    settings = get_settings()
    setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)
    logger.info("Starting Telegram Forum Daily Summarizer Bot...")

    # 1. Initialize SQLite Database
    await init_db(settings.DB_PATH)
    logger.info("Database initialized successfully at %s", settings.DB_PATH)

    # 2. Build Telegram Bot with custom session & default HTML parsing
    session = create_bot_session(settings.PROXY_URL)
    bot = Bot(
        token=settings.BOT_TOKEN,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # 3. Initialize Dispatcher and register routers (Admin first, then Listener)
    dp = Dispatcher()
    dp.include_router(admin.router)
    dp.include_router(listener.router)

    # 4. Initialize and start scheduler
    scheduler = build_scheduler(bot)
    scheduler.start()
    logger.info("Scheduler started successfully in timezone %s.", settings.TIMEZONE)

    # Log next run time of nightly job
    nightly_job = scheduler.get_job("nightly_summary")
    if nightly_job and nightly_job.next_run_time:
        logger.info("Next nightly summarization scheduled at: %s", nightly_job.next_run_time)

    # 5. Best-effort startup notification to admins
    await notify_admins_startup(bot)

    # 6. Graceful shutdown handler
    stop_event = asyncio.Event()

    def _handle_exit_signal(*_: Any) -> None:
        logger.info("Received termination signal. Shutting down gracefully...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_exit_signal)

    try:
        # Start polling in background task
        polling_task = asyncio.create_task(
            dp.start_polling(
                bot,
                allowed_updates=["message"],
                handle_signals=False,
            )
        )
        if sys.platform == "win32":
            # On Windows, keep running until polling_task finishes or KeyboardInterrupt
            await polling_task
        else:
            await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Termination initiated by user.")
    finally:
        logger.info("Executing cleanup tasks...")
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")

        try:
            await dp.stop_polling()
        except Exception:
            pass
        logger.info("Bot polling stopped.")

        await bot.session.close()
        logger.info("Bot session closed.")

        await close_llm_client()
        logger.info("LLM client closed.")

        await close_db()
        logger.info("Database connection closed.")
        logger.info("Bot shutdown completed cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
