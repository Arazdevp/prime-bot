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
from aiogram.fsm.storage.memory import MemoryStorage
from bot.config import get_settings
from bot.database.db import close_db, init_db
from bot.handlers import admin, listener, panel, private
from bot.middlewares.throttle import PrivateThrottleMiddleware
from bot.scheduler.jobs import build_scheduler, run_catchup_if_needed, set_scheduler
from bot.services.llm_client import close_llm_client
from bot.utils.dates import format_persian_date, get_tehran_now
from bot.utils.html_utils import escape_html
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


async def notify_admins_crash(bot: Bot, exc: BaseException) -> None:
    """Send Persian crash alert to configured admins before process exit."""
    settings = get_settings()
    if not settings.ADMIN_IDS:
        return

    text = (
        "🚨 <b>خطای بحرانی در ربات خلاصه‌ساز تلگرام:</b>\n\n"
        "فرآیند دریافت پیام‌ها (Polling) به دلیل خطای زیر متوقف شد:\n"
        f"<code>{escape_html(str(exc))}</code>\n\n"
        "<i>سرویس جهت بازیابی و راه‌اندازی مجدد متوقف می‌شود.</i>"
    )

    for admin_id in settings.ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as err:
            logger.warning("Failed sending crash notification to admin %s: %s", admin_id, err)


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

    # 3. Initialize Dispatcher with MemoryStorage, private throttle middleware, and routers
    dp = Dispatcher(storage=MemoryStorage())
    private_throttle = PrivateThrottleMiddleware(rate_limit_seconds=1.0)
    dp.message.outer_middleware(private_throttle)
    dp.callback_query.outer_middleware(private_throttle)

    dp.include_router(admin.router)
    dp.include_router(listener.router)
    dp.include_router(private.router)
    dp.include_router(panel.router)

    # 4. Initialize and start scheduler
    scheduler = build_scheduler(bot)
    set_scheduler(scheduler)
    scheduler.start()
    logger.info("Scheduler started successfully in timezone %s.", settings.TIMEZONE)

    # Log next run time of nightly job
    nightly_job = scheduler.get_job("nightly_summary")
    if nightly_job and nightly_job.next_run_time:
        logger.info("Next nightly summarization scheduled at: %s", nightly_job.next_run_time)

    # Check if today's scheduled nightly run was missed and run catch-up
    asyncio.create_task(run_catchup_if_needed(bot))

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

    exit_code = 0
    polling_task = asyncio.create_task(
        dp.start_polling(
            bot,
            allowed_updates=["message", "callback_query", "my_chat_member"],
            handle_signals=False,
        )
    )
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        done, pending = await asyncio.wait(
            {polling_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if polling_task in done:
            exc = polling_task.exception()
            if exc:
                logger.critical("Bot polling task crashed: %s", exc, exc_info=exc)
                await notify_admins_crash(bot, exc)
                exit_code = 1
            else:
                logger.info("Bot polling task finished normally.")
        elif stop_task in done:
            logger.info("Termination signal received. Shutting down gracefully...")
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass
    except (KeyboardInterrupt, SystemExit):
        logger.info("Termination initiated by user.")
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
    finally:
        if not stop_task.done():
            stop_task.cancel()

        logger.info("Executing cleanup tasks...")
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")
        set_scheduler(None)

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

        if exit_code != 0:
            sys.exit(exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
