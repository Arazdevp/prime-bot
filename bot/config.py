"""Application configuration using pydantic-settings."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Bot settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE_PATH, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram Bot Configuration
    BOT_TOKEN: str = Field(..., description="Telegram Bot Token from @BotFather")
    ADMIN_IDS: Any = Field(
        default_factory=list,
        description="List of admin user IDs allowed to run global bot management",
    )
    ALLOWED_CHAT_IDS: Any = Field(
        default_factory=list,
        description="Allowed supergroup chat IDs (empty list allows any forum supergroup)",
    )

    # Proxy Configuration
    PROXY_URL: str | None = Field(
        default=None,
        description="Optional proxy URL (e.g. socks5://127.0.0.1:1080 or http://127.0.0.1:8080)",
    )

    # AI Provider Configuration
    AI_PROVIDER: Literal["gemini", "openai"] = Field(
        default="gemini",
        description="Active AI provider: 'gemini' or 'openai'",
    )

    # Google Gemini Settings
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="API key for Google Gemini",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-3.8-flash",
        description="Google Gemini model identifier",
    )

    # OpenAI-compatible Settings
    OPENAI_BASE_URL: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoint",
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="API key for OpenAI-compatible endpoint",
    )
    OPENAI_MODEL: str | None = Field(
        default=None,
        description="Model name for OpenAI-compatible endpoint",
    )
    AI_MAX_OUTPUT_TOKENS: int = Field(
        default=8192,
        ge=256,
        description="Maximum output tokens for AI response generation",
    )

    # Thresholds & Limits
    MIN_MESSAGES_PER_TOPIC: int = Field(
        default=5,
        ge=1,
        description="Minimum messages in a topic to trigger summarization",
    )
    MIN_MESSAGE_LENGTH: int = Field(
        default=3,
        ge=1,
        description="Minimum text length for a message to be recorded",
    )
    CHUNK_MAX_CHARS: int = Field(
        default=24000,
        ge=1000,
        description="Maximum characters before map-reduce chunking kicks in",
    )

    # Nightly Schedule
    SUMMARY_TIME_HOUR: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Hour (0-23) to run nightly summary in TIMEZONE",
    )
    SUMMARY_TIME_MINUTE: int = Field(
        default=1,
        ge=0,
        le=59,
        description="Minute (0-59) to run nightly summary in TIMEZONE",
    )
    TIMEZONE: str = Field(
        default="Asia/Tehran",
        description="Target timezone for schedule and timestamps",
    )

    # Data Retention & Flow Control
    RETENTION_DAYS: int = Field(
        default=7,
        ge=1,
        description="Days to retain raw message history before purge",
    )
    SEND_DELAY_SECONDS: float = Field(
        default=1.8,
        ge=0.1,
        description="Delay in seconds between Telegram publish calls",
    )
    AI_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on retryable AI failures",
    )
    AI_CHUNK_DELAY_SECONDS: float = Field(
        default=4.0,
        ge=0.0,
        description="Delay in seconds between consecutive LLM calls to prevent RPM/TPM bursts",
    )

    # Storage & Paths
    DB_PATH: str = Field(
        default="data/bot.db",
        description="Path to SQLite database file",
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for rotating log files",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Private Chat & Membership Behavior
    OWNER_NAMES: str = Field(
        default="آرش و پیت",
        description="Display names of bot owners used in private chat responses",
    )
    NOTIFY_ON_UNKNOWN_START: bool = Field(
        default=True,
        description="Notify ADMIN_IDS on first private contact from an unknown user",
    )
    AUTO_LEAVE_UNKNOWN_GROUPS: bool = Field(
        default=False,
        description="Automatically leave groups not listed in ALLOWED_CHAT_IDS when added",
    )

    # Interactive Timer & Reminder Limits
    TIMER_MAX_PER_USER: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum concurrent active timers allowed per user",
    )
    TIMER_RETENTION_DAYS: int = Field(
        default=30,
        ge=1,
        description="Days to retain completed/cancelled timers before purge",
    )

    # Daily Morning Greeting (Hills / Prime)
    MORNING_GREETING_HOUR: int = Field(
        default=5,
        ge=0,
        le=23,
        description="Hour (0-23) in TIMEZONE to send daily morning greeting",
    )
    MORNING_GREETING_MINUTE: int = Field(
        default=1,
        ge=0,
        le=59,
        description="Minute (0-59) in TIMEZONE to send daily morning greeting",
    )
    HILLS_START_DATE: str = Field(
        default="2026-08-28",
        description="ISO YYYY-MM-DD start date for Hills day counter (Day 1)",
    )
    MORNING_CATCHUP_END_HOUR: int = Field(
        default=11,
        ge=1,
        le=23,
        description="Latest local hour to send a missed morning greeting catch-up on startup",
    )

    @field_validator("HILLS_START_DATE", mode="before")
    @classmethod
    def validate_hills_start_date(cls, value: Any) -> str:
        """Ensure HILLS_START_DATE is a valid ISO YYYY-MM-DD date string."""
        raw = str(value).strip() if value is not None else "2026-08-28"
        if not raw:
            return "2026-08-28"
        date.fromisoformat(raw)
        return raw

    @field_validator("OWNER_NAMES", mode="before")
    @classmethod
    def parse_owner_names(cls, value: Any) -> str:
        """Fallback to default owner names if empty or whitespace."""
        if value is None:
            return "آرش و پیت"
        stripped = str(value).strip()
        return stripped if stripped else "آرش و پیت"

    @field_validator("ADMIN_IDS", "ALLOWED_CHAT_IDS", mode="after")
    @classmethod
    def parse_int_list(cls, value: Any) -> list[int]:
        """Parse comma-separated strings, ints, or raw lists into list[int]."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return [int(item) for item in items]
        if isinstance(value, (list, tuple, set)):
            return [int(item) for item in value]
        if isinstance(value, int):
            return [value]
        return [int(value)]

    @field_validator("PROXY_URL", mode="before")
    @classmethod
    def parse_optional_str(cls, value: Any) -> str | None:
        """Treat empty string as None."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Settings":
        """Ensure required keys exist for the selected AI provider."""
        if self.AI_PROVIDER == "gemini":
            if not self.GEMINI_API_KEY or not self.GEMINI_API_KEY.strip():
                raise ValueError("GEMINI_API_KEY must be provided when AI_PROVIDER='gemini'")
        elif self.AI_PROVIDER == "openai":
            if not self.OPENAI_BASE_URL or not self.OPENAI_BASE_URL.strip():
                raise ValueError("OPENAI_BASE_URL must be provided when AI_PROVIDER='openai'")
            if not self.OPENAI_API_KEY or not self.OPENAI_API_KEY.strip():
                raise ValueError("OPENAI_API_KEY must be provided when AI_PROVIDER='openai'")
            if not self.OPENAI_MODEL or not self.OPENAI_MODEL.strip():
                raise ValueError("OPENAI_MODEL must be provided when AI_PROVIDER='openai'")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
