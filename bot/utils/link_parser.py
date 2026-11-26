"""Telegram topic and message link parser utility."""

from dataclasses import dataclass
import re
from typing import Tuple
from urllib.parse import parse_qs, urlparse

_RESERVED_TME_PATHS = frozenset(
    {
        "c",
        "s",
        "addstickers",
        "addemoji",
        "share",
        "joinchat",
        "proxy",
        "socks",
        "iv",
        "bg",
        "addtheme",
        "setlanguage",
        "login",
    }
)

_TME_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?t\.me/[^\s<>()\"']+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedMessageLink:
    """Structured representation of a Telegram group/channel message link."""

    chat_id: int | None
    username: str | None
    thread_id: int | None
    message_id: int


def parse_message_link(raw_input: str | None) -> ParsedMessageLink | None:
    """Extract a Telegram message link (private t.me/c/... or public t.me/<user>/...) from text.

    Supports topic message links (t.me/c/chat/thread/msg), direct message links
    (t.me/c/chat/msg), query parameters (?thread=42&single), and embedded URLs.
    Returns None if no valid Telegram message link is found.
    """
    if not raw_input or not raw_input.strip():
        return None

    for match in _TME_URL_RE.finditer(raw_input):
        candidate = match.group(0).rstrip(".,!?;:)]}>\"'")
        normalized = candidate if candidate.lower().startswith(("http://", "https://")) else f"https://{candidate}"
        try:
            parsed = urlparse(normalized)
        except ValueError:
            continue

        segments = [seg for seg in parsed.path.split("/") if seg]
        if len(segments) < 2:
            continue

        qs = parse_qs(parsed.query)
        query_thread: int | None = None
        raw_thread_vals = qs.get("thread")
        if raw_thread_vals and raw_thread_vals[0].isdigit():
            val = int(raw_thread_vals[0])
            if val > 0:
                query_thread = val

        first = segments[0]
        if first.lower() == "c":
            if len(segments) < 3 or not segments[1].isdigit() or not segments[2].isdigit():
                continue
            clean_chat = segments[1]
            if len(segments) >= 4 and segments[3].isdigit():
                thread_id: int | None = int(segments[2])
                message_id = int(segments[3])
            else:
                thread_id = query_thread
                message_id = int(segments[2])

            if message_id <= 0 or (thread_id is not None and thread_id <= 0):
                continue

            return ParsedMessageLink(
                chat_id=-int(f"100{clean_chat}"),
                username=None,
                thread_id=thread_id,
                message_id=message_id,
            )

        if first.lower() in _RESERVED_TME_PATHS:
            continue
        if not re.fullmatch(r"[a-zA-Z0-9_]{4,}", first):
            continue

        if len(segments) >= 3 and segments[1].isdigit() and segments[2].isdigit():
            thread_id = int(segments[1])
            message_id = int(segments[2])
        elif segments[1].isdigit():
            thread_id = query_thread
            message_id = int(segments[1])
        else:
            continue

        if message_id <= 0 or (thread_id is not None and thread_id <= 0):
            continue

        return ParsedMessageLink(
            chat_id=None,
            username=first.lower(),
            thread_id=thread_id,
            message_id=message_id,
        )

    return None


def get_clean_chat_id(chat_id: int | str) -> str:
    """Normalize Telegram supergroup chat_id by stripping -100 prefix or minus sign."""
    s = str(chat_id).strip()
    if s.startswith("-100"):
        return s[4:]
    return s.lstrip("-")


def parse_topic_link(
    raw_input: str,
    current_chat_id: int | None = None,
) -> Tuple[int | None, int | None, str | None]:
    """Parse a Telegram topic link, message link, or thread ID.

    Supported formats:
    - https://t.me/c/2458921841/1234
    - https://t.me/c/2458921841/1234/5678
    - t.me/c/2458921841/1234
    - https://t.me/group_username/1234
    - Raw integer thread ID (e.g. "1234" or "1")

    Returns:
        (extracted_chat_id, thread_id, error_message)
        If parsing succeeds, error_message is None.
    """
    if not raw_input or not raw_input.strip():
        return None, None, "ورودی خالی است."

    cleaned = raw_input.strip()

    # 1. Direct integer thread ID
    if cleaned.isdigit():
        thread_id = int(cleaned)
        if thread_id <= 0:
            return None, None, "شناسه تاپیک باید یک عدد مثبت باشد."
        return current_chat_id, thread_id, None

    # 2. Private supergroup links: t.me/c/{chat_id}/{thread_id} or t.me/c/{chat_id}/{thread_id}/{msg_id}
    # Pattern captures:
    # group 1: chat_id
    # group 2: thread_id
    # group 3 (optional): msg_id
    private_match = re.search(
        r"(?:https?://)?(?:www\.)?t\.me/c/(\d+)/(\d+)(?:/(\d+))?",
        cleaned,
        re.IGNORECASE,
    )
    if private_match:
        extracted_clean_chat = private_match.group(1)
        thread_id = int(private_match.group(2))

        # Reconstruct standard bot API chat_id: -100 + clean_chat_id
        extracted_chat_id = -int(f"100{extracted_clean_chat}")

        # If current_chat_id is provided, verify match
        if current_chat_id is not None:
            current_clean = get_clean_chat_id(current_chat_id)
            if extracted_clean_chat != current_clean:
                return (
                    None,
                    None,
                    f"⚠️ لینک وارد شده متعلق به این گروه نیست!\n"
                    f"شناسه گروه در لینک: <code>{extracted_clean_chat}</code>\n"
                    f"شناسه این گروه: <code>{current_clean}</code>",
                )

        return extracted_chat_id, thread_id, None

    # 3. Public group links: t.me/{username}/{thread_id}
    public_match = re.search(
        r"(?:https?://)?(?:www\.)?t\.me/([a-zA-Z0-9_]{4,})/(\d+)",
        cleaned,
        re.IGNORECASE,
    )
    if public_match:
        thread_id = int(public_match.group(2))
        return current_chat_id, thread_id, None

    return (
        None,
        None,
        "❌ فرمت لینک یا شناسه تاپیک نامعتبر است.\n"
        "فرمت‌های مجاز:\n"
        "• لینک تاپیک: <code>https://t.me/c/1234567890/42</code>\n"
        "• شناسه عددی: <code>42</code>",
    )
