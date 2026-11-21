"""Telegram topic and message link parser utility."""

import re
from typing import Tuple


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
