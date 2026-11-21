"""Helper functions for resolving and formatting Telegram forum topic thread IDs."""

from typing import Any


def resolve_thread_id(message: Any) -> int:
    """Resolve Telegram forum thread ID.

    Normalizes None and 1 to 1 (General topic).
    In Telegram, for non-forum or non-topic messages, message_thread_id may
    appear on replies when is_topic_message is False. Only treat
    message_thread_id as a forum topic ID when is_topic_message is True.
    """
    is_topic_message = getattr(message, "is_topic_message", None)
    thread_id = getattr(message, "message_thread_id", None)

    # If is_topic_message is explicitly False, this is a reply in a regular group, not a topic thread
    if is_topic_message is False:
        return 1

    if thread_id is None or thread_id == 1:
        return 1

    return int(thread_id)


def to_send_thread_id(thread_id: int | None) -> int | None:
    """Convert normalized internal thread ID (1 = General) to Telegram API send target.

    In Telegram Bot API, sending to the General topic requires message_thread_id=None.
    Non-general topics require their actual integer ID.
    """
    if thread_id is None or thread_id == 1:
        return None
    return int(thread_id)
