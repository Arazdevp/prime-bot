"""Unit tests for General topic thread ID resolution and send target conversion."""

from types import SimpleNamespace
import pytest
from bot.utils.general_topic import resolve_thread_id, to_send_thread_id


def test_resolve_thread_id_general_none() -> None:
    """Telegram returns message_thread_id=None for messages in the General topic."""
    msg = SimpleNamespace(message_thread_id=None, is_topic_message=True)
    assert resolve_thread_id(msg) == 1


def test_resolve_thread_id_general_one() -> None:
    """Telegram can also return message_thread_id=1 for the General topic."""
    msg = SimpleNamespace(message_thread_id=1, is_topic_message=True)
    assert resolve_thread_id(msg) == 1


def test_resolve_thread_id_real_topic() -> None:
    """Specific forum topics return their numeric thread ID."""
    msg = SimpleNamespace(message_thread_id=158, is_topic_message=True)
    assert resolve_thread_id(msg) == 158


def test_resolve_thread_id_non_forum_reply() -> None:
    """In non-forum supergroups, replies can carry message_thread_id while is_topic_message is False."""
    msg = SimpleNamespace(message_thread_id=987, is_topic_message=False)
    assert resolve_thread_id(msg) == 1


def test_to_send_thread_id_general() -> None:
    """Sending to General topic in Telegram Bot API requires message_thread_id=None."""
    assert to_send_thread_id(1) is None
    assert to_send_thread_id(None) is None


def test_to_send_thread_id_specific_topic() -> None:
    """Sending to specific forum topics requires their exact integer thread ID."""
    assert to_send_thread_id(158) == 158
    assert to_send_thread_id(42) == 42
