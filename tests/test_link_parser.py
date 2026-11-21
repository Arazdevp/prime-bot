"""Tests for Telegram topic link and thread ID parser."""

import pytest
from bot.utils.link_parser import get_clean_chat_id, parse_topic_link


def test_get_clean_chat_id() -> None:
    assert get_clean_chat_id(-1002458921841) == "2458921841"
    assert get_clean_chat_id("-1002458921841") == "2458921841"
    assert get_clean_chat_id(-2458921841) == "2458921841"
    assert get_clean_chat_id("2458921841") == "2458921841"


def test_parse_private_topic_link() -> None:
    current_chat = -1002458921841
    url = "https://t.me/c/2458921841/1234"
    chat_id, thread_id, error = parse_topic_link(url, current_chat_id=current_chat)

    assert error is None
    assert chat_id == -1002458921841
    assert thread_id == 1234


def test_parse_private_message_link_in_topic() -> None:
    current_chat = -1002458921841
    url = "https://t.me/c/2458921841/1234/5678"
    chat_id, thread_id, error = parse_topic_link(url, current_chat_id=current_chat)

    assert error is None
    assert chat_id == -1002458921841
    assert thread_id == 1234


def test_parse_short_url() -> None:
    current_chat = -1002458921841
    url = "t.me/c/2458921841/42"
    chat_id, thread_id, error = parse_topic_link(url, current_chat_id=current_chat)

    assert error is None
    assert chat_id == -1002458921841
    assert thread_id == 42


def test_parse_direct_numeric_thread_id() -> None:
    current_chat = -1002458921841
    chat_id, thread_id, error = parse_topic_link("99", current_chat_id=current_chat)

    assert error is None
    assert chat_id == current_chat
    assert thread_id == 99


def test_parse_public_group_link() -> None:
    current_chat = -1002458921841
    url = "https://t.me/my_supergroup/777"
    chat_id, thread_id, error = parse_topic_link(url, current_chat_id=current_chat)

    assert error is None
    assert chat_id == current_chat
    assert thread_id == 777


def test_mismatched_chat_id_returns_error() -> None:
    current_chat = -1002458921841
    url = "https://t.me/c/9999999999/1234"
    chat_id, thread_id, error = parse_topic_link(url, current_chat_id=current_chat)

    assert error is not None
    assert "متعلق به این گروه نیست" in error
    assert chat_id is None
    assert thread_id is None


def test_invalid_input() -> None:
    _, _, error = parse_topic_link("")
    assert error is not None

    _, _, error = parse_topic_link("not-a-valid-link")
    assert error is not None
    assert "نامعتبر است" in error
