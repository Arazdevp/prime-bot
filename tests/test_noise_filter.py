"""Unit tests for noise_filter module."""

import pytest
from bot.utils.noise_filter import is_noise, normalize_noise_text


def test_noise_filter_empty_and_short() -> None:
    """Test empty, whitespace, and short texts are filtered."""
    assert is_noise("") is True
    assert is_noise("   ") is True
    assert is_noise("ok", min_len=3) is True
    assert is_noise("ab", min_len=3) is True


def test_noise_filter_emoji_and_punctuation_only() -> None:
    """Test messages containing only emojis, digits, or punctuation are filtered."""
    assert is_noise("👍") is True
    assert is_noise("👍👍👍") is True
    assert is_noise("1234567") is True
    assert is_noise("!!!???...") is True
    assert is_noise("🔥❤️👏") is True


def test_noise_filter_persian_greetings_and_filler() -> None:
    """Test common Persian low-value greetings and filler words."""
    assert is_noise("سلام") is True
    assert is_noise("درود") is True
    assert is_noise("مرسی") is True
    assert is_noise("ممنون") is True
    assert is_noise("دمت گرم") is True
    assert is_noise("باشه") is True
    assert is_noise("اوکی") is True
    assert is_noise("چشم") is True
    assert is_noise("حله") is True
    assert is_noise("سپاسگزارم") is True


def test_noise_filter_english_greetings() -> None:
    """Test common English low-value greetings and filler words."""
    assert is_noise("hi") is True
    assert is_noise("hello") is True
    assert is_noise("thanks") is True
    assert is_noise("thank you") is True
    assert is_noise("good morning") is True


def test_noise_filter_persian_arabic_normalization() -> None:
    """Test normalization of Arabic characters ي and ك to Persian ی and ک."""
    # Arabic ي and ك in noise phrases
    assert is_noise("سلام عليكم") is True
    assert is_noise("مرسي") is True
    assert is_noise("اوكي") is True


def test_noise_filter_real_sentences_preserved() -> None:
    """Test meaningful conversational and technical sentences are preserved."""
    assert is_noise("سلام دوستان، جلسه امروز ساعت ۵ برگزار خواهد شد.") is False
    assert is_noise("لطفاً پول ریکوئست گیت‌هاب را بررسی و مرج کنید.") is False
    assert is_noise("آیا باگ لاگین در نسخه جدید برطرف شده است؟") is False
    assert is_noise("The release pipeline passed all automated tests.") is False
    assert is_noise("برای معماری دیتابیس تصمیم گرفتیم از SQLite در حالت WAL استفاده کنیم.") is False
