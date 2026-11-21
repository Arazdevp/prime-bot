"""Verification script for Phase 2."""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.utils.general_topic import resolve_thread_id, to_send_thread_id
from bot.utils.noise_filter import is_noise, normalize_noise_text
from bot.filters import AdminFilter, ForumGroupFilter
from bot.handlers import listener, admin


def test_thread_id_resolution() -> None:
    print("[1/3] Verifying resolve_thread_id & to_send_thread_id...")

    # Case 1: General topic with message_thread_id = None
    msg_general_none = SimpleNamespace(message_thread_id=None, is_topic_message=True)
    assert resolve_thread_id(msg_general_none) == 1, "Failed for None thread ID"

    # Case 2: General topic with message_thread_id = 1
    msg_general_one = SimpleNamespace(message_thread_id=1, is_topic_message=True)
    assert resolve_thread_id(msg_general_one) == 1, "Failed for thread ID = 1"

    # Case 3: Real forum topic (e.g. 57)
    msg_topic_57 = SimpleNamespace(message_thread_id=57, is_topic_message=True)
    assert resolve_thread_id(msg_topic_57) == 57, "Failed for thread ID = 57"

    # Case 4: Non-forum reply (is_topic_message is False, but message_thread_id is present)
    msg_non_topic_reply = SimpleNamespace(message_thread_id=999, is_topic_message=False)
    assert resolve_thread_id(msg_non_topic_reply) == 1, "Failed for non-topic message reply"

    # Test to_send_thread_id
    assert to_send_thread_id(1) is None, "Expected None for General topic (1)"
    assert to_send_thread_id(None) is None, "Expected None for None"
    assert to_send_thread_id(57) == 57, "Expected 57 for topic 57"

    print(" Thread ID resolution verified.")


def test_noise_filtering() -> None:
    print("[2/3] Verifying noise filter over diverse sample strings...")

    samples = [
        # Noise samples (should return True)
        ("", True, "empty string"),
        ("   ", True, "whitespace only"),
        ("ok", True, "short English filler"),
        ("مرسی", True, "Persian thanks"),
        ("سلام", True, "Persian greeting"),
        ("سلام دوستان", False, "valid conversation greeting"),
        ("دمت گرم", True, "Persian acknowledgment"),
        ("👍👍👍", True, "emojis only"),
        ("123456", True, "digits only"),
        ("???!!!...", True, "punctuation only"),
        ("thanks", True, "English thanks"),
        ("hello", True, "English greeting"),
        ("باشه", True, "Persian okay"),
        ("چشم", True, "Persian acknowledgment"),
        ("حله", True, "Persian filler"),
        ("سپاسگزارم", True, "Persian thanks"),
        # Valid messages (should return False)
        ("جلسه معماری فردا ساعت ۱۰ برگزار خواهد شد.", False, "meaningful meeting announcement"),
        ("لینک پول ریکوئست را ارسال کردم، لطفاً بررسی کنید: https://github.com/org/repo/pull/1", False, "PR link with context"),
        ("آیا باگ مربوط به لاگین در نسخه جدید برطرف شد؟", False, "meaningful question"),
        ("The new deployment pipeline is running smoothly in production.", False, "English sentence"),
        ("برای پیاده‌سازی این فیچر بهتر است از الگوی Factory استفاده کنیم.", False, "Architecture discussion"),
    ]

    for text, expected, desc in samples:
        actual = is_noise(text, min_len=3)
        assert actual == expected, (
            f"Failed on [{desc}]: text={text!r}, expected={expected}, got={actual}"
        )
        print(f"  [{'NOISE' if actual else 'VALID'}] '{text[:35]}' ({desc})")

    print(" Noise filter successfully classified all samples.")


def test_module_imports() -> None:
    print("[3/3] Checking filter and handler module imports...")
    assert AdminFilter is not None
    assert ForumGroupFilter is not None
    assert listener.router is not None
    assert admin.router is not None
    print(" All handler and filter modules imported cleanly.")


if __name__ == "__main__":
    test_thread_id_resolution()
    test_noise_filtering()
    test_module_imports()
    print("\n>>> ALL PHASE 2 CHECKS PASSED SUCCESSFULLY! <<<")
