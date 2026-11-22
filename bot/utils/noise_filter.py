"""Noise filter utility to detect and filter low-value messages."""

import os
from pathlib import Path
import re

# Base list of low-value greetings, acknowledgments, and conversational filler
DEFAULT_NOISE_PHRASES = {
    # Persian greetings & conversational filler
    "سلام",
    "سلام علیکم",
    "درود",
    "خوبی",
    "چطوری",
    "مرسی",
    "ممنون",
    "ممنونم",
    "تشکر",
    "سپاس",
    "سپاسگزارم",
    "خواهش میکنم",
    "دمت گرم",
    "اوکی",
    "اوکیه",
    "باشه",
    "حله",
    "چشم",
    "فدات",
    "قربانت",
    "عالی",
    "عالیه",
    "خسته نباشید",
    "شب خوش",
    "صبح بخیر",
    "روز بخیر",
    "خداحافظ",
    "فعلا",
    "نوکرم",
    "چاکرم",
    "ارادت",
    "ایول",
    "دستت درد نکنه",
    "دست شما درد نکنه",
    "لطفا",
    "خواهش",
    "درسته",
    "همچنین",
    "بله",
    "خیر",
    # English greetings & filler
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "ok",
    "okay",
    "k",
    "sure",
    "bye",
    "goodbye",
    "good morning",
    "good night",
    "yes",
    "no",
    "nice",
    "great",
    "cool",
    "welcome",
    "yw",
    "np",
}

# Regex to check if text contains any alphabetic characters (Persian/Arabic or Latin).
# [^\W\d_] matches Unicode letters and excludes decimal digits (including Persian/Arabic-Indic digits), punctuation, and symbols.
ALPHABETIC_PATTERN = re.compile(r"[^\W\d_]")


def normalize_noise_text(text: str) -> str:
    """Normalize text by converting Persian/Arabic chars, lowercase, and removing punctuation."""
    # Persian / Arabic normalization
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("ۀ", "ه")
    # Persian digits to English digits
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    for i, d in enumerate(persian_digits):
        text = text.replace(d, str(i))
    for i, d in enumerate(arabic_digits):
        text = text.replace(d, str(i))

    text = text.lower().strip()
    # Remove punctuation, symbols, emojis, leaving letters, numbers, and single spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_custom_noise_words(custom_path: str = "data/noise_words.txt") -> set[str]:
    """Load additional noise words from external file if it exists."""
    words: set[str] = set()
    path = Path(custom_path)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        words.add(normalize_noise_text(stripped))
        except Exception:
            pass
    return words


_CUSTOM_NOISE_WORDS = load_custom_noise_words()
_NORMALIZED_DEFAULT_NOISE: set[str] = {normalize_noise_text(w) for w in DEFAULT_NOISE_PHRASES}
ALL_NOISE_PHRASES: set[str] = _NORMALIZED_DEFAULT_NOISE | _CUSTOM_NOISE_WORDS
_ALL_NOISE_NO_SPACE: set[str] = {p.replace(" ", "") for p in ALL_NOISE_PHRASES}


def is_noise(text: str | None, min_len: int = 3) -> bool:
    """Determine whether a message is considered low-value noise.

    Returns True if:
    - Text is None or empty / whitespace-only.
    - Text stripped length is less than min_len.
    - Text contains NO letters (e.g. only emojis, numbers, or symbols).
    - Normalized text exactly matches a known noise phrase.
    """
    if text is None:
        return True

    stripped = text.strip()
    if not stripped or len(stripped) < min_len:
        return True

    # Check if there are any alphabetic characters (Persian or Latin)
    if not ALPHABETIC_PATTERN.search(stripped):
        # Purely numbers, punctuation, or emojis
        return True

    # Check against normalized noise phrases
    normalized = normalize_noise_text(stripped)
    if not normalized or len(normalized) < min_len:
        return True

    if normalized in ALL_NOISE_PHRASES:
        return True

    # Also check if text without spaces matches any noise phrase without spaces
    no_space = normalized.replace(" ", "")
    if no_space in _ALL_NOISE_NO_SPACE:
        return True

    return False
