"""Inline keyboard factories and CallbackData schema for the Music Downloader."""

from typing import Any
from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from bot.services.music_service import format_track_label
from bot.utils.timer_parser import to_persian_digits

MUSIC_PAGE_SIZE = 6


class MusicCB(CallbackData, prefix="ms"):
    """Compact CallbackData for Music Downloader actions (always <= 64 bytes)."""

    action: str
    arg: str = ""
    page: int = 0


def build_music_home_kb(*, has_tracks: bool) -> InlineKeyboardMarkup:
    """Main Music Downloader keyboard shown under /music."""
    rows: list[list[InlineKeyboardButton]] = []
    if has_tracks:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🎶 آخرین موزیک‌ها",
                    callback_data=MusicCB(action="list", page=0).pack(),
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_music_list_kb(
    tracks: list[dict[str, Any]],
    page: int,
    total_pages: int,
) -> InlineKeyboardMarkup:
    """Paginated recent tracks keyboard with one-tap download buttons."""
    rows: list[list[InlineKeyboardButton]] = []
    media_icons = {
        "audio": "🎵",
        "document": "🎧",
        "voice": "🎤",
        "video": "🎬",
    }

    for track in tracks:
        icon = media_icons.get(str(track.get("media_type") or "audio"), "🎵")
        label = format_track_label(track, max_len=34)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{icon} {label}",
                    callback_data=MusicCB(action="dl", arg=str(track["id"]), page=page).pack(),
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="⬅️ قبلی",
                    callback_data=MusicCB(action="list", page=page - 1).pack(),
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"صفحه {to_persian_digits(page + 1)}/{to_persian_digits(total_pages)}",
                callback_data=MusicCB(action="noop").pack(),
            )
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="بعدی ➡️",
                    callback_data=MusicCB(action="list", page=page + 1).pack(),
                )
            )
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔙 بازگشت",
                callback_data=MusicCB(action="home").pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
