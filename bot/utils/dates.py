"""Date and calendar helpers for Jalali and Gregorian dates in Asia/Tehran."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import jdatetime

PERSIAN_WEEKDAYS = {
    0: "شنبه",
    1: "یکشنبه",
    2: "دوشنبه",
    3: "سه‌شنبه",
    4: "چهارشنبه",
    5: "پنج‌شنبه",
    6: "جمعه",
}

PERSIAN_MONTHS = {
    1: "فروردین",
    2: "اردیبهشت",
    3: "خرداد",
    4: "تیر",
    5: "مرداد",
    6: "شهریور",
    7: "مهر",
    8: "آبان",
    9: "آذر",
    10: "دی",
    11: "بهمن",
    12: "اسفند",
}


def get_tehran_now() -> datetime:
    """Return the current datetime in Asia/Tehran timezone."""
    return datetime.now(ZoneInfo("Asia/Tehran"))


def format_persian_date(dt: datetime) -> str:
    """Format datetime into Persian date: 'شنبه ۳۰ آبان ۱۴۰۵'."""
    tz = ZoneInfo("Asia/Tehran")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
    weekday_name = PERSIAN_WEEKDAYS.get(j_dt.weekday(), "")
    month_name = PERSIAN_MONTHS.get(j_dt.month, "")
    return f"{weekday_name} {j_dt.day} {month_name} {j_dt.year}"


def format_persian_short_date(dt: datetime) -> str:
    """Format datetime into short Persian date: '۱۴۰۵/۰۸/۳۰'."""
    tz = ZoneInfo("Asia/Tehran")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
    return f"{j_dt.year:04d}/{j_dt.month:02d}/{j_dt.day:02d}"


def format_gregorian_date(dt: datetime) -> str:
    """Format datetime into Gregorian string: '2026-11-21'."""
    tz = ZoneInfo("Asia/Tehran")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt.strftime("%Y-%m-%d")


def compute_nightly_cutoff(run_time: datetime | None = None, tz_name: str | None = None) -> datetime:
    """Compute cutoff = the latest 00:00 (TIMEZONE) at or before run_time, converted to UTC."""
    if tz_name is None:
        from bot.config import get_settings
        tz_name = get_settings().TIMEZONE

    tz = ZoneInfo(tz_name)
    if run_time is None:
        run_time = datetime.now(tz)
    elif run_time.tzinfo is None:
        run_time = run_time.replace(tzinfo=tz)
    else:
        run_time = run_time.astimezone(tz)

    midnight = run_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone(timezone.utc)


def summary_day_for_run(run_time: datetime, is_nightly: bool = True) -> datetime:
    """Return the calendar day for date formatting."""
    tz = ZoneInfo("Asia/Tehran")
    if run_time.tzinfo is None:
        return run_time.replace(tzinfo=tz)
    return run_time.astimezone(tz)


def format_date_heading(dt: datetime, is_nightly: bool = True) -> str:
    """Generate date heading for summary messages containing both Jalali and Gregorian dates.

    Derives heading directly from dt (the message timestamp), removing the is_nightly date hack.
    """
    target_date = summary_day_for_run(dt, is_nightly=is_nightly)
    jalali_str = format_persian_date(target_date)
    jalali_short = format_persian_short_date(target_date)
    gregorian_str = format_gregorian_date(target_date)
    return f"📅 <b>تاریخ:</b> {jalali_str} ({jalali_short} - {gregorian_str})"
