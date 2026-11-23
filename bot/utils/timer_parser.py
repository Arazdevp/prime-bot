"""Pure utilities for Persian/Jalali timer parsing, calendar math, and formatting."""

from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo
import jdatetime
from bot.utils.dates import PERSIAN_MONTHS, PERSIAN_WEEKDAYS, format_persian_date

MIN_TIMER_MINUTES = 1
MAX_TIMER_DAYS = 365
MAX_TIMER_MINUTES = MAX_TIMER_DAYS * 24 * 60
MAX_NOTE_LENGTH = 250

_TO_ASCII_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يك", "01234567890123456789یک")
_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

_UNIT_ALT = r"days?|d|روز|ر|hours?|hrs?|hr|h|ساعت|س|minutes?|mins?|min|m|دقیقه|د"
_NOT_LETTER = r"(?![a-zA-Z\u0600-\u06FF])"

_DURATION_PART_RE = re.compile(
    rf"(\d+)\s*({_UNIT_ALT}){_NOT_LETTER}",
    re.IGNORECASE,
)
_FULL_DURATION_RE = re.compile(
    rf"^\s*(?:(?:\d+\s*(?:{_UNIT_ALT}){_NOT_LETTER})(?:\s*(?:و|\+|,)?\s*)?)+$",
    re.IGNORECASE,
)
_PREFIX_DURATION_RE = re.compile(
    rf"^\s*((?:\d+\s*(?:{_UNIT_ALT}){_NOT_LETTER}(?:\s*(?:و|\+)\s*|\s+)?)+)",
    re.IGNORECASE,
)
_CLOCK_TIME_RE = re.compile(r"^(\d{1,2})[:.](\d{1,2})$")
_DATE_YMD_RE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")
_DATE_MD_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})$")


def normalize_digits(text: str) -> str:
    """Convert Persian/Arabic-Indic digits and Arabic ye/kaf into canonical forms."""
    return (text or "").translate(_TO_ASCII_DIGITS)


def to_persian_digits(value: str | int) -> str:
    """Convert ASCII digits to Persian digits for readable Persian UI labels."""
    return str(value).translate(_TO_PERSIAN_DIGITS)


def sanitize_note(raw: str | None) -> str | None:
    """Clean and bound a user-provided timer note (returns None if empty)."""
    if not raw:
        return None
    cleaned = re.sub(r"\s+", " ", raw).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_NOTE_LENGTH:
        cleaned = cleaned[: MAX_NOTE_LENGTH - 1].rstrip() + "…"
    return cleaned


def jalali_month_days(year: int, month: int) -> int:
    """Return exact number of days in a Jalali month (accounts for leap Esfand = 30)."""
    if 1 <= month <= 6:
        return 31
    if 7 <= month <= 11:
        return 30
    if month == 12:
        return 30 if jdatetime.date(year, 1, 1).isleap() else 29
    raise ValueError(f"Invalid Jalali month: {month}")


def add_jalali_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (jyear, jmonth) pair by delta months."""
    zero_based = (year * 12 + (month - 1)) + delta
    new_year, rem = divmod(zero_based, 12)
    return new_year, rem + 1


def clamp_jalali_date(year: int, month: int, day: int) -> jdatetime.date:
    """Construct a valid Jalali date by clamping day to the month's maximum day."""
    max_day = jalali_month_days(year, month)
    safe_day = max(1, min(day, max_day))
    return jdatetime.date(year, month, safe_day)


def jalali_to_tehran_dt(
    jyear: int,
    jmonth: int,
    jday: int,
    hour: int,
    minute: int,
    tz_name: str = "Asia/Tehran",
) -> datetime:
    """Convert a Jalali date and wall-clock hour/minute into a timezone-aware datetime."""
    g_date = jdatetime.date(jyear, jmonth, jday).togregorian()
    tz = ZoneInfo(tz_name)
    return datetime(g_date.year, g_date.month, g_date.day, hour, minute, 0, tzinfo=tz)


def to_utc_iso(dt: datetime) -> str:
    """Normalize a timezone-aware datetime to UTC ISO8601 string with second precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_utc_iso(value: str) -> datetime:
    """Parse an ISO8601 timestamp into timezone-aware UTC datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_duration_persian(total_minutes: int) -> str:
    """Format a positive minute count into fluent Persian (e.g. '۱ ساعت و ۳۰ دقیقه')."""
    if total_minutes <= 0:
        return "۰ دقیقه"
    days, rem = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{to_persian_digits(days)} روز")
    if hours:
        parts.append(f"{to_persian_digits(hours)} ساعت")
    if minutes:
        parts.append(f"{to_persian_digits(minutes)} دقیقه")
    return " و ".join(parts)


def format_remaining_persian(delta: timedelta) -> str:
    """Format remaining timedelta into friendly Persian relative phrasing."""
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "الان"
    if total_seconds < 60:
        return "کمتر از ۱ دقیقه دیگه"
    # Round up partial minute slightly only when < 1 hour for accurate visual feel
    total_minutes = (total_seconds + 30) // 60
    if total_minutes < 1:
        total_minutes = 1
    days, rem = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{to_persian_digits(days)} روز")
    if hours:
        parts.append(f"{to_persian_digits(hours)} ساعت")
    if minutes and days == 0:
        parts.append(f"{to_persian_digits(minutes)} دقیقه")
    elif minutes and hours == 0 and days > 0:
        parts.append(f"{to_persian_digits(minutes)} دقیقه")
    return " و ".join(parts) + " دیگه"


def format_jalali_day_label(jdate: jdatetime.date) -> str:
    """Format a Jalali date as 'دوشنبه ۲ آذر ۱۴۰۵'."""
    weekday = PERSIAN_WEEKDAYS.get(jdate.weekday(), "")
    month_name = PERSIAN_MONTHS.get(jdate.month, "")
    return f"{weekday} {to_persian_digits(jdate.day)} {month_name} {to_persian_digits(jdate.year)}"


def format_fire_time_persian(
    fire_at: datetime,
    now_tehran: datetime | None = None,
    tz_name: str = "Asia/Tehran",
) -> str:
    """Format a fire timestamp naturally relative to Tehran today/tomorrow/after-tomorrow."""
    tz = ZoneInfo(tz_name)
    target = fire_at.astimezone(tz) if fire_at.tzinfo else fire_at.replace(tzinfo=tz)
    now = now_tehran.astimezone(tz) if now_tehran else datetime.now(tz)

    time_str = target.strftime("%H:%M")
    day_diff = (target.date() - now.date()).days

    if day_diff == 0:
        return f"امروز ساعت {time_str}"
    if day_diff == 1:
        return f"فردا ساعت {time_str}"
    if day_diff == 2:
        return f"پس‌فردا ساعت {time_str}"
    return f"{format_persian_date(target)} ساعت {time_str}"


def _unit_to_minutes(unit: str) -> int:
    u = unit.lower()
    if u in ("d", "day", "days", "ر", "روز"):
        return 24 * 60
    if u in ("h", "hr", "hrs", "hour", "hours", "س", "ساعت"):
        return 60
    return 1


def parse_duration_input(raw: str | None) -> int | None:
    """Parse custom duration text into total minutes within [1, MAX_TIMER_MINUTES].

    Accepts:
      - Bare integer minutes: '30', '۴۵'
      - Hours:minutes: '1:30', '۰۲:۱۵'
      - Units in Persian or English: '30m', '۳۰د', '2h', '۲س', '1h30m', '۱ ساعت و ۴۵ دقیقه', '2d'
    """
    if not raw:
        return None
    text = normalize_digits(raw).strip()
    if not text:
        return None

    # 1. Bare integer -> minutes
    if text.isdigit():
        mins = int(text)
        return mins if MIN_TIMER_MINUTES <= mins <= MAX_TIMER_MINUTES else None

    # 2. H:MM format -> hours and minutes
    hm_match = _CLOCK_TIME_RE.match(text)
    if hm_match:
        h = int(hm_match.group(1))
        m = int(hm_match.group(2))
        if m > 59:
            return None
        total = h * 60 + m
        return total if MIN_TIMER_MINUTES <= total <= MAX_TIMER_MINUTES else None

    # 3. Compound unit tokens
    if not _FULL_DURATION_RE.match(text):
        return None
    matches = _DURATION_PART_RE.findall(text)
    if not matches:
        return None

    total = 0
    for amount_str, unit in matches:
        total += int(amount_str) * _unit_to_minutes(unit)

    return total if MIN_TIMER_MINUTES <= total <= MAX_TIMER_MINUTES else None


def parse_time_input(raw: str | None) -> tuple[int, int] | None:
    """Parse wall-clock time like '14:30', '۱۴:۳۰', '9.05', or bare hour '14'."""
    if not raw:
        return None
    text = normalize_digits(raw).strip()
    if not text:
        return None

    if text.isdigit():
        hour = int(text)
        if 0 <= hour <= 23:
            return hour, 0
        return None

    m = _CLOCK_TIME_RE.match(text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def parse_jalali_date_token(token: str, now_tehran: datetime) -> jdatetime.date | None:
    """Parse a Jalali date token ('امروز', 'فردا', 'پس‌فردا', '1405/09/10', '09/10')."""
    cleaned = normalize_digits(token).strip().replace("\u200c", "")
    today_j = jdatetime.date.fromgregorian(date=now_tehran.date())

    if cleaned == "امروز":
        return today_j
    if cleaned == "فردا":
        return today_j + jdatetime.timedelta(days=1)
    if cleaned in ("پسفردا",):
        return today_j + jdatetime.timedelta(days=2)

    ymd = _DATE_YMD_RE.match(cleaned)
    if ymd:
        y, mo, d = int(ymd.group(1)), int(ymd.group(2)), int(ymd.group(3))
        if not (1300 <= y <= 1500):
            return None
        try:
            return jdatetime.date(y, mo, d)
        except ValueError:
            return None

    md = _DATE_MD_RE.match(cleaned)
    if md:
        mo, d = int(md.group(1)), int(md.group(2))
        try:
            candidate = jdatetime.date(today_j.year, mo, d)
        except ValueError:
            return None
        if candidate < today_j:
            try:
                candidate = jdatetime.date(today_j.year + 1, mo, d)
            except ValueError:
                return None
        return candidate

    return None


def _validate_future_bounds(
    target_tehran: datetime,
    now_tehran: datetime,
    *,
    min_lead_seconds: int = 30,
) -> bool:
    """Ensure target fire time is strictly in the future and within MAX_TIMER_DAYS."""
    delta = target_tehran - now_tehran
    if delta.total_seconds() < min_lead_seconds:
        return False
    if delta > timedelta(days=MAX_TIMER_DAYS):
        return False
    return True


def parse_jalali_date_time_input(
    raw: str | None,
    now_tehran: datetime,
    tz_name: str = "Asia/Tehran",
) -> datetime | None:
    """Parse a date/time input such as '1405/09/10 14:30', 'فردا 09:00', or '18:30'."""
    if not raw:
        return None
    text = normalize_digits(raw).strip()
    if not text:
        return None

    # Normalize 'پس فردا' -> 'پس‌فردا'
    text = re.sub(r"^پس\s+فردا\b", "پس‌فردا", text)
    tokens = text.split()
    if not tokens:
        return None

    # Case 1: Just a time ('18:30')
    if len(tokens) == 1:
        parsed_time = parse_time_input(tokens[0])
        if parsed_time is not None and (":" in tokens[0] or "." in tokens[0]):
            h, m = parsed_time
            target = now_tehran.replace(hour=h, minute=m, second=0, microsecond=0)
            if target <= now_tehran:
                target += timedelta(days=1)
            return target if _validate_future_bounds(target, now_tehran) else None

        # Or just a date ('1405/09/10' / 'فردا') -> default 09:00
        parsed_date = parse_jalali_date_token(tokens[0], now_tehran)
        if parsed_date is not None:
            target = jalali_to_tehran_dt(parsed_date.year, parsed_date.month, parsed_date.day, 9, 0, tz_name)
            return target if _validate_future_bounds(target, now_tehran) else None
        return None

    # Case 2: '<date> <time>' (optionally with word 'ساعت' between them)
    parsed_date = parse_jalali_date_token(tokens[0], now_tehran)
    if parsed_date is None:
        return None

    time_tokens = tokens[1:]
    if time_tokens and time_tokens[0] == "ساعت":
        time_tokens = time_tokens[1:]
    if len(time_tokens) != 1:
        return None

    parsed_time = parse_time_input(time_tokens[0])
    if parsed_time is None:
        return None

    h, m = parsed_time
    target = jalali_to_tehran_dt(parsed_date.year, parsed_date.month, parsed_date.day, h, m, tz_name)
    return target if _validate_future_bounds(target, now_tehran) else None


def parse_quick_timer_command(
    args_text: str | None,
    now_tehran: datetime,
    tz_name: str = "Asia/Tehran",
) -> tuple[str, datetime, str | None] | None:
    """Parse shorthand '/timer <time_spec> [note]' into (timer_type, fire_at_tehran, note).

    Examples:
      - '30m برم شام'            -> ('relative', now + 30m, 'برم شام')
      - '۳۰د برم شام'            -> ('relative', now + 30m, 'برم شام')
      - '45 تمرین'               -> ('relative', now + 45m, 'تمرین')
      - '1h30m جلسه'             -> ('relative', now + 90m, 'جلسه')
      - '18:30 تماس با دکتر'     -> ('scheduled', today/tomorrow 18:30, 'تماس با دکتر')
      - 'فردا 09:00 تحویل کار'   -> ('scheduled', tomorrow 09:00, 'تحویل کار')
      - '1405/09/10 14:00 امتحان' -> ('scheduled', 1405-09-10 14:00, 'امتحان')
    """
    if not args_text or not args_text.strip():
        return None

    raw_trimmed = args_text.strip()
    norm = normalize_digits(raw_trimmed)
    # Normalize 'پس فردا' prefix to single token
    norm = re.sub(r"^پس\s+فردا(?=\s|$)", "پس‌فردا", norm)

    tokens = norm.split()
    if not tokens:
        return None

    # 1. Explicit date token first ('امروز', 'فردا', 'پس‌فردا', '1405/09/10', '09/10')
    parsed_date = parse_jalali_date_token(tokens[0], now_tehran)
    if parsed_date is not None:
        idx = 1
        if idx < len(tokens) and tokens[idx] == "ساعت":
            idx += 1
        hour, minute = 9, 0
        if idx < len(tokens):
            maybe_time = parse_time_input(tokens[idx])
            if maybe_time is not None:
                hour, minute = maybe_time
                idx += 1
        target = jalali_to_tehran_dt(parsed_date.year, parsed_date.month, parsed_date.day, hour, minute, tz_name)
        if not _validate_future_bounds(target, now_tehran):
            return None
        # Preserve original Persian/English text for note after consumed tokens
        note_raw = _slice_remaining_after_tokens(raw_trimmed, idx)
        return "scheduled", target, sanitize_note(note_raw)

    # 2. Clock time ('18:30' / '۱۴:۳۰')
    if ":" in tokens[0] or "." in tokens[0]:
        maybe_time = parse_time_input(tokens[0])
        if maybe_time is not None:
            h, m = maybe_time
            target = now_tehran.replace(hour=h, minute=m, second=0, microsecond=0)
            if target <= now_tehran:
                target += timedelta(days=1)
            if not _validate_future_bounds(target, now_tehran):
                return None
            note_raw = _slice_remaining_after_tokens(raw_trimmed, 1)
            return "scheduled", target, sanitize_note(note_raw)

    # 3. Relative duration with units at prefix (e.g. '30m شام', '1h 30m جلسه', '۲ ساعت و ۱۵ دقیقه کار')
    prefix_match = _PREFIX_DURATION_RE.match(norm)
    if prefix_match:
        prefix_str = prefix_match.group(1)
        # Ensure prefix didn't end with dangling 'و' that belongs to the note
        cleaned_prefix = re.sub(r"(?:\s+و\s*|\s*\+\s*)$", "", prefix_str).strip()
        duration_mins = parse_duration_input(cleaned_prefix)
        if duration_mins is not None:
            # Count exact prefix length in normalized text (1-to-1 char mapping with raw_trimmed)
            # Note: 'پس فردا' wasn't matched here so raw_trimmed and norm have identical lengths
            consumed_len = len(cleaned_prefix)
            # Check boundary: next char must be end or whitespace
            if consumed_len == len(norm) or norm[consumed_len].isspace():
                target = now_tehran + timedelta(minutes=duration_mins)
                note_raw = raw_trimmed[consumed_len:].strip()
                return "relative", target, sanitize_note(note_raw)

    # 4. Bare integer minutes prefix (e.g. '30 برم شام' or '45')
    if tokens[0].isdigit():
        mins = int(tokens[0])
        if MIN_TIMER_MINUTES <= mins <= MAX_TIMER_MINUTES:
            target = now_tehran + timedelta(minutes=mins)
            note_raw = _slice_remaining_after_tokens(raw_trimmed, 1)
            return "relative", target, sanitize_note(note_raw)

    return None


def _slice_remaining_after_tokens(raw_text: str, token_count: int) -> str:
    """Extract the remainder of raw_text after skipping token_count whitespace-separated tokens."""
    # Handle 'پس فردا' (2 raw tokens collapsed to 1 normalized token)
    adjusted = re.sub(r"^پس\s+فردا(?=\s|$)", "پس‌فردا", raw_text.strip())
    parts = adjusted.split(maxsplit=token_count)
    if len(parts) <= token_count:
        return ""
    return parts[token_count].strip()
