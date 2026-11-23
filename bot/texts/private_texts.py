"""Casual Persian private-chat and group-membership message pools."""

import random
from bot.config import get_settings
from bot.utils.html_utils import escape_html

DEFAULT_OWNERS = "آرش و پیت"

POOLS: dict[str, list[str]] = {
    "unauthorized": [
        "سلام {name} 👋 این ربات توسط {owners} مدیریت میشه و اجازهی استفاده ازش رو نداری، قربونت 🔥",
        "ببخشید {name} جان، این ربات اختصاصیه و فقط {owners} میتونن باهاش کار کنن. ناراحت نشی ها، قربونت 🔥",
        "هوی {name}! اینجا کاری برای انجام دادن نیست 😅 ربات تحت مدیریت {owners}ه و دسترسی نداری، قربونت",
        "{name} عزیز، دکمهی استارت رو زدی ولی در قفله 🔒 کلیدش دست {owners}ه، قربونت 🔥",
        "من فقط از {owners} دستور میگیرم {name} جان 😎 دمت گرم که اومدی ولی فعلاً نوبت تو نیست، قربونت",
        "اینجا بخش ویژهست و ورودت آزاد نیست 😄 اگه کاری داری با {owners} هماهنگ کن، قربونت 🔥",
        "سلام! من ربات خلاصهساز گروهیم و توی پیوی فقط به {owners} جواب میدم. بیخیال شو {name} جان، قربونت",
        "ای جانم {name}! ولی متأسفانه اجازهی استفاده از این ربات رو نداری. مدیرهاش {owners}ن، قربونت 🔥",
        "من یه ربات خدمتکارم ولی فقط برای {owners} 😅 به بقیه خدمات نمیدم، قربونت",
        "نه {name} جان، نمیشه 🙈 ربات اختصاصیه و {owners} مدیریتش میکنن، قربونت 🔥",
    ],
    "repeat": [
        "باز اومدی؟ 😄 جواب همونه {name} جان، فقط {owners} اجازه دارن، قربونت",
        "{name} جان تکرار نکن دیگه 🙈 قفله، قربونت 🔥",
        "هیچی عوض نشده ها 😅 اجازه نداری، قربونت",
    ],
    "ignore_notice": [
        "خب من دیگه تا یه ساعت جواب نمیدم {name} جان 😄 بعداً بیا، قربونت 🔥",
    ],
    "admin_welcome": [
        "به به {name}! 🔥 مدیر اومد، همهچی آمادهست. پنل رو باز کنم؟",
        "سلام {name} جان 😎 خوش اومدی، هرچی تنظیمات داری از پنل بزن",
        "اومدی {name}! 🔥 ربات سر پاست، بزن بریم سراغ پنل",
        "درود بر مدیر 🫡 {name} جان، پنل رو باز کن و هر کاری خواستی بکن",
        "{name} خودمونی! ربات منتظر دستورته 🔥",
        "چه خبر {name}؟ 😄 گروهها رو با پنل مدیریت کن، من هوای همهچی رو دارم",
    ],
    "group_admin_welcome": [
        "سلام {name} جان 👋 تو ادمین گروهی، پس میتونی تنظیمات همون گروه رو از پنل عوض کنی 🔥",
        "خوش اومدی {name}! 😎 فقط گروههایی که توش ادمینی توی پنل میبینی",
    ],
    "admin_hint": [
        "برای مدیریت ربات /panel رو بزن {name} جان 😄",
        "پیام رو نفهمیدم 🙈 /panel رو بزن تا پنل بیاد، قربونت 🔥",
    ],
    "callback_denied": [
        "این دکمه مال تو نیست {name} جان 😅",
        "دسترسی نداری، قربونت 🔥",
        "فقط مدیرها میتونن اینو بزنن 🙈",
    ],
    "group_added": [
        "سلام به همه 👋 من ربات خلاصهساز تاپیکام! برای کار کردن باید ادمین باشم و Privacy Mode خاموش باشه. تنظیماتم رو مدیرها از پیوی ربات با /panel عوض میکنن 🔥",
    ],
    "group_not_allowed": [
        "این گروه توی لیست مجاز من نیست، {owners} باید تأییدش کنن. فعلاً خداحافظ، قربونت 🔥",
    ],
}

# In-memory fallback tracker: (user_id, pool_key) -> last variant_key
_last_picked_memory: dict[tuple[int, str], str] = {}


def clear_last_picked_cache() -> None:
    """Clear the in-memory last-picked variant cache (useful for test isolation)."""
    _last_picked_memory.clear()


def _resolve_default_owners() -> str:
    """Fetch OWNER_NAMES from settings with a safe default fallback."""
    try:
        owners = get_settings().OWNER_NAMES
        if owners and owners.strip():
            return owners.strip()
    except Exception:
        pass
    return DEFAULT_OWNERS


def pick(
    pool_key: str,
    user_id: int | None = None,
    exclude_last: str | None = None,
    *,
    name: str = "دوست عزیز",
    owners: str | None = None,
    escape: bool = True,
) -> tuple[str, str]:
    """Pick a random formatted message from POOLS[pool_key], avoiding back-to-back repeats.

    Returns (formatted_text, variant_key) where variant_key has form '{pool_key}:{index}'.
    Set escape=False only for plain-text contexts such as callback.answer(show_alert=True).
    """
    if pool_key not in POOLS or not POOLS[pool_key]:
        raise KeyError(f"Unknown or empty text pool: {pool_key}")

    pool = POOLS[pool_key]
    effective_exclude = exclude_last
    if effective_exclude is None and user_id is not None:
        effective_exclude = _last_picked_memory.get((user_id, pool_key))

    if len(pool) == 1:
        chosen_idx = 0
    else:
        eligible = [
            idx
            for idx, template in enumerate(pool)
            if f"{pool_key}:{idx}" != effective_exclude
            and str(idx) != effective_exclude
            and template != effective_exclude
        ]
        if not eligible:
            eligible = list(range(len(pool)))
        chosen_idx = random.choice(eligible)

    variant_key = f"{pool_key}:{chosen_idx}"
    if user_id is not None:
        _last_picked_memory[(user_id, pool_key)] = variant_key
        if len(_last_picked_memory) > 5000:
            # Drop oldest inserted keys to keep memory bounded in long-running processes
            for old_key in list(_last_picked_memory.keys())[: len(_last_picked_memory) - 4000]:
                _last_picked_memory.pop(old_key, None)

    raw_name = (name or "").strip() or "دوست عزیز"
    raw_owners = (owners or "").strip() if owners is not None else _resolve_default_owners()
    if not raw_owners:
        raw_owners = DEFAULT_OWNERS

    formatted_name = escape_html(raw_name) if escape else raw_name
    formatted_owners = escape_html(raw_owners) if escape else raw_owners

    text = pool[chosen_idx].format(name=formatted_name, owners=formatted_owners)
    return text, variant_key
