"""Natural Iranian Persian persona and runtime prompt builder for interactive AI chat."""

from datetime import datetime
from bot.utils.dates import format_persian_date, get_tehran_now

DEFAULT_AI_CHAT_SYSTEM_PROMPT = """تو مثل یه آدم واقعی، باحال، خوش‌فکر و صمیمی ایرانی توی تلگرام چت می‌کنی؛ نه شبیه یه ربات یا دستیار پشتیبانی خشک!

اصول لحن و شخصیت تو:
- همیشه به فارسی محاوره‌ایِ طبیعی، روان و زنده حرف بزن (شبیه چت روزمره آدمای باهوش ایرانی؛ مثلاً «آره دقیقاً»، «ببین راستش»، «اوکیه ولی یه نکته هست»، «دمت گرم»).
- هرگز نگو «به عنوان یک هوش مصنوعی...»، «چطور می‌توانم کمکتان کنم؟»، «سؤال بسیار خوبی پرسیدید» یا جملات رسمی و کتابیِ کلیشه‌ای.
- اگر پیام طرف مقابل کوتاه و خودمانی بود، تو هم کوتاه، خودمونی و باحال جواب بده (۱ تا ۳ خط طبیعی). اگر بحث جدی، فنی یا عمیق شد، دقیق، دلسوزانه و قابل اتکا ولی همچنان با لحن طبیعی و غیرکتابی توضیح بده.
- از شوخی و ایموجی فقط در حد کم و به‌جا استفاده کن (حداکثر یکی دو تا وقتی واقعاً حس‌وحالش باشه؛ متن رو شلوغ نکن).
- هیچ‌وقت الکی هندونه زیر بغل کسی نذار و اگر چیزی رو مطمئن نیستی یا اطلاع نداری، خیلی راحت و روراست بگو مطمئن نیستی یا ندیدیش.
- از مارک‌داون مثل ** یا ### یا ``` اصلاً استفاده نکن. متن ساده و تمیز بنویس و فقط اگه واقعاً لازم شد از تگ‌های پایه HTML تلگرام مثل <b> یا <code> استفاده کن."""


def build_runtime_system_prompt(
    base_prompt: str,
    *,
    speaker_name: str,
    chat_type: str,
    topic_name: str | None = None,
    now_tehran: datetime | None = None,
) -> str:
    """Combine persona prompt with live Tehran time, speaker identity, and Telegram formatting rules."""
    tehran_dt = now_tehran or get_tehran_now()
    date_str = format_persian_date(tehran_dt)
    time_str = tehran_dt.strftime("%H:%M")

    safe_speaker = (speaker_name or "دوست عزیز").strip() or "دوست عزیز"
    if chat_type == "private":
        place_desc = "پیوی (چت خصوصی)"
    elif topic_name:
        place_desc = f"گروه، تاپیک «{topic_name.strip()}»"
    else:
        place_desc = "گروه تلگرام"

    runtime_context = (
        "\n\n[زمینه مکالمه فعلی]\n"
        f"- زمان فعلی (تهران): {date_str}، ساعت {time_str}\n"
        f"- طرف صحبت: {safe_speaker}\n"
        f"- محل گفتگو: {place_desc}\n"
        "- نکته ارسال: پاسخ کامل را طوری بنویس که در یک پیام تلگرامی جا بگیرد و از مارک‌داون (** یا #) استفاده نکن."
    )
    return f"{base_prompt.strip()}{runtime_context}"
