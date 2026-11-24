"""Natural Iranian Persian persona, runtime context builder, and long-term memory learning prompts."""

from datetime import datetime
from typing import Any
from bot.utils.dates import format_persian_date, get_tehran_now

DEFAULT_AI_CHAT_SYSTEM_PROMPT = """تو مثل یه آدم واقعی، باحال، خوش‌فکر و صمیمی ایرانی توی تلگرام چت می‌کنی؛ نه شبیه یه ربات یا دستیار پشتیبانی خشک!

اصول لحن و شخصیت تو:
- همیشه به فارسی محاوره‌ایِ طبیعی، روان و زنده حرف بزن (شبیه چت روزمره آدمای باهوش ایرانی؛ مثلاً «آره دقیقاً»، «ببین راستش»، «اوکیه ولی یه نکته هست»، «دمت گرم»).
- هرگز نگو «به عنوان یک هوش مصنوعی...»، «چطور می‌توانم کمکتان کنم؟»، «سؤال بسیار خوبی پرسیدید» یا جملات رسمی و کتابیِ کلیشه‌ای.
- وقتی توی گروه تلگرام هستی، دقیقاً مثل یکی از بچه‌های خودِ گروه رفتار کن که همون وسط نشسته و جریان بحث‌ها و روحیه بچه‌ها رو می‌شناسه. اگر از قبل اخلاق یا دغدغه کسی رو می‌شناسی، خیلی طبیعی و نامحسوس توی حرف‌هات لحاظش کن (نه اینکه پروفایل طرف رو مثل لیست بخونی!).
- اگر پیام طرف مقابل کوتاه و خودمانی بود، تو هم کوتاه، خودمونی و باحال جواب بده (۱ تا ۳ خط طبیعی). اگر بحث جدی، فنی یا عمیق شد، دقیق، دلسوزانه و قابل اتکا ولی همچنان با لحن طبیعی و غیرکتابی توضیح بده.
- از شوخی و ایموجی فقط در حد کم و به‌جا استفاده کن (حداکثر یکی دو تا وقتی واقعاً حس‌وحالش باشه؛ متن رو شلوغ نکن).
- هیچ‌وقت الکی هندونه زیر بغل کسی نذار و اگر چیزی رو مطمئن نیستی یا اطلاع نداری، خیلی راحت و روراست بگو مطمئن نیستی یا ندیدیش.
- از مارک‌داون مثل ** یا ### یا ``` اصلاً استفاده نکن. متن ساده و تمیز بنویس و فقط اگه واقعاً لازم شد از تگ‌های پایه HTML تلگرام مثل <b> یا <code> استفاده کن."""

SYSTEM_PROMPT_LEARN_MEMORY = """شما یک تحلیل‌گر رفتارشناسی و حافظه بلندمدت برای یک گروه تلگرامی فارسی هستید.
به شما حافظه فعلی گروه، پروفایل‌های قبلی اعضا (در صورت وجود) و پیام‌های جدید تاپیک «گفتگوی آزاد» (با فرمت [uid:USER_ID] [HH:MM] نام: متن) داده می‌شود.

وظیفه شما استخراج و ادغام شناخت واقعی درباره فضای کلی گروه و شخصیت هر یک از اعضاست تا ربات مثل یک هم‌گروهی قدیمی همه را بشناسد.

قوانین حیاتی:
۱. فقط از شواهد واقعی موجود در پیام‌ها و حافظه قبلی استفاده کنید؛ هیچ چیز از خودتان حدس نزنید یا اضافه نکنید.
۲. اطلاعات قبلی ارزشمند در پروفایل اعضا و فضای گروه را حفظ کرده و مشاهدات جدید را به‌شکل هوشمندانه با آن‌ها ادغام کنید (پروفایل قبلی را پاک نکنید مگر اینکه اصلاح شده باشد).
۳. فقط کاربرانی را در آرایه members بیاورید که `user_id` عددی مثبت معتبر در متن پیام‌ها دارند (حداقل وقتی یک پیام معنادار یا نشانه شخصیتی از آن‌ها هست). ربات‌ها یا افراد بدون شناسه را نادیده بگیرید.
۴. تمام متن‌های توصیفی را به فارسی کوتاه، دقیق و کاربردی بنویسید (برای هر فیلد ۱ تا ۲ جمله موجز).
۵. خروجی شما باید **فقط و فقط یک شیء JSON معتبر** و دقیقاً مطابق ساختار زیر باشد (بدون هیچ متن اضافه یا توضیح بیرون از JSON):

{
  "group_culture": "خلاصه جو کلی، لحن غالب و فرهنگ فعلی گروه",
  "inside_jokes": "اصطلاحات، شوخی‌های تکرارشونده یا تکیه‌کلام‌های خاص بچه‌های گروه",
  "ongoing_topics": "موضوعات، دغدغه‌ها یا پروژه‌هایی که این روزها بچه‌ها زیاد درگیرش هستند",
  "members": [
    {
      "user_id": 123456,
      "display_name": "نام کاربر",
      "personality_summary": "تیپ شخصیتی، خلق‌وخو و مدل برخورد این فرد در جمع",
      "speaking_style": "لحن حرف زدن، تکیه‌کلام‌ها و میزان صمیمیت/جدیت",
      "interests_and_topics": "حوزه‌های علاقه، تخصص یا موضوعاتی که دنبال می‌کند",
      "notable_facts": "نکات یا پروژه‌های مهم شخصی/کاری که خودش در جمع گفته"
    }
  ]
}"""


def _format_member_profiles_section(
    member_profiles: list[dict[str, Any]],
    max_chars: int = 3200,
) -> str:
    """Format learned member personality profiles cleanly within a strict character budget."""
    if not member_profiles:
        return ""
    lines: list[str] = ["[شناخت تو از شخصیت اعضای گروه]"]
    used = len(lines[0])
    for prof in member_profiles:
        name = str(prof.get("display_name") or "عضو").strip()
        summary = str(prof.get("personality_summary") or "").strip()
        style = str(prof.get("speaking_style") or "").strip()
        interests = str(prof.get("interests_and_topics") or "").strip()
        facts = str(prof.get("notable_facts") or "").strip()

        parts = [f"- {name}"]
        if summary:
            parts.append(f"شخصیت: {summary}")
        if style:
            parts.append(f"لحن: {style}")
        if interests:
            parts.append(f"علاقه‌مندی‌ها: {interests}")
        if facts:
            parts.append(f"نکات: {facts}")
        if len(parts) == 1:
            continue

        entry = " | ".join(parts)
        if used + len(entry) + 1 > max_chars:
            break
        lines.append(entry)
        used += len(entry) + 1

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_group_memory_section(group_memory: dict[str, Any] | None) -> str:
    """Format persisted group culture, inside jokes, and ongoing topics."""
    if not group_memory:
        return ""
    culture = str(group_memory.get("culture_summary") or "").strip()
    jokes = str(group_memory.get("inside_jokes") or "").strip()
    topics = str(group_memory.get("ongoing_topics") or "").strip()
    if not (culture or jokes or topics):
        return ""

    lines = ["[شناخت تو از فضای گروه]"]
    if culture:
        lines.append(f"- حال‌وهوای کلی: {culture}")
    if jokes:
        lines.append(f"- شوخی‌ها و اصطلاحات جمع: {jokes}")
    if topics:
        lines.append(f"- بحث‌های داغ و پروژه‌های فعلی: {topics}")
    return "\n".join(lines)


def build_runtime_system_prompt(
    base_prompt: str,
    *,
    speaker_name: str,
    chat_type: str,
    topic_name: str | None = None,
    group_title: str | None = None,
    group_memory: dict[str, Any] | None = None,
    member_profiles: list[dict[str, Any]] | None = None,
    recent_messages_text: str | None = None,
    now_tehran: datetime | None = None,
) -> str:
    """Combine persona prompt with Tehran time, group presence, learned memory, and live topic context."""
    tehran_dt = now_tehran or get_tehran_now()
    date_str = format_persian_date(tehran_dt)
    time_str = tehran_dt.strftime("%H:%M")

    safe_speaker = (speaker_name or "دوست عزیز").strip() or "دوست عزیز"
    clean_group = (group_title or "").strip()
    clean_topic = (topic_name or "").strip()

    if chat_type == "private":
        place_desc = "پیوی (چت خصوصی)"
        presence_line = f"- وضعیت حضور: داری در پیوی با {safe_speaker} صحبت می‌کنی"
        if clean_group:
            presence_line += f" (هر دو عضو گروه «{clean_group}» هستید)"
        presence_line += "."
    else:
        group_part = f"گروه تلگرامی «{clean_group}»" if clean_group else "گروه تلگرام"
        topic_part = f" (تاپیک «{clean_topic}»)" if clean_topic else ""
        place_desc = f"{group_part}{topic_part}"
        presence_line = (
            f"- وضعیت حضور: تو الان داخل {group_part}{topic_part} هستی و وسط جمع بچه‌های گروه داری صحبت می‌کنی؛ "
            "کاملاً حواست به اینکه داخل گروهی و کی داره باهات حرف می‌زنه هست."
        )

    sections = [
        base_prompt.strip(),
        (
            "[زمینه مکالمه فعلی]\n"
            f"- زمان فعلی (تهران): {date_str}، ساعت {time_str}\n"
            f"- طرف صحبت الان: {safe_speaker}\n"
            f"- محل گفتگو: {place_desc}\n"
            f"{presence_line}\n"
            "- نکته ارسال: پاسخ کامل را طوری بنویس که در یک پیام تلگرامی جا بگیرد و از مارک‌داون (** یا #) استفاده نکن."
        ),
    ]

    group_mem_block = _format_group_memory_section(group_memory)
    if group_mem_block:
        sections.append(group_mem_block)

    members_block = _format_member_profiles_section(member_profiles or [])
    if members_block:
        sections.append(members_block)

    if recent_messages_text and recent_messages_text.strip():
        trimmed_recent = recent_messages_text.strip()
        if len(trimmed_recent) > 2400:
            trimmed_recent = "…\n" + trimmed_recent[-2400:]
        sections.append(
            "[آخرین پیام‌های همین الان در تاپیک]\n"
            f"{trimmed_recent}\n"
            "(از شناخت اعضا و پیام‌های اخیر فقط به‌شکل طبیعی مثل یک دوست قدیمی استفاده کن و هرگز به شکل ماشینی پروفایل کسی را روخوانی نکن.)"
        )
    elif group_mem_block or members_block:
        sections.append(
            "(از این شناخت درباره فضای گروه و شخصیت اعضا خیلی طبیعی و نامحسوس استفاده کن، نه با لحن ربات‌گونه.)"
        )

    return "\n\n".join(sections)
