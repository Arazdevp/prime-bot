"""HTML escaping helper for Telegram HTML formatting."""

import html
from typing import Any


def escape_html(text: Any) -> str:
    """Escape dynamic string for safe Telegram HTML formatting.

    Converts &, <, >, ", and ' to standard HTML entities.
    Returns empty string if text is None.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)
