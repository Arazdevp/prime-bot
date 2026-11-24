"""Smart message splitting with HTML tag awareness and balance preservation."""

import html
import re

TELEGRAM_MESSAGE_MAX_CHARS = 4096

TAG_RE = re.compile(r"<(/?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^<>]*)?)>")
_INCOMPLETE_ENTITY_RE = re.compile(r"&(?:#x?[0-9a-fA-F]*|[a-zA-Z]*)$")
_ENTITY_OR_CHAR_RE = re.compile(
    r"&(?:#\d{1,7}|#x[0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,31});|.",
    re.DOTALL,
)


def _rendered_utf16_len(text: str) -> int:
    """Return length in UTF-16 code units after HTML entity unescaping (Telegram's text limit unit)."""
    if not text:
        return 0
    return len(html.unescape(text).encode("utf-16-le")) // 2


def telegram_text_length(text: str) -> int:
    """Return rendered Telegram message length in UTF-16 code units (excluding HTML tags)."""
    if not text:
        return 0
    return _rendered_utf16_len(TAG_RE.sub("", text))


def _prefix_end_for_utf16_limit(text_val: str, max_units: int) -> int:
    """Find largest raw index in text_val whose rendered UTF-16 length <= max_units without tearing entities."""
    if max_units <= 0 or not text_val:
        return 0

    total_units = 0
    end_idx = 0
    for match in _ENTITY_OR_CHAR_RE.finditer(text_val):
        tok = match.group(0)
        if len(tok) > 1 and html.unescape(tok) == tok:
            # Not a recognized HTML entity; count character by character
            for idx, ch in enumerate(tok):
                ch_units = len(ch.encode("utf-16-le")) // 2
                if total_units + ch_units > max_units:
                    return match.start() + idx
                total_units += ch_units
                end_idx = match.start() + idx + 1
        else:
            tok_units = _rendered_utf16_len(tok)
            if total_units + tok_units > max_units:
                return end_idx
            total_units += tok_units
            end_idx = match.end()
    return end_idx


def _first_atomic_unit_end(text_val: str) -> int:
    """Return end index of the first atomic character or HTML entity for forward progress."""
    if not text_val:
        return 0
    match = _ENTITY_OR_CHAR_RE.match(text_val)
    if not match:
        return 1
    tok = match.group(0)
    if len(tok) > 1 and html.unescape(tok) == tok:
        return 1
    return match.end()


def _has_visible_content(chunk: str) -> bool:
    """Return True if chunk contains non-whitespace visible text outside HTML tags."""
    return bool(TAG_RE.sub("", chunk).strip())


def _adjust_cut_for_html_entity(text_val: str, cut_idx: int) -> int:
    """Avoid splitting inside an HTML entity like '&amp;' or '&#x27;'."""
    if cut_idx <= 0 or cut_idx >= len(text_val):
        return cut_idx
    window_start = max(0, cut_idx - 12)
    window = text_val[window_start:cut_idx]
    match = _INCOMPLETE_ENTITY_RE.search(window)
    if match:
        candidate = window_start + match.start()
        if candidate > 0:
            return candidate
    return cut_idx


def _find_natural_cut(candidate_slice: str) -> int | None:
    """Return a clean split index at paragraph, line, or word/sentence boundary, or None."""
    p_idx = candidate_slice.rfind("\n\n")
    if p_idx != -1:
        return p_idx + 2

    l_idx = candidate_slice.rfind("\n")
    if l_idx != -1:
        return l_idx + 1

    s_idx = max(
        candidate_slice.rfind(". "),
        candidate_slice.rfind("! "),
        candidate_slice.rfind("؟ "),
        candidate_slice.rfind("; "),
        candidate_slice.rfind(" "),
    )
    if s_idx != -1:
        return s_idx + 1

    return None


def smart_split(text: str, limit: int = 4000, count_rendered: bool = False) -> list[str]:
    """Split text into chunks of length <= limit, preserving HTML tag balance.

    When count_rendered=False (default), measures raw string length.
    When count_rendered=True, measures Telegram rendered UTF-16 length (excluding
    HTML tags and after entity unescaping) so rich <a href="..."> links never
    trigger premature message splits.

    Splits progressively at paragraph (\\n\\n), newline (\\n), sentence/space,
    or hard character boundary as a last resort, never inside an HTML tag.
    Open tags at split boundaries are cleanly closed at the end of the chunk
    and re-opened at the start of the next chunk.
    """
    if not text or not text.strip():
        return []

    if count_rendered:
        if telegram_text_length(text) <= limit:
            return [text]
    elif len(text) <= limit:
        return [text]

    # Tokenize into tags and text segments
    # Token structure: ('tag', raw_tag, is_closing, tag_name) or ('text', content)
    tokens: list[tuple] = []
    last_end = 0
    for m in TAG_RE.finditer(text):
        if m.start() > last_end:
            tokens.append(("text", text[last_end:m.start()]))
        raw_tag = m.group(0)
        inner = m.group(1).strip()
        is_closing = inner.startswith("/")
        tag_name = inner[1:].split()[0].lower() if is_closing else inner.split()[0].lower()
        tokens.append(("tag", raw_tag, is_closing, tag_name))
        last_end = m.end()
    if last_end < len(text):
        tokens.append(("text", text[last_end:]))

    def get_closing_tags(stack: list[tuple[str, str]]) -> str:
        """Generate closing tags string in reverse order for active stack."""
        return "".join(f"</{name}>" for name, _ in reversed(stack))

    def get_opening_tags(stack: list[tuple[str, str]]) -> str:
        """Generate opening tags string in forward order for active stack."""
        return "".join(raw_open for _, raw_open in stack)

    parts: list[str] = []
    current_part: list[str] = []
    current_len = 0
    open_stack: list[tuple[str, str]] = []  # Stack of (tag_name, raw_opening_tag)

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token[0] == "tag":
            _, raw_tag, is_closing, tag_name = token

            temp_stack = list(open_stack)
            if is_closing:
                for i in range(len(temp_stack) - 1, -1, -1):
                    if temp_stack[i][0] == tag_name:
                        temp_stack.pop(i)
                        break
            else:
                temp_stack.append((tag_name, raw_tag))

            if count_rendered:
                # Avoid dangling empty opening tags at the exact end of a full chunk
                if not is_closing and current_len >= limit and current_part:
                    closing = get_closing_tags(open_stack)
                    flush_str = "".join(current_part) + closing
                    if _has_visible_content(flush_str):
                        parts.append(flush_str)
                    reopening = get_opening_tags(open_stack)
                    current_part = [reopening] if reopening else []
                    current_len = 0
                    continue

                current_part.append(raw_tag)
                open_stack = temp_stack
                idx += 1
                continue

            tag_len = len(raw_tag)
            needed_closing = get_closing_tags(temp_stack)
            if current_len + tag_len + len(needed_closing) <= limit:
                current_part.append(raw_tag)
                current_len += tag_len
                open_stack = temp_stack
                idx += 1
            else:
                reopening = get_opening_tags(open_stack)
                if current_len > len(reopening):
                    # Flush current part and retry placing this tag on a fresh chunk
                    closing = get_closing_tags(open_stack)
                    flush_str = "".join(current_part) + closing
                    if _has_visible_content(flush_str):
                        parts.append(flush_str)
                    current_part = [reopening] if reopening else []
                    current_len = len(reopening)
                    continue

                # Progress guard: tag + open_stack cannot fit even on an empty chunk -> skip tag safely
                if is_closing:
                    open_stack = temp_stack
                idx += 1
                continue

        elif token[0] == "text":
            _, text_val = token
            closing = get_closing_tags(open_stack)

            if count_rendered:
                text_units = _rendered_utf16_len(text_val)
                if current_len + text_units <= limit:
                    current_part.append(text_val)
                    current_len += text_units
                    idx += 1
                    continue

                remaining = limit - current_len
                if remaining <= 0:
                    flush_str = "".join(current_part) + closing
                    if _has_visible_content(flush_str):
                        parts.append(flush_str)
                    reopening = get_opening_tags(open_stack)
                    current_part = [reopening] if reopening else []
                    current_len = 0
                    continue

                max_raw_end = _prefix_end_for_utf16_limit(text_val, remaining)
                if max_raw_end <= 0:
                    if current_len > 0:
                        flush_str = "".join(current_part) + closing
                        if _has_visible_content(flush_str):
                            parts.append(flush_str)
                        reopening = get_opening_tags(open_stack)
                        current_part = [reopening] if reopening else []
                        current_len = 0
                        continue
                    cut_idx = _first_atomic_unit_end(text_val)
                else:
                    candidate_slice = text_val[:max_raw_end]
                    natural_cut = _find_natural_cut(candidate_slice)
                    cut_idx = natural_cut if natural_cut is not None else max_raw_end
                    cut_idx = max(1, min(cut_idx, max_raw_end))

                head = text_val[:cut_idx]
                tail = text_val[cut_idx:]

                if head:
                    current_part.append(head)
                flush_str = "".join(current_part) + closing
                if _has_visible_content(flush_str):
                    parts.append(flush_str)

                reopening = get_opening_tags(open_stack)
                current_part = [reopening] if reopening else []
                current_len = 0

                if tail:
                    tokens[idx] = ("text", tail)
                else:
                    idx += 1
                continue

            if current_len + len(text_val) + len(closing) <= limit:
                current_part.append(text_val)
                current_len += len(text_val)
                idx += 1
            else:
                max_content_len = limit - current_len - len(closing)
                if max_content_len <= 0:
                    reopening = get_opening_tags(open_stack)
                    if current_len > len(reopening):
                        flush_str = "".join(current_part) + closing
                        if _has_visible_content(flush_str):
                            parts.append(flush_str)
                        current_part = [reopening] if reopening else []
                        current_len = len(reopening)
                        continue

                    # Progress guard: open_stack wrapper alone fills/exceeds limit -> drop wrapper on new chunk
                    flush_str = "".join(current_part) + closing
                    if _has_visible_content(flush_str):
                        parts.append(flush_str)
                    open_stack = []
                    current_part = []
                    current_len = 0
                    continue

                candidate_slice = text_val[:max_content_len]
                natural_cut = _find_natural_cut(candidate_slice)
                if natural_cut is not None:
                    cut_idx = natural_cut
                else:
                    # Hard character boundary (without splitting an HTML entity)
                    cut_idx = _adjust_cut_for_html_entity(text_val, max_content_len)

                cut_idx = max(1, min(cut_idx, max_content_len))
                head = text_val[:cut_idx]
                tail = text_val[cut_idx:]

                if head:
                    current_part.append(head)
                flush_str = "".join(current_part) + closing
                if _has_visible_content(flush_str):
                    parts.append(flush_str)

                reopening = get_opening_tags(open_stack)
                current_part = [reopening] if reopening else []
                current_len = len(reopening)

                if tail:
                    tokens[idx] = ("text", tail)
                else:
                    idx += 1

    if current_part:
        closing = get_closing_tags(open_stack)
        final_str = "".join(current_part) + closing
        if _has_visible_content(final_str):
            parts.append(final_str)

    return parts
