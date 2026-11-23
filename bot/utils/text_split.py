"""Smart message splitting with HTML tag awareness and balance preservation."""

import re

TAG_RE = re.compile(r"<(/?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^<>]*)?)>")
_INCOMPLETE_ENTITY_RE = re.compile(r"&(?:#x?[0-9a-fA-F]*|[a-zA-Z]*)$")


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


def smart_split(text: str, limit: int = 4000) -> list[str]:
    """Split text into chunks of length <= limit, preserving HTML tag balance.

    Splits progressively at paragraph (\\n\\n), newline (\\n), sentence/space,
    or hard character boundary as a last resort, never inside an HTML tag.
    Open tags at split boundaries are cleanly closed at the end of the chunk
    and re-opened at the start of the next chunk.
    """
    if not text or not text.strip():
        return []

    if len(text) <= limit:
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
            tag_len = len(raw_tag)

            temp_stack = list(open_stack)
            if is_closing:
                for i in range(len(temp_stack) - 1, -1, -1):
                    if temp_stack[i][0] == tag_name:
                        temp_stack.pop(i)
                        break
            else:
                temp_stack.append((tag_name, raw_tag))

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
                cut_idx = -1

                # 1. Paragraph break (\n\n)
                p_idx = candidate_slice.rfind("\n\n")
                if p_idx != -1:
                    cut_idx = p_idx + 2
                else:
                    # 2. Line break (\n)
                    l_idx = candidate_slice.rfind("\n")
                    if l_idx != -1:
                        cut_idx = l_idx + 1
                    else:
                        # 3. Sentence or word boundary
                        s_idx = max(
                            candidate_slice.rfind(". "),
                            candidate_slice.rfind("! "),
                            candidate_slice.rfind("؟ "),
                            candidate_slice.rfind("; "),
                            candidate_slice.rfind(" "),
                        )
                        if s_idx != -1:
                            cut_idx = s_idx + 1
                        else:
                            # 4. Hard character boundary (without splitting an HTML entity)
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
