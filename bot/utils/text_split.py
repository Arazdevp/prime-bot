"""Smart message splitting with HTML tag awareness and balance preservation."""

import re

TAG_RE = re.compile(r"<(/?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^<>]*)?)>")


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
                # Flush current part
                closing = get_closing_tags(open_stack)
                flush_str = "".join(current_part) + closing
                if flush_str.strip():
                    parts.append(flush_str)
                reopening = get_opening_tags(open_stack)
                current_part = [reopening] if reopening else []
                current_len = len(reopening)
                # Next loop iteration will retry placing this tag on the fresh part
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
                    flush_str = "".join(current_part) + closing
                    if flush_str.strip():
                        parts.append(flush_str)
                    reopening = get_opening_tags(open_stack)
                    current_part = [reopening] if reopening else []
                    current_len = len(reopening)
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
                            # 4. Hard character boundary
                            cut_idx = max_content_len

                head = text_val[:cut_idx]
                tail = text_val[cut_idx:]

                if head:
                    current_part.append(head)
                flush_str = "".join(current_part) + closing
                if flush_str.strip():
                    parts.append(flush_str)

                reopening = get_opening_tags(open_stack)
                current_part = [reopening] if reopening else []
                current_len = len(reopening)

                tokens[idx] = ("text", tail)

    if current_part:
        closing = get_closing_tags(open_stack)
        final_str = "".join(current_part) + closing
        if final_str.strip():
            parts.append(final_str)

    return parts
