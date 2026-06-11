"""Helper: rearrange [{type:image,...}*, {type:text,text:T}] content into
interleaved form when T contains `<image N>` placeholders.

Upstream simple-class wrappers (glm4v, OpenAICompatible) build the user-message
content as images-first-then-text or text-first-then-images, ignoring positional
placeholders our prompt builder now emits for insdet/insdet_video. This helper
lets a plugin subclass intercept the messages just before they're consumed and
re-shape the content array so each `<image N>` site gets the matching image.
"""

import re

_IMAGE_PLACEHOLDER_RE = re.compile(r"<image (\d+)>")


def _interleave_one_content(content):
    """Rearrange a single user-message `content` list. Returns the (possibly
    new) content list. No-op when no `<image N>` placeholder is found."""
    text_items = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
    image_items = [c for c in content if isinstance(c, dict) and c.get("type") in ("image", "image_url")]
    if not text_items or not image_items:
        return content

    text_item = text_items[-1]
    text = text_item.get("text", "")
    if not _IMAGE_PLACEHOLDER_RE.search(text):
        return content

    parts = _IMAGE_PLACEHOLDER_RE.split(text)
    new_content = []
    if parts[0].strip():
        new_content.append({**text_item, "text": parts[0]})
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1
        if 0 <= idx < len(image_items):
            new_content.append(image_items[idx])
        if i + 1 < len(parts) and parts[i + 1].strip():
            new_content.append({**text_item, "text": parts[i + 1]})
    return new_content


def interleave_chat_messages(messages):
    """messages is a list of message dicts (one conversation). Rearrange the
    content of each user message that has `<image N>` placeholders. Modifies a
    copy; original is untouched."""
    out = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            out.append({**msg, "content": _interleave_one_content(msg["content"])})
        else:
            out.append(msg)
    return out


def interleave_batched_messages(batched_messages):
    """batched_messages is a list of conversations (each a list of message
    dicts). Apply interleave_chat_messages per conversation."""
    return [interleave_chat_messages(msgs) for msgs in batched_messages]
