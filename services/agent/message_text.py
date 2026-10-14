"""Reading the text out of a model message.

A model message's `content` is not always a string. Gemini answers with content
blocks, so it arrives as a list shaped like `[{'type': 'text', 'text': '…'}]`, and a
caller that assumes a string gets the repr of that list instead of the prose.

`final_text` has unwrapped that since the first empty-answer bug, and the unwrapping
lived inside it. A second caller appeared when the streaming paths began announcing
prose chunk by chunk (`graph._streamed_turn`, `deterministic._formatting_turn`): each
chunk is shaped the same way as the whole message, so the naive `str(chunk.content)`
put `[{'type': 'text', 'text': …}]` on the wire. Measured live against `/ehr` on
2026-10-14 — 6,116 announced characters against a 1,021-character answer, every one of
them envelope. That is why this is a module rather than a second inline copy: the
rule has two callers and had already drifted once.
"""

from __future__ import annotations

from typing import Any


def text_of(content: Any) -> str:
    """The text in a message's `content`, whether it is a string or blocks.

    Blocks that carry no text — a function call, a thought signature, an image —
    contribute nothing. A list entry that is not a dict is stringified rather than
    dropped, which is what the original unwrapping did and is what keeps a plain list
    of strings working.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""
