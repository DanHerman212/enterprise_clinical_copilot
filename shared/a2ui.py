"""Walk an A2UI envelope for the component names it renders.

The surface is A2UI *messages*, not a flat list: components live under
`updateComponents.components`, with the root a `Column` whose children carry the card.
Walking generically means a surface gaining a wrapper does not silently report zero
components to an assertion looking for one.

There were two copies of this before 2026-10-29 — one in the retired collector, one in
`evaluation/platform/a2ui.py` whose docstring said it existed to avoid the import. The
second copy is now this module, and both callers read it here.
"""

from __future__ import annotations

from typing import Any


def components(node: Any, out: set[str] | None = None) -> list[str]:
    """Every component name in one A2UI payload node, sorted and deduplicated."""
    if out is None:
        out = set()
    if isinstance(node, dict):
        name = node.get("component")
        if isinstance(name, str):
            out.add(name)
        for value in node.values():
            components(value, out)
    elif isinstance(node, list):
        for item in node:
            components(item, out)
    return sorted(out)
